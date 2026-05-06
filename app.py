"""
MarketMind — Web Frontend with Profile Support
Run with:  streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import io
import json
import re
import streamlit as st
import pandas as pd
from datetime import date, datetime
import yfinance as yf
import pdfplumber

# ── Load API keys ─────────────────────────────────────────────────────────────
# On Streamlit Cloud: secrets come from the dashboard (Settings → Secrets).
# Locally: falls back to your .env file via python-dotenv.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_SECRET_KEYS = ["ANTHROPIC_API_KEY", "FINNHUB_API_KEY", "ALPHA_VANTAGE_API_KEY"]
try:
    for _key in _SECRET_KEYS:
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass

import marketmind as mm

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MarketMind",
    page_icon="📈",
    layout="wide",
)

PROFILES_DIR = os.path.join(os.path.dirname(__file__), "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)


# ── Profile helpers ───────────────────────────────────────────────────────────

def profile_path(name: str) -> str:
    return os.path.join(PROFILES_DIR, f"portfolio_{name.lower()}.json")


def list_profiles() -> list[str]:
    files = [f for f in os.listdir(PROFILES_DIR) if f.startswith("portfolio_") and f.endswith(".json")]
    return [f.replace("portfolio_", "").replace(".json", "").capitalize() for f in sorted(files)]


def load_profile(name: str) -> dict:
    path = profile_path(name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_profile(name: str, state: dict) -> None:
    with open(profile_path(name), "w") as f:
        json.dump(state, f, indent=2)


def create_profile(name: str, capital: float, risk_level: str, tax_bracket_pct: float = 22.0) -> dict:
    tp = {"conservative": 3.0, "moderate": 5.0, "aggressive": 8.0}[risk_level]
    sl = {"conservative": 1.5, "moderate": 2.0, "aggressive": 3.0}[risk_level]
    state = {
        "profile_name":          name,
        "initial_investment":    capital,
        "cash_available":        capital,
        "total_portfolio_value": capital,
        "start_date":            date.today().isoformat(),
        "end_date":              (date.today().replace(year=date.today().year + 1)).isoformat(),
        "risk_level":            risk_level,
        "tax_bracket_pct":       tax_bracket_pct,
        "take_profit_pct":       tp,
        "stop_loss_pct":         sl,
        "holdings":              {},
        "transaction_history":   [],
        "realized_gains":        0.0,
        "unrealized_gains":      0.0,
        "total_return_pct":      0.0,
        "robinhood_imported":    False,
    }
    save_profile(name, state)
    return state


# ── Robinhood data parsers (CSV + PDF) ───────────────────────────────────────

def _build_holdings_from_rows(rows: list[dict]) -> dict:
    """
    Given a list of dicts with keys: action, ticker, qty, price, date
    reconstruct net holdings and return summary dict.
    """
    holdings: dict[str, dict] = {}
    transactions: list[dict]  = []
    total_invested = 0.0
    total_proceeds = 0.0
    buy_count = sell_count = 0

    for r in rows:
        action = r["action"]
        ticker = r["ticker"]
        qty    = r["qty"]
        price  = r["price"]
        dt     = r["date"]

        if action == "BUY":
            buy_count      += 1
            total_invested += qty * price
            if ticker not in holdings:
                holdings[ticker] = {"shares": 0.0, "avg_buy_price": 0.0, "buy_date": dt}
            old_cost   = holdings[ticker]["shares"] * holdings[ticker]["avg_buy_price"]
            new_shares = holdings[ticker]["shares"] + qty
            holdings[ticker]["avg_buy_price"] = (old_cost + qty * price) / new_shares
            holdings[ticker]["shares"]        = round(new_shares, 6)

        elif action == "SELL":
            sell_count     += 1
            total_proceeds += qty * price
            if ticker in holdings:
                holdings[ticker]["shares"] = round(holdings[ticker]["shares"] - qty, 6)
                if holdings[ticker]["shares"] <= 0.001:
                    del holdings[ticker]

        transactions.append({
            "date": dt, "action": action,
            "ticker": ticker, "shares": qty, "price": price,
            "total": round(qty * price, 2),
        })

    return {
        "holdings":       holdings,
        "transactions":   transactions,
        "total_invested": round(total_invested, 2),
        "total_proceeds": round(total_proceeds, 2),
        "buy_count":      buy_count,
        "sell_count":     sell_count,
        "tickers_found":  sorted(set(holdings.keys())),
    }


def parse_robinhood_csv(uploaded_file) -> dict:
    """
    Parse a Robinhood account history CSV.
    Expected columns: Activity Date, Instrument, Trans Code, Quantity, Price, Amount
    """
    try:
        df = pd.read_csv(uploaded_file, on_bad_lines="skip", engine="python")
    except Exception as e:
        return {"error": f"Could not read CSV: {e}"}

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {"instrument", "trans_code", "quantity", "price"}
    missing  = required - set(df.columns)
    if missing:
        return {"error": f"Missing columns: {missing}. Expected Robinhood account history CSV."}

    for col in ["quantity", "price"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r"[$,()]", "", regex=True).str.strip(),
            errors="coerce",
        ).fillna(0.0)

    rows = []
    for _, row in df.iterrows():
        code   = str(row.get("trans_code", "")).strip().upper()
        ticker = str(row.get("instrument", "")).strip().upper()
        qty    = abs(float(row.get("quantity", 0)))
        price  = abs(float(row.get("price", 0)))
        dt     = str(row.get("activity_date", date.today().isoformat()))

        if code not in ("BUY", "SELL") or not ticker or qty == 0 or price == 0:
            continue
        rows.append({"action": code, "ticker": ticker, "qty": qty, "price": price, "date": dt})

    if not rows:
        return {"error": "No BUY/SELL transactions found in CSV."}

    return _build_holdings_from_rows(rows)


def parse_robinhood_pdf(uploaded_file) -> dict:
    """
    Parse a Robinhood account statement PDF.

    Robinhood PDFs contain a transaction table with rows like:
      Date | Description | Amount
    Description examples:
      "Buy  AAPL  5 shares at $169.89"
      "Sell TSLA  2 shares at $255.00"
      "AAPL  Buy    5  $169.89  $849.45"

    We also handle the newer Robinhood format where each column is separate.
    Falls back to full-text regex scan if table extraction yields nothing.
    """
    rows = []

    # Patterns to match transaction descriptions
    # Format A: "Buy AAPL 5 shares at $169.89"
    PAT_A = re.compile(
        r"(Buy|Sell)\s+([A-Z]{1,5})\s+([\d.]+)\s+shares?\s+at\s+\$?([\d,]+\.?\d*)",
        re.IGNORECASE,
    )
    # Format B: ticker then action then qty then price (table columns merged)
    # e.g. "AAPL Buy 5 $169.89"
    PAT_B = re.compile(
        r"([A-Z]{1,5})\s+(Buy|Sell)\s+([\d.]+)\s+\$?([\d,]+\.?\d*)",
        re.IGNORECASE,
    )
    # Format C: date prefix "MM/DD/YYYY Buy AAPL ..."
    PAT_C = re.compile(
        r"\d{2}/\d{2}/\d{4}\s+(Buy|Sell)\s+([A-Z]{1,5})\s+([\d.]+)\s+\$?([\d,]+\.?\d*)",
        re.IGNORECASE,
    )

    date_pat = re.compile(r"(\d{2}/\d{2}/\d{4})")

    try:
        pdf_bytes = uploaded_file.read()
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                # ── Try table extraction first ──
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if not row:
                            continue
                        cells = [str(c).strip() if c else "" for c in row]
                        text  = " ".join(cells)

                        # Try all patterns on the merged cell text
                        for pat, grp_action, grp_ticker, grp_qty, grp_price in [
                            (PAT_A, 1, 2, 3, 4),
                            (PAT_B, 2, 1, 3, 4),
                            (PAT_C, 1, 2, 3, 4),
                        ]:
                            m = pat.search(text)
                            if m:
                                action = m.group(grp_action).upper()
                                ticker = m.group(grp_ticker).upper()
                                qty    = float(m.group(grp_qty).replace(",", ""))
                                price  = float(m.group(grp_price).replace(",", ""))
                                # Try to find a date nearby in this row
                                dm = date_pat.search(text)
                                dt = dm.group(1) if dm else date.today().isoformat()
                                if qty > 0 and price > 0:
                                    rows.append({
                                        "action": action, "ticker": ticker,
                                        "qty": qty, "price": price, "date": dt,
                                    })
                                break

                # ── Fallback: scan raw page text ──
                if not rows:
                    text = page.extract_text() or ""
                    current_date = date.today().isoformat()
                    for line in text.split("\n"):
                        dm = date_pat.search(line)
                        if dm:
                            current_date = dm.group(1)
                        for pat, grp_action, grp_ticker, grp_qty, grp_price in [
                            (PAT_A, 1, 2, 3, 4),
                            (PAT_B, 2, 1, 3, 4),
                            (PAT_C, 1, 2, 3, 4),
                        ]:
                            m = pat.search(line)
                            if m:
                                action = m.group(grp_action).upper()
                                ticker = m.group(grp_ticker).upper()
                                qty    = float(m.group(grp_qty).replace(",", ""))
                                price  = float(m.group(grp_price).replace(",", ""))
                                if qty > 0 and price > 0:
                                    rows.append({
                                        "action": action, "ticker": ticker,
                                        "qty": qty, "price": price,
                                        "date": current_date,
                                    })
                                break

    except Exception as e:
        return {"error": f"Could not read PDF: {e}"}

    if not rows:
        return {
            "error": (
                "No BUY/SELL transactions found in PDF. "
                "Make sure this is a Robinhood account history statement. "
                "Try downloading the CSV export instead."
            )
        }

    # Deduplicate (same ticker/action/qty/price/date)
    seen = set()
    deduped = []
    for r in rows:
        key = (r["action"], r["ticker"], r["qty"], r["price"], r["date"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return _build_holdings_from_rows(deduped)


def parse_robinhood_file(uploaded_file) -> dict:
    """Route to CSV or PDF parser based on file extension."""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return parse_robinhood_pdf(uploaded_file)
    elif name.endswith(".csv"):
        return parse_robinhood_csv(uploaded_file)
    else:
        return {"error": "Unsupported file type. Please upload a .csv or .pdf file."}


def enrich_holdings_with_live_prices(holdings: dict) -> tuple[dict, float]:
    """Fetch current prices and compute unrealized P&L for each holding."""
    enriched       = {}
    total_mkt_val  = 0.0

    for ticker, pos in holdings.items():
        try:
            price = float(yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1])
        except Exception:
            price = pos.get("avg_buy_price", 0)

        shares    = pos.get("shares", 0)
        avg_buy   = pos.get("avg_buy_price", 0)
        mkt_val   = shares * price
        cost      = shares * avg_buy
        unreal    = mkt_val - cost
        unreal_pct = (unreal / cost * 100) if cost else 0

        enriched[ticker] = {
            **pos,
            "current_price":   round(price, 2),
            "market_value":    round(mkt_val, 2),
            "cost_basis":      round(cost, 2),
            "unrealized_pnl":  round(unreal, 2),
            "unrealized_pct":  round(unreal_pct, 2),
        }
        total_mkt_val += mkt_val

    return enriched, round(total_mkt_val, 2)


def signal_colour(sig: str) -> str:
    sig = sig.upper()
    if "STRONG BUY" in sig or "ENTER" in sig: return "🟢"
    if "BUY" in sig:                          return "🟢"
    if "AVOID" in sig or "STOP" in sig:       return "🔴"
    if "WAIT" in sig or "HOLD" in sig:        return "🟡"
    return "⚪"


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📈 MarketMind")
st.sidebar.caption("Free swing trading signals — yfinance powered")
st.sidebar.markdown("---")

# Profile selector
st.sidebar.subheader("👤 Profile")
existing = list_profiles()

with st.sidebar.expander("＋ Create new profile"):
    new_name    = st.text_input("Profile name", placeholder="e.g. Hakapi")
    new_capital = st.number_input("Starting capital ($)", value=10000, step=500, min_value=100)
    new_risk    = st.selectbox("Risk level", ["moderate", "conservative", "aggressive"], key="new_risk")
    new_tax     = st.selectbox(
        "Tax bracket (marginal rate)",
        [10, 12, 22, 24, 32, 35, 37],
        index=2,
        key="new_tax",
        help="Your federal income tax bracket — used to estimate capital gains tax on recommendations.",
    )
    if st.button("Create profile"):
        if new_name.strip():
            create_profile(new_name.strip().capitalize(), float(new_capital), new_risk, float(new_tax))
            st.success(f"Profile '{new_name.capitalize()}' created!")
            st.rerun()
        else:
            st.warning("Enter a profile name.")

if not existing:
    st.sidebar.warning("No profiles yet. Create one above.")
    st.stop()

active_profile = st.sidebar.selectbox("Active profile", existing)
profile_state  = load_profile(active_profile)

# Sync mm config from profile
mm.PORTFOLIO_STATE_FILE = profile_path(active_profile)
mm.TRADING_CONFIG["starting_capital"]  = profile_state.get("initial_investment", 10000)
mm.TRADING_CONFIG["risk_level"]        = profile_state.get("risk_level", "moderate")
mm.TRADING_CONFIG["take_profit_pct"]   = profile_state.get("take_profit_pct", 5.0)
mm.TRADING_STATE_FILE = profile_path(active_profile)

tp = profile_state.get("take_profit_pct", 5.0)
sl = profile_state.get("stop_loss_pct",   2.0)
mm.TRADING_CONFIG["take_profit_pct"] = tp
mm.TRADING_CONFIG["stop_loss_pct"]   = sl

st.sidebar.markdown("---")
tax_bracket = profile_state.get("tax_bracket_pct", 22.0)
st.sidebar.caption(f"**Risk:** {profile_state.get('risk_level','—').title()}")
st.sidebar.caption(f"**Take-profit:** +{tp}%  |  **Stop-loss:** -{sl}%")
st.sidebar.caption(f"**Tax bracket:** {int(tax_bracket)}%")
st.sidebar.caption(f"**Robinhood data:** {'✅ Imported' if profile_state.get('robinhood_imported') else '⬜ Not imported'}")

# Mode nav
st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "Mode",
    ["Portfolio", "Daily Trade Scan", "Stock Forecast", "Top 10 Snapshot", "Backtest"],
)

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ Educational purposes only. Not financial advice.")


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO — upload Robinhood data + view holdings
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Portfolio":
    st.title(f"💼 Portfolio — {active_profile}")

    # ── Robinhood upload ──
    st.subheader("📥 Import Robinhood Data")
    st.caption("Export from Robinhood → Account → Statements & History → Download CSV")

    uploaded = st.file_uploader(
        "Upload your Robinhood account history (CSV or PDF)",
        type=["csv", "pdf"],
        key=f"upload_{active_profile}",
    )

    if uploaded:
        with st.spinner("Parsing Robinhood data..."):
            result = parse_robinhood_file(uploaded)

        if result.get("error"):
            st.error(f"Import failed: {result['error']}")
        else:
            st.success(
                f"Parsed {result['buy_count']} buys + {result['sell_count']} sells  "
                f"|  {len(result['tickers_found'])} active holdings: {', '.join(result['tickers_found'])}"
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Invested",  f"${result['total_invested']:,.2f}")
            col2.metric("Total Proceeds",  f"${result['total_proceeds']:,.2f}")
            col3.metric("Active Holdings", len(result["tickers_found"]))

            if st.button("✅ Import into MarketMind", type="primary"):
                with st.spinner("Fetching live prices..."):
                    enriched, mkt_val = enrich_holdings_with_live_prices(result["holdings"])

                cash_remaining = max(0, result["total_invested"] - sum(
                    p["shares"] * p["avg_buy_price"] for p in result["holdings"].values()
                ))

                profile_state["holdings"]            = enriched
                profile_state["transaction_history"] = result["transactions"][-50:]
                profile_state["cash_available"]      = round(cash_remaining, 2)
                profile_state["total_portfolio_value"] = round(mkt_val + cash_remaining, 2)
                profile_state["initial_investment"]  = result["total_invested"]
                profile_state["robinhood_imported"]  = True
                profile_state["import_date"]         = date.today().isoformat()

                save_profile(active_profile, profile_state)
                st.success("Data imported! Scroll down to see your portfolio.")
                st.rerun()

    st.markdown("---")

    # ── Holdings ──
    holdings = profile_state.get("holdings", {})
    if not holdings:
        st.info("No holdings yet. Import your Robinhood data above, or run the Daily Trade Scan to get started.")
    else:
        st.subheader("📊 Current Holdings")

        with st.spinner("Refreshing live prices..."):
            enriched, mkt_val = enrich_holdings_with_live_prices(holdings)

        # Update portfolio value
        cash          = profile_state.get("cash_available", 0)
        total_val     = round(mkt_val + cash, 2)
        initial       = profile_state.get("initial_investment", total_val)
        total_ret_pct = ((total_val - initial) / initial * 100) if initial else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio Value",  f"${total_val:,.2f}")
        c2.metric("Cash",             f"${cash:,.2f}")
        c3.metric("Invested Value",   f"${mkt_val:,.2f}")
        c4.metric("Total Return",     f"{total_ret_pct:+.2f}%")

        # Compute tax implications for all holdings
        tax_data = mm.compute_tax_implications(enriched, tax_bracket)

        rows = []
        for ticker, pos in enriched.items():
            pct  = pos.get("unrealized_pct", 0)
            td   = tax_data.get(ticker, {})
            days = td.get("holding_days")
            gtype = td.get("gain_type", "Unknown")
            dtlt  = td.get("days_to_lt")
            est_tax = td.get("est_tax_if_sold_now", 0)

            hold_str = f"{days}d" if days is not None else "?"
            if gtype == "Short-term" and dtlt and dtlt <= 60:
                hold_str += f" ⚠️ {dtlt}d to LT"
            rows.append({
                "Ticker":         ticker,
                "Shares":         pos.get("shares", 0),
                "Avg Buy":        f"${pos.get('avg_buy_price', 0):.2f}",
                "Current Price":  f"${pos.get('current_price', 0):.2f}",
                "Market Value":   f"${pos.get('market_value', 0):,.2f}",
                "Unrealized P&L": f"${pos.get('unrealized_pnl', 0):+,.2f}",
                "Return %":       f"{pct:+.2f}%",
                "Held":           hold_str,
                "Gain Type":      gtype,
                "Est Tax if Sold": f"${est_tax:,.2f}" if est_tax else "—",
                "Status":         "🟢 Profit" if pct > 0 else "🔴 Loss" if pct < 0 else "⚪ Flat",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Tax summary
        harvest_candidates = [t for t, td in tax_data.items() if td.get("tax_loss_harvest")]
        near_lt_candidates = [
            t for t, td in tax_data.items()
            if td.get("gain_type") == "Short-term" and td.get("days_to_lt") is not None and td["days_to_lt"] <= 60
        ]
        total_est_tax = sum(td.get("est_tax_if_sold_now", 0) for td in tax_data.values())

        with st.expander(f"💰 Tax Summary (bracket: {int(tax_bracket)}%)"):
            tx1, tx2, tx3 = st.columns(3)
            tx1.metric("Est. Tax if All Sold Today", f"${total_est_tax:,.2f}")
            tx2.metric("Tax-Loss Harvest Candidates", len(harvest_candidates))
            tx3.metric("Positions Near LT Threshold", len(near_lt_candidates))

            if harvest_candidates:
                st.warning(
                    f"**Tax-loss harvest opportunities:** {', '.join(harvest_candidates)}  \n"
                    "Selling these losers offsets gains elsewhere. Note: wash-sale rule — "
                    "don't buy back within 30 days."
                )
            if near_lt_candidates:
                for t in near_lt_candidates:
                    dtlt = tax_data[t]["days_to_lt"]
                    gain = tax_data[t]["unrealized_gain"]
                    st.info(
                        f"**{t}** — {dtlt} days until long-term threshold. "
                        f"Unrealized gain: ${gain:+,.2f}. "
                        f"Waiting saves ~${gain * (tax_bracket/100 - 0.15):,.2f} in taxes vs selling now."
                    )
            st.caption("Estimates assume federal rates only. Consult a tax advisor for state taxes and your full situation.")

        # ── Engine check on holdings ──
        st.markdown("---")
        st.subheader("🔎 Engine Check on Your Holdings")
        st.caption("Hold or exit each position? Includes tax-aware notes.")

        if st.button("▶  Run Engine Check", type="primary"):
            results = []
            prog = st.progress(0)
            tickers = list(enriched.keys())

            for i, ticker in enumerate(tickers):
                prog.progress((i + 1) / len(tickers), text=f"Checking {ticker}...")
                pos   = enriched[ticker]
                tech  = mm.get_technical_indicators(ticker)
                pct   = pos.get("unrealized_pct", 0)
                td    = tax_data.get(ticker, {})

                if tech.get("error"):
                    action = "⚠️ No data"
                    reason = tech["error"]
                    tax_note = ""
                else:
                    rsi      = tech.get("rsi_14d", 50)
                    macd_sig = tech.get("macd_signal", "neutral")
                    ema200   = tech.get("ema_200d")
                    cur      = tech.get("current_price", pos["current_price"])

                    if pct >= tp:
                        action = "🎯 EXIT — Take-profit hit"
                        reason = f"Up {pct:.1f}% — lock in gains"
                    elif pct <= -sl:
                        action = "🛑 EXIT — Stop-loss hit"
                        reason = f"Down {pct:.1f}% — cut losses"
                    elif rsi > 75:
                        action = "⚠️ CONSIDER EXITING"
                        reason = f"RSI={rsi} — overbought, pullback likely"
                    elif ema200 and cur < ema200:
                        action = "⚠️ CONSIDER EXITING"
                        reason = "Price below 200-day EMA — long-term trend broken"
                    elif macd_sig == "bearish crossover" and pct > 0:
                        action = "⚠️ WATCH CLOSELY"
                        reason = "MACD turning bearish — momentum fading"
                    else:
                        action = "✅ HOLD"
                        reason = f"RSI={rsi}, MACD={macd_sig} — no exit signal"

                    # Tax note
                    dtlt      = td.get("days_to_lt", 0)
                    gain_type = td.get("gain_type", "")
                    est_tax   = td.get("est_tax_if_sold_now", 0)
                    if td.get("tax_loss_harvest") and "EXIT" in action:
                        tax_note = f"Loss = harvest opportunity. Offset gains elsewhere."
                    elif gain_type == "Short-term" and dtlt and dtlt <= 60 and pct > 0:
                        tax_note = f"Wait {dtlt}d for long-term rate — saves ~${est_tax * 0.3:,.0f}"
                    elif gain_type == "Long-term" and pct > 0:
                        tax_note = f"LT rate applies — est tax ${est_tax:,.2f}"
                    elif gain_type == "Short-term" and pct > 0:
                        tax_note = f"ST rate {int(tax_bracket)}% — est tax ${est_tax:,.2f}"
                    else:
                        tax_note = ""

                results.append({
                    "Ticker":      ticker,
                    "Return %":    f"{pct:+.2f}%",
                    "RSI":         tech.get("rsi_14d", "—") if not tech.get("error") else "—",
                    "MACD":        tech.get("macd_signal", "—") if not tech.get("error") else "—",
                    "Action":      action,
                    "Reason":      reason,
                    "Tax Note":    tax_note,
                })

            prog.empty()
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

        # ── Transaction history ──
        txns = profile_state.get("transaction_history", [])
        if txns:
            with st.expander(f"Transaction history ({len(txns)} records)"):
                st.dataframe(pd.DataFrame(txns[:50]), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# DAILY TRADE SCAN
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Daily Trade Scan":
    st.title(f"📊 Daily Trade Scan — {active_profile}")
    st.caption(f"Scans {len(mm.TRADING_UNIVERSE)} stocks for swing trade setups — {date.today().strftime('%A, %B %d %Y')}")

    col1, col2 = st.columns([2, 1])
    with col1:
        run = st.button("▶  Run Scan Now", type="primary", use_container_width=True)
    with col2:
        st.caption("Takes ~30 seconds")

    if run:
        with st.spinner("Scanning the market..."):
            report = mm.run_agent()

        state = mm.get_portfolio_state()
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Capital",     f"${state.get('total_portfolio_value', 0):,.2f}")
        c2.metric("Cash Available",    f"${state.get('cash_available', 0):,.2f}")
        c3.metric("Open Trades",       f"{len(state.get('holdings', {}))}/{mm.TRADING_CONFIG['max_open_trades']}")
        c4.metric("Return Since Start",f"{state.get('total_return_pct', 0):+.2f}%")

        st.markdown("---")
        st.subheader("Market Conditions")
        benchmark = mm.get_benchmark_comparison()
        bc1, bc2, bc3 = st.columns(3)
        for i, (name, data) in enumerate(benchmark.items()):
            if not data.get("error"):
                wk = data.get("weekly_return_pct", 0)
                [bc1, bc2, bc3][i].metric(name, f"{data.get('current_value','N/A')}", f"{wk:+.2f}% this week")

        st.markdown("---")
        st.subheader("Trade Signals")

        lines = report.split("\n")
        entries, passed_stocks = [], []
        current = None

        for line in lines:
            line = line.strip()
            if line.startswith("✅ ENTER"):
                if current:
                    entries.append(current)
                current = {"ticker": line.replace("✅ ENTER —", "").strip()}
            elif current:
                for field in ["Setup", "Entry price", "Take-profit", "Stop-loss",
                              "Position size", "Max hold", "Why", "RSI"]:
                    if line.startswith(field + ":"):
                        current[field] = line.split(":", 1)[1].strip()
            if "PASSED — NO TRADE" in line:
                if current:
                    entries.append(current)
                current = None
        if current:
            entries.append(current)

        in_passed = False
        for line in lines:
            line = line.strip()
            if "PASSED — NO TRADE" in line:
                in_passed = True
                continue
            if "BEGINNER TIP" in line:
                in_passed = False
            if in_passed and line and not line.startswith("##"):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    passed_stocks.append({"Ticker": parts[0], "Reason": parts[1]})

        if entries:
            st.success(f"**{len(entries)} trade setup(s) found today**")
            for e in entries:
                with st.expander(f"🟢 ENTER {e.get('ticker','')}  —  {e.get('Setup','')}  |  Entry: {e.get('Entry price','')}"):
                    ec1, ec2, ec3 = st.columns(3)
                    ec1.metric("Entry Price", e.get("Entry price", "—"))
                    ec2.metric("Take-Profit", e.get("Take-profit", "—").split("←")[0].strip())
                    ec3.metric("Stop-Loss",   e.get("Stop-loss",   "—").split("←")[0].strip())
                    st.caption(f"**Why:** {e.get('Why','—')}")
                    st.caption(f"**Size:** {e.get('Position size','—')}  |  **Hold:** {e.get('Max hold','—')}")
                    st.caption(f"**Indicators:** {e.get('RSI','—')}")
        else:
            st.warning("No qualifying setups found today. Sitting tight.")

        if passed_stocks:
            with st.expander(f"Stocks scanned but skipped ({len(passed_stocks)})"):
                st.dataframe(pd.DataFrame(passed_stocks), use_container_width=True, hide_index=True)

        with st.expander("View full report text"):
            st.code(report, language=None)

        path = mm._save_daily_report(report)
        st.caption(f"Saved: {path}")

        # ── New Opportunities ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💡 New Opportunities")
        st.caption(
            "Stocks beyond your current holdings with bullish signals — "
            "scanned across 11 sectors (~165 stocks)."
        )

        held_tickers = list(profile_state.get("holdings", {}).keys())
        opp_col1, opp_col2 = st.columns([2, 1])
        with opp_col1:
            run_opps = st.button("▶  Find Opportunities", key="btn_opps")
        with opp_col2:
            st.caption("Takes ~2 min (165 stocks)")

        if run_opps:
            with st.spinner("Scanning 165 stocks across 11 sectors..."):
                opps = mm.find_new_opportunities(held_tickers, n=10)

            if opps:
                st.success(f"Found {len(opps)} opportunity candidates")
                opp_rows = []
                for o in opps:
                    opp_rows.append({
                        "Ticker":      o["ticker"],
                        "Sector":      o["sector"],
                        "Signal":      o["signal"],
                        "Score":       o["bull_score"],
                        "Price":       f"${o['price']:.2f}",
                        "RSI":         o["rsi"],
                        "MACD":        o["macd"],
                        "Volume":      f"{o['volume']:.1f}x",
                        "Sentiment":   o["sentiment"],
                        "Take-Profit": f"${o['take_profit']:.2f}",
                        "Stop-Loss":   f"${o['stop_loss']:.2f}",
                        "Why":         o["reason"],
                    })
                st.dataframe(pd.DataFrame(opp_rows), use_container_width=True, hide_index=True)

                # Tax note on new buys
                st.info(
                    "**Tax note on new buys:** Any position you open today starts the holding-period clock. "
                    f"If sold within 1 year, gains are taxed at your short-term rate ({int(tax_bracket)}%). "
                    "Holding ≥1 year qualifies for lower long-term rates (0/15/20%)."
                )
            else:
                st.warning("No high-confidence opportunities found today. Market conditions may be unfavorable.")

        # ── Portfolio Balance ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🗺️ Portfolio Balance")
        st.caption("Sector breakdown of your holdings and diversification gaps.")

        if held_tickers:
            run_balance = st.button("▶  Analyse Portfolio Balance", key="btn_balance")
            if run_balance:
                with st.spinner("Fetching sector data..."):
                    analysis = mm.get_portfolio_sector_analysis(held_tickers)

                if analysis.get("error"):
                    st.warning(analysis["error"])
                else:
                    weights = analysis["sector_weights"]
                    bal_rows = []
                    for sector, info in sorted(weights.items(), key=lambda x: -x[1]["pct"]):
                        bal_rows.append({
                            "Sector":   sector,
                            "Tickers":  ", ".join(info["tickers"]),
                            "Count":    info["count"],
                            "Weight %": f"{info['pct']:.1f}%",
                            "Status":   "⚠️ Concentrated" if info["pct"] > 35 else "✅ OK",
                        })
                    st.dataframe(pd.DataFrame(bal_rows), use_container_width=True, hide_index=True)

                    for note in analysis["suggestions"]:
                        if "Reduce" in note or "No exposure" in note:
                            st.warning(note)
                        else:
                            st.success(note)

                    if analysis["underweight"]:
                        st.caption(f"Sectors with no exposure: {', '.join(analysis['underweight'])}")
        else:
            st.info("Import your Robinhood data first to see your portfolio balance.")


# ══════════════════════════════════════════════════════════════════════════════
# STOCK FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Stock Forecast":
    st.title(f"🔍 Stock Forecast — {active_profile}")

    tab_recs, tab_analyse = st.tabs(["💡 Stock Recommendations", "🔎 Analyse a Stock"])

    # ── Tab 1: Stock Recommendations ─────────────────────────────────────────
    with tab_recs:
        st.subheader("💡 Stock Recommendations")
        st.caption("Scans ~165 stocks across 11 sectors for bullish setups you don't already hold.")

        held_tickers = list(profile_state.get("holdings", {}).keys())

        rec_col1, rec_col2 = st.columns([2, 1])
        with rec_col1:
            run_recs = st.button("▶  Find Stocks to Buy", type="primary", use_container_width=True)
        with rec_col2:
            st.caption("Takes ~2 min")

        if run_recs:
            with st.spinner("Scanning 165 stocks across 11 sectors..."):
                opps = mm.find_new_opportunities(held_tickers, n=15)

            if opps:
                st.success(f"Found {len(opps)} buy candidates")
                opp_rows = []
                for o in opps:
                    opp_rows.append({
                        "Ticker":      o["ticker"],
                        "Sector":      o["sector"],
                        "Signal":      o["signal"],
                        "Score":       o["bull_score"],
                        "Price":       f"${o['price']:.2f}",
                        "RSI":         o["rsi"],
                        "MACD":        o["macd"],
                        "Volume":      f"{o['volume']:.1f}x",
                        "Sentiment":   o["sentiment"],
                        "Take-Profit": f"${o['take_profit']:.2f}",
                        "Stop-Loss":   f"${o['stop_loss']:.2f}",
                        "Why":         o["reason"],
                    })
                st.dataframe(pd.DataFrame(opp_rows), use_container_width=True, hide_index=True)
                st.info(
                    f"**Tax note on new buys:** Positions opened today start the 1-year holding clock. "
                    f"Short-term rate: {int(tax_bracket)}%. Holding ≥1 year qualifies for lower long-term rates."
                )
            else:
                st.warning("No high-confidence opportunities found today. Market conditions may be unfavorable.")

    # ── Tab 2: Analyse a specific stock ──────────────────────────────────────
    with tab_analyse:
        st.subheader("🔎 Analyse a Specific Stock")
        st.caption("Enter any US ticker for a full technical + fundamental analysis")

        col1, col2 = st.columns([3, 1])
        with col1:
            ticker_input = st.text_input("Ticker symbol", placeholder="e.g. AAPL, TSLA, NVDA").upper().strip()
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_forecast = st.button("▶  Analyse", type="primary", use_container_width=True)

        if run_forecast and ticker_input:
            with st.spinner(f"Analysing {ticker_input}..."):
                report = mm.run_stock_forecast(ticker_input)

            verdict, conf = "—", "—"
            for line in report.split("\n"):
                line = line.strip()
                if line.startswith("VERDICT:"):    verdict = line.replace("VERDICT:", "").strip()
                if line.startswith("CONFIDENCE:"): conf    = line.replace("CONFIDENCE:", "").strip()

            st.markdown("---")
            st.markdown(f"## {signal_colour(verdict)}  {verdict}")
            st.caption(f"Confidence: {conf}")

            held_pos = profile_state.get("holdings", {}).get(ticker_input)
            if held_pos:
                td = mm.compute_tax_implications({ticker_input: held_pos}, tax_bracket).get(ticker_input, {})
                gtype = td.get("gain_type", "Unknown")
                dtlt  = td.get("days_to_lt", 0)
                est   = td.get("est_tax_if_sold_now", 0)
                if gtype == "Short-term" and dtlt and dtlt <= 60 and verdict in ("STRONG BUY", "BUY", "WAIT"):
                    st.warning(
                        f"**Tax alert:** You already hold {ticker_input} with {dtlt} days until long-term threshold. "
                        f"If you sell now, est. tax = ${est:,.2f} at short-term rate ({int(tax_bracket)}%). "
                        f"Waiting {dtlt} days could save ~${est * 0.3:,.0f}."
                    )
                elif gtype == "Long-term":
                    st.info(f"You already hold {ticker_input} — long-term rate applies. Est. tax if sold: ${est:,.2f}.")
            elif verdict in ("STRONG BUY", "BUY"):
                st.info(
                    f"**Tax note (new position):** Any shares bought today start the 1-year clock. "
                    f"Short-term rate: {int(tax_bracket)}%. Long-term rate: {'0%' if tax_bracket <= 12 else '15%' if tax_bracket <= 35 else '20%'}. "
                    "Plan your hold period accordingly."
                )

            tech = mm.get_technical_indicators(ticker_input)
            fund = mm.get_fundamentals(ticker_input)
            stw  = mm.get_stocktwits_sentiment(ticker_input)
            wkly = mm.get_weekly_return(ticker_input)

            st.markdown("---")
            st.subheader("Key Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Price",      f"${tech.get('current_price','N/A')}")
            m2.metric("7-Day",      f"{wkly.get('weekly_return_pct',0):+.2f}%" if not wkly.get("error") else "N/A")
            m3.metric("RSI",        tech.get("rsi_14d","N/A"), tech.get("rsi_signal",""))
            m4.metric("MACD",       tech.get("macd_signal","N/A"))
            m5.metric("Sentiment",  stw.get("sentiment_label","N/A") if not stw.get("error") else "N/A",
                      f"{stw.get('bullish_pct','?')}% bullish" if not stw.get("error") else "")

            t1, t2 = st.columns(2)
            with t1:
                st.subheader("Technical Signals")
                cur = tech.get("current_price", 0) or 0
                rows = [
                    {"Indicator": "RSI (14d)",       "Value": tech.get("rsi_14d","—"),     "Signal": tech.get("rsi_signal","—")},
                    {"Indicator": "MACD",            "Value": tech.get("macd_signal","—"),  "Signal": "bullish" if (tech.get("macd_histogram") or 0) > 0 else "bearish"},
                    {"Indicator": "Bollinger Bands", "Value": f"{tech.get('bb_price_position_pct','—')}th %ile", "Signal": tech.get("bb_signal","—")},
                    {"Indicator": "EMA 20d",         "Value": f"${tech.get('ema_20d','—')}", "Signal": "above ✓" if cur > (tech.get("ema_20d") or 0) else "below ✗"},
                    {"Indicator": "EMA 200d",        "Value": f"${tech.get('ema_200d','—')}", "Signal": "above ✓" if cur > (tech.get("ema_200d") or 0) else "below ✗"},
                    {"Indicator": "Volume",          "Value": f"{tech.get('volume_ratio','—')}x avg", "Signal": tech.get("volume_signal","—")},
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            with t2:
                st.subheader("Fundamentals")
                if not fund.get("error"):
                    mcap = fund.get("market_cap")
                    rows = [
                        {"Metric": "P/E Ratio",      "Value": fund.get("pe_ratio","—")},
                        {"Metric": "Forward P/E",    "Value": fund.get("forward_pe","—")},
                        {"Metric": "Profit Margin",  "Value": fund.get("profit_margin","—")},
                        {"Metric": "Analyst Rating", "Value": (fund.get("analyst_rating") or "—").replace("_"," ").title()},
                        {"Metric": "Price Target",   "Value": f"${fund.get('analyst_target_price','—')}"},
                        {"Metric": "Market Cap",     "Value": f"${mcap/1e9:.1f}B" if mcap else "—"},
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            with st.expander("View full report"):
                st.code(report, language=None)

            path = mm._save_forecast(ticker_input, report)
            st.caption(f"Saved: {path}")

        elif run_forecast:
            st.warning("Please enter a ticker symbol.")


# ══════════════════════════════════════════════════════════════════════════════
# TOP 10 SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Top 10 Snapshot":
    st.title("🏆 Top 10 US Stocks Snapshot")
    st.caption("Live prices, analyst targets, and 7-day rule-based predictions")

    if st.button("▶  Fetch Snapshot", type="primary"):
        TOP10 = [
            ("AAPL","Apple"), ("MSFT","Microsoft"), ("NVDA","NVIDIA"),
            ("AMZN","Amazon"), ("GOOGL","Alphabet"), ("META","Meta"),
            ("BRK-B","Berkshire"), ("LLY","Eli Lilly"), ("AVGO","Broadcom"), ("TSLA","Tesla"),
        ]
        rows = []
        prog = st.progress(0)
        for i, (ticker, name) in enumerate(TOP10):
            prog.progress((i + 1) / len(TOP10), text=f"Fetching {ticker}...")
            try:
                weekly = mm.get_weekly_return(ticker)
                fund   = mm.get_fundamentals(ticker)
                tech   = mm.get_technical_indicators(ticker)
                cur    = weekly.get("current_price")
                tgt    = fund.get("analyst_target_price")
                upside = round((tgt - cur) / cur * 100, 1) if cur and tgt else None
                pred   = mm._get_ai_predictions([{
                    "ticker": ticker, "name": name,
                    "current_price": cur, "technicals": tech, "fundamentals": fund,
                }]).get(ticker, {})
                rows.append({
                    "Ticker":    ticker, "Company":  name,
                    "Price":     f"${cur:,.2f}" if cur else "N/A",
                    "7-Day":     f"{weekly.get('weekly_return_pct',0):+.2f}%" if not weekly.get("error") else "N/A",
                    "Analyst Tgt": f"${tgt:,.2f}" if tgt else "N/A",
                    "Upside":    f"{upside:+.1f}%" if upside else "N/A",
                    "RSI":       tech.get("rsi_14d","—"),
                    "MACD":      tech.get("macd_signal","—"),
                    "7d Predict":f"{pred.get('arrow','→')} ${pred.get('predicted_price','N/A')}",
                    "Confidence":pred.get("confidence","—"),
                    "Rating":    (fund.get("analyst_rating") or "N/A").replace("_"," ").title(),
                })
            except Exception:
                rows.append({"Ticker": ticker, "Company": name,
                             "Price":"Error","7-Day":"—","Analyst Tgt":"—","Upside":"—",
                             "RSI":"—","MACD":"—","7d Predict":"—","Confidence":"—","Rating":"—"})
        prog.empty()
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Backtest":
    from datetime import timedelta
    st.title("🔬 Backtest MarketMind Predictions")

    pred_date   = date.today() - timedelta(days=10)
    actual_date = date.today() - timedelta(days=1)
    st.info(f"Predicted on **{pred_date.strftime('%B %d')}** → compared to actual on **{actual_date.strftime('%B %d')}**")

    if st.button("▶  Run Backtest", type="primary"):
        TOP10 = [
            ("AAPL","Apple"), ("MSFT","Microsoft"), ("NVDA","NVIDIA"),
            ("AMZN","Amazon"), ("GOOGL","Alphabet"), ("META","Meta"),
            ("BRK-B","Berkshire"), ("LLY","Eli Lilly"), ("AVGO","Broadcom"), ("TSLA","Tesla"),
        ]
        rows = []
        prog = st.progress(0)
        for i, (ticker, name) in enumerate(TOP10):
            prog.progress((i + 1) / len(TOP10), text=f"Fetching {ticker}...")
            try:
                data         = mm._get_historical_stock_data(ticker, pred_date)
                actual_price = mm._get_actual_price_on(ticker, actual_date)
                if data.get("error"):
                    raise ValueError(data["error"])
                pred = mm._get_ai_predictions([{
                    "ticker": ticker, "name": name,
                    "current_price": data["current_price"],
                    "price_on_pred_date": data["current_price"],
                    "technicals": data["technicals"],
                    "fundamentals": data["fundamentals"],
                }]).get(ticker, {})
                pp   = pred.get("predicted_price")
                base = data["current_price"]
                if pp and actual_price and base:
                    err  = round((pp - actual_price) / actual_price * 100, 2)
                    ok   = (pp > base) == (actual_price > base)
                    verdict = f"{'✅' if ok else '❌'} {abs(err):.1f}% off"
                else:
                    verdict = "N/A"
                rows.append({
                    "Ticker":   ticker, "Company": name,
                    f"Price ({pred_date.strftime('%b %d')})": f"${base:,.2f}" if base else "N/A",
                    "Predicted": f"${pp:,.2f}" if pp else "N/A",
                    f"Actual ({actual_date.strftime('%b %d')})": f"${actual_price:,.2f}" if actual_price else "N/A",
                    "Verdict":  verdict,
                })
            except Exception as e:
                rows.append({"Ticker": ticker, "Company": name,
                             f"Price ({pred_date.strftime('%b %d')})": "Error",
                             "Predicted":"—",
                             f"Actual ({actual_date.strftime('%b %d')})":"—",
                             "Verdict":"Error"})
        prog.empty()
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        correct = sum(1 for r in rows if str(r["Verdict"]).startswith("✅"))
        total   = sum(1 for r in rows if r["Verdict"] not in ("N/A","Error"))
        if total:
            st.markdown("---")
            s1, s2 = st.columns(2)
            s1.metric("Direction Accuracy", f"{correct}/{total} correct")
            s2.metric("Hit Rate", f"{correct/total*100:.0f}%")

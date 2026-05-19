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
from marketmind import ALPACA_PAPER_URL, ALPACA_LIVE_URL
from pre_market import fetch_gap_info, apply_gap_filter, GapRisk, GapType, premarket_snapshot
from prebell import compute_signal

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MarketMind",
    page_icon="📈",
    layout="wide",
)

PROFILES_FILE = os.path.join(os.path.dirname(__file__), "profiles.json")
USER_DATA_DIR  = os.path.join(os.path.dirname(__file__), "user_data")
os.makedirs(USER_DATA_DIR, exist_ok=True)


# ── Profile helpers ───────────────────────────────────────────────────────────

def save_profiles(profiles: dict) -> None:
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=2)


def load_profiles() -> dict:
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE) as f:
            return json.load(f)
    # One-time migration from old per-file layout
    old_dir = os.path.join(os.path.dirname(__file__), "profiles")
    if os.path.isdir(old_dir):
        migrated = {}
        for fname in sorted(os.listdir(old_dir)):
            if fname.startswith("portfolio_") and fname.endswith(".json"):
                try:
                    with open(os.path.join(old_dir, fname)) as fh:
                        data = json.load(fh)
                    pname = data.get("profile_name") or fname.replace("portfolio_", "").replace(".json", "").capitalize()
                    migrated[pname] = data
                except Exception:
                    pass
        if migrated:
            save_profiles(migrated)
        return migrated
    return {}


def _mm_state_path(name: str) -> str:
    return os.path.join(USER_DATA_DIR, f"{name.lower()}_portfolio.json")


def list_profiles() -> list[str]:
    return sorted(st.session_state.profiles.keys())


def load_profile(name: str) -> dict:
    return st.session_state.profiles.get(name, {})


def save_profile(name: str, state: dict) -> None:
    st.session_state.profiles[name] = state
    save_profiles(st.session_state.profiles)
    with open(_mm_state_path(name), "w") as f:
        json.dump(state, f, indent=2)


_PAGES: dict[str, str] = {
    "portfolio":          "Portfolio",
    "daily_trade_scan":   "Daily Trade Scan",
    "new_opportunities":  "New Opportunities",
    "pre_bell_scanner":   "Pre-Bell Scanner",
    "ask_marketmind":     "Ask MarketMind",
    "tax_calculator":     "Tax Calculator",
    "stock_forecast":     "Stock Forecast",
    "top_10_snapshot":    "Top 10 Snapshot",
    "backtest":           "Backtest",
}


def _init_session() -> None:
    if "profiles" not in st.session_state:
        st.session_state.profiles = load_profiles()
    if "authenticated_profile" not in st.session_state:
        st.session_state.authenticated_profile = None
    if "page" not in st.session_state:
        st.session_state.page = "portfolio"


def go_to(page_name: str) -> None:
    st.session_state.page = page_name
    st.rerun()


def verify_pin(name: str, pin: str) -> bool:
    import hashlib
    state = load_profile(name)
    stored_hash = state.get("pin_hash")
    if not stored_hash:
        return True  # legacy profiles without a PIN are always accessible
    return hashlib.sha256(pin.strip().encode()).hexdigest() == stored_hash


def create_profile(name: str, capital: float, risk_level: str,
                   tax_bracket_pct: float = 22.0, trading_persona: str = "swing",
                   pin: str = "0000") -> dict:
    import hashlib
    tp = {"conservative": 3.0, "moderate": 5.0, "aggressive": 8.0}[risk_level]
    sl = {"conservative": 1.5, "moderate": 2.0, "aggressive": 3.0}[risk_level]

    # Persona overrides take-profit / stop-loss / max hold
    persona_cfg = mm.PERSONA_CONFIGS.get(trading_persona, mm.PERSONA_CONFIGS["swing"])
    state = {
        "profile_name":          name,
        "pin_hash":              hashlib.sha256(pin.strip().encode()).hexdigest(),
        "initial_investment":    capital,
        "cash_available":        capital,
        "total_portfolio_value": capital,
        "start_date":            date.today().isoformat(),
        "end_date":              (date.today().replace(year=date.today().year + 1)).isoformat(),
        "risk_level":            risk_level,
        "tax_bracket_pct":       tax_bracket_pct,
        "trading_persona":       trading_persona,
        "persona_label":         persona_cfg["label"],
        "take_profit_pct":       persona_cfg["take_profit_pct"],
        "stop_loss_pct":         persona_cfg["stop_loss_pct"],
        "max_hold_days":         persona_cfg["max_hold_days"],
        "max_open_trades":       persona_cfg["max_open_trades"],
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
_init_session()

st.sidebar.title("📈 MarketMind")
st.sidebar.caption("Free swing trading signals — yfinance powered")
st.sidebar.markdown("---")

existing = list_profiles()

# ── Authentication gate ───────────────────────────────────────────────────────
if st.session_state.authenticated_profile is None:
    st.sidebar.subheader("👤 Profile")
    _login_tab, _create_tab = st.sidebar.tabs(["Login", "Create"])

    with _login_tab:
        if existing:
            _login_name = st.selectbox("Select profile", existing, key="login_select")
            _login_pin  = st.text_input("PIN", type="password", max_chars=8, key="login_pin",
                                        placeholder="Leave blank for legacy profiles")
            if st.button("Login", key="btn_login"):
                if verify_pin(_login_name, _login_pin):
                    st.session_state.authenticated_profile = _login_name
                    st.session_state["active_profile"] = _login_name
                    st.rerun()
                else:
                    st.error("Incorrect PIN.")
        else:
            st.info("No profiles yet. Create one in the **Create** tab.")

    with _create_tab:
        new_name    = st.text_input("Profile name", placeholder="e.g. My Growth Fund", key="new_name_auth")
        new_pin     = st.text_input("PIN (4–8 digits)", type="password", max_chars=8,
                                    placeholder="e.g. 1234", key="new_pin_auth")
        new_pin2    = st.text_input("Confirm PIN", type="password", max_chars=8, key="new_pin2_auth")
        new_capital = st.number_input("Starting budget ($)", value=10000, step=500, min_value=100, key="new_cap_auth")
        new_persona = st.selectbox(
            "Trading persona", ["swing", "day", "longterm"],
            format_func=lambda x: {
                "day":      "⚡ Day Trading",
                "swing":    "🔄 Swing Trading",
                "longterm": "📈 Long-term Investing",
            }[x],
            key="new_persona_auth",
        )
        new_risk = st.selectbox("Risk appetite", ["moderate", "conservative", "aggressive"], key="new_risk_auth")
        new_tax  = st.selectbox("Federal tax bracket", [10, 12, 22, 24, 32, 35, 37], index=2, key="new_tax_auth")

        if st.button("Create & Login", key="btn_create_auth"):
            if not new_name.strip():
                st.warning("Enter a profile name.")
            elif len(new_pin.strip()) < 4:
                st.warning("PIN must be at least 4 digits.")
            elif new_pin != new_pin2:
                st.error("PINs do not match.")
            else:
                _name  = new_name.strip()
                create_profile(_name, float(new_capital), new_risk, float(new_tax), new_persona, new_pin)
                st.session_state.authenticated_profile = _name
                st.session_state["active_profile"] = _name
                st.success(f"Profile **{_name}** created!")
                st.rerun()

    st.stop()

# ── Logged-in sidebar ─────────────────────────────────────────────────────────
active_profile = st.session_state.authenticated_profile

# Refresh profile list (may have changed) and allow switching within session
existing = list_profiles()
st.sidebar.subheader("👤 Profile")
_default_index = existing.index(active_profile) if active_profile in existing else 0
_switched = st.sidebar.selectbox("Active profile", existing, index=_default_index, key="profile_switcher")
if _switched != active_profile:
    # Require re-authentication when switching profiles
    st.session_state.authenticated_profile = None
    st.session_state["active_profile"] = _switched
    st.rerun()

if st.sidebar.button("Logout", key="btn_logout"):
    st.session_state.authenticated_profile = None
    st.rerun()

profile_state = load_profile(active_profile)

# Point the agent at this profile's state file
mm.PORTFOLIO_STATE_FILE = _mm_state_path(active_profile)

st.sidebar.markdown("---")

# ── Persona switcher ──────────────────────────────────────────────────────────
# Users can switch personas within the same profile at any time.
# The change is saved back to portfolio_state.json immediately.
_persona_options = ["day", "swing", "longterm"]
_persona_labels  = {
    "day":      "⚡ Day Trading",
    "swing":    "🔄 Swing Trading",
    "longterm": "📈 Long-term Investing",
}
_current_persona = profile_state.get("trading_persona", "swing")
_persona_index   = _persona_options.index(_current_persona) if _current_persona in _persona_options else 1

_selected_persona = st.sidebar.selectbox(
    "Trading persona",
    _persona_options,
    index=_persona_index,
    format_func=lambda x: _persona_labels[x],
    key=f"persona_switcher_{active_profile}",
)

# If persona changed, save it to the profile immediately
if _selected_persona != _current_persona:
    _new_pcfg = mm.PERSONA_CONFIGS[_selected_persona]
    profile_state["trading_persona"]   = _selected_persona
    profile_state["persona_label"]     = _new_pcfg["label"]
    profile_state["take_profit_pct"]   = _new_pcfg["take_profit_pct"]
    profile_state["stop_loss_pct"]     = _new_pcfg["stop_loss_pct"]
    profile_state["max_hold_days"]     = _new_pcfg["max_hold_days"]
    profile_state["max_open_trades"]   = _new_pcfg["max_open_trades"]
    save_profile(active_profile, profile_state)
    st.rerun()

# Apply the active persona to TRADING_CONFIG
mm.apply_persona(_selected_persona)
mm.TRADING_CONFIG["risk_level"]       = profile_state.get("risk_level", "moderate")
mm.TRADING_CONFIG["starting_capital"] = profile_state.get("initial_investment", 10000)
_persona_key = _selected_persona   # short alias used throughout the page

_active_pcfg = mm.PERSONA_CONFIGS[_selected_persona]
tp = profile_state.get("take_profit_pct", _active_pcfg["take_profit_pct"])
sl = profile_state.get("stop_loss_pct",   _active_pcfg["stop_loss_pct"])

tax_bracket = profile_state.get("tax_bracket_pct", 22.0)
st.sidebar.caption(f"Take-profit **+{tp}%**  |  Stop-loss **−{sl}%**  |  Hold **{_active_pcfg['hold_label']}**")
st.sidebar.caption(f"Risk: {profile_state.get('risk_level','—').title()}  |  Tax: {int(tax_bracket)}%")
st.sidebar.caption(f"{'✅' if profile_state.get('robinhood_imported') else '⬜'} Robinhood data")

# Mode nav
st.sidebar.markdown("---")
_page_keys   = list(_PAGES.keys())
_page_labels = list(_PAGES.values())
_page_index  = _page_keys.index(st.session_state.page) if st.session_state.page in _page_keys else 0
_selected_label = st.sidebar.radio("Mode", _page_labels, index=_page_index)
_selected_key   = _page_keys[_page_labels.index(_selected_label)]
if _selected_key != st.session_state.page:
    go_to(_selected_key)

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ Educational purposes only. Not financial advice.")


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO — upload Robinhood data + view holdings
# ══════════════════════════════════════════════════════════════════════════════
def show_portfolio():
    st.title(f"💼 Portfolio — {active_profile}")
    st.markdown(
        f"**Your command centre.** This is where you see everything you currently own, "
        f"how each position is performing, and whether you should hold or exit. "
        f"You can also connect your brokerage account here so MarketMind reads your real positions automatically — "
        f"no manual entry needed."
    )
    st.info(
        f"**Active persona: {_active_pcfg['emoji']} {_active_pcfg['label']}** — {_active_pcfg['tagline']}  \n"
        f"Exit signals use: take-profit **+{tp}%** and stop-loss **−{sl}%**. "
        f"Positions held longer than **{_active_pcfg['hold_label']}** will be flagged for review."
    )

    # ── Robinhood upload ──
    st.subheader("📥 Import Robinhood Data")
    st.caption(
        "Upload your Robinhood transaction history and MarketMind will rebuild your portfolio automatically — "
        "all your stocks, how many shares, and what you paid. "
        "Export it from: Robinhood app → Account → Statements & History → Download CSV"
    )

    uploaded = st.file_uploader(
        "Upload your Robinhood account history (CSV or PDF)",
        type=["csv", "pdf"],
        key=f"upload_{active_profile}",
    )

    if uploaded:
        raw_path = os.path.join(USER_DATA_DIR, f"{active_profile}_{uploaded.name}")
        with open(raw_path, "wb") as _fh:
            _fh.write(uploaded.getvalue())

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

    # ── Live broker connection ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔗 Connect a Live Brokerage Account")
    st.caption(
        "Link your real brokerage so MarketMind can read your actual positions. "
        "Credentials are stored only in your local `.env` file — never uploaded or shared. "
        "MarketMind reads positions only; it cannot place or cancel orders."
    )

    broker_tab1, broker_tab2 = st.tabs(["Alpaca (recommended)", "Interactive Brokers (IBKR)"])

    with broker_tab1:
        st.markdown(
            "**Alpaca** is a free, regulated US brokerage with an official API.  \n"
            "You can connect a **paper** (simulated) account for testing, or a **live** account when ready.  \n"
            "Get free keys in ~1 min: [alpaca.markets](https://alpaca.markets) → Dashboard → API Keys → Generate New Key"
        )
        alp_col1, alp_col2 = st.columns(2)
        with alp_col1:
            alp_key    = st.text_input("API Key ID",    type="password", key="alp_key",
                                        help="Found in your Alpaca dashboard under API Keys. Starts with 'PK' for paper accounts.")
            alp_secret = st.text_input("Secret Key",    type="password", key="alp_secret",
                                        help="Shown only once when you generate the key. Store it safely.")
        with alp_col2:
            alp_type = st.radio("Account type", ["Paper (simulated)", "Live (real money)"], key="alp_type",
                                 help="Paper = fake money, no real risk. Use this to test MarketMind first.")
            if alp_type == "Live (real money)":
                st.warning("⚠️ Live account selected. No trades are placed automatically.")

        if st.button("Test & Connect Alpaca", key="btn_alpaca"):
            if not alp_key or not alp_secret:
                st.warning("Please enter both your API Key ID and Secret Key.")
            else:
                base_url = ALPACA_PAPER_URL if "Paper" in alp_type else ALPACA_LIVE_URL
                with st.spinner("Testing connection..."):
                    account = mm._test_alpaca_connection(alp_key, alp_secret, base_url)

                if "error" in account:
                    st.error(f"Connection failed: {account['error']}")
                else:
                    equity = float(account.get("equity", 0))
                    cash   = float(account.get("cash",   0))
                    st.success(
                        f"✓ Connected to Alpaca ({alp_type})  |  "
                        f"Status: {account.get('status','?').upper()}  |  "
                        f"Equity: ${equity:,.2f}  |  Cash: ${cash:,.2f}"
                    )
                    mm._save_env_keys(
                        {"ALPACA_API_KEY": alp_key, "ALPACA_API_SECRET": alp_secret, "ALPACA_BASE_URL": base_url},
                        "# Alpaca brokerage — https://alpaca.markets"
                    )
                    profile_state.update({"broker": "alpaca", "broker_url": base_url,
                                          "broker_acct_type": "paper" if "Paper" in alp_type else "live"})

                    # Fetch positions
                    positions = mm._fetch_alpaca_positions(alp_key, alp_secret, base_url)
                    if positions:
                        pos_rows = [{
                            "Symbol": p.get("symbol"),
                            "Qty": p.get("qty"),
                            "Avg Entry": f"${float(p.get('avg_entry_price',0)):.2f}",
                            "Market Value": f"${float(p.get('market_value',0)):,.2f}",
                            "P&L": f"${float(p.get('unrealized_pl',0)):+,.2f}",
                        } for p in positions]
                        st.markdown(f"**{len(positions)} open position(s) in your Alpaca account:**")
                        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

                        if st.button("Import these positions into MarketMind", key="btn_alpaca_import"):
                            holdings = profile_state.setdefault("holdings", {})
                            for p in positions:
                                sym  = p.get("symbol", "")
                                qty  = float(p.get("qty", 0))
                                cost = float(p.get("avg_entry_price", 0))
                                if sym and qty > 0:
                                    holdings[sym] = {"shares": qty, "avg_buy_price": cost,
                                                     "source": "alpaca_import",
                                                     "import_date": date.today().isoformat()}
                            profile_state["cash_available"]        = cash
                            profile_state["total_portfolio_value"] = equity
                            save_profile(active_profile, profile_state)
                            st.success(f"✓ {len(positions)} position(s) imported. Scroll down to see your holdings.")
                            st.rerun()
                    else:
                        save_profile(active_profile, profile_state)
                        st.info("No open positions found in your Alpaca account.")

    with broker_tab2:
        st.markdown(
            "**Interactive Brokers** is a powerful, official brokerage API supporting stocks, options, futures, and more.  \n"
            "Requires **Trader Workstation (TWS)** or **IB Gateway** running on your computer.  \n"
            "[Download TWS](https://www.interactivebrokers.com/en/trading/tws.php)  |  "
            "In TWS: Edit → Global Configuration → API → Enable Socket → port **7497** (paper) or **7496** (live)"
        )
        ibkr_col1, ibkr_col2 = st.columns(2)
        with ibkr_col1:
            ibkr_type = st.radio("Account type", ["Paper (port 7497)", "Live (port 7496)"], key="ibkr_type",
                                  help="Paper = TWS paper trading session. Live = real funded account.")
        with ibkr_col2:
            ibkr_port = 7497 if "7497" in ibkr_type else 7496
            st.metric("TWS Port", ibkr_port)
            if ibkr_port == 7496:
                st.warning("⚠️ Live account selected. No trades are placed automatically.")

        if st.button("Test & Connect IBKR", key="btn_ibkr"):
            try:
                from ib_insync import IB, util
                util.startLoop()
            except ImportError:
                st.error("ib_insync is not installed. Run `pip install ib_insync` in your terminal, then restart the app.")
                st.stop()

            with st.spinner(f"Connecting to TWS on port {ibkr_port}..."):
                ib = IB()
                try:
                    ib.connect("127.0.0.1", ibkr_port, clientId=10)
                except Exception as e:
                    st.error(f"Could not connect: {e}. Make sure TWS is running with API access enabled.")
                    st.stop()

            summary = {s.tag: s.value for s in ib.accountSummary()}
            equity  = float(summary.get("NetLiquidation", 0))
            cash    = float(summary.get("TotalCashValue",  0))
            acct_id = summary.get("AccountId", "unknown")
            st.success(
                f"✓ Connected to IBKR  |  Account: {acct_id}  |  Equity: ${equity:,.2f}  |  Cash: ${cash:,.2f}"
            )
            mm._save_env_keys(
                {"IBKR_PORT": str(ibkr_port), "IBKR_ACCT_TYPE": "paper" if ibkr_port == 7497 else "live"},
                "# Interactive Brokers — requires TWS or IB Gateway running locally"
            )
            profile_state.update({"broker": "ibkr", "broker_port": ibkr_port})

            try:
                raw = ib.positions()
                positions = []
                for p in raw:
                    sym  = p.contract.symbol
                    qty  = float(p.position)
                    cost = float(p.avgCost) / qty if qty else 0
                    positions.append({"symbol": sym, "qty": qty, "avg_buy_price": round(cost, 4)})
            except Exception:
                positions = []

            ib.disconnect()

            if positions:
                st.markdown(f"**{len(positions)} open position(s) in your IBKR account:**")
                st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)
                if st.button("Import these positions into MarketMind", key="btn_ibkr_import"):
                    holdings = profile_state.setdefault("holdings", {})
                    for p in positions:
                        if p["symbol"] and p["qty"] > 0:
                            holdings[p["symbol"]] = {
                                "shares": p["qty"], "avg_buy_price": p["avg_buy_price"],
                                "source": "ibkr_import", "import_date": date.today().isoformat()
                            }
                    profile_state["cash_available"]        = cash
                    profile_state["total_portfolio_value"] = equity
                    save_profile(active_profile, profile_state)
                    st.success(f"✓ {len(positions)} position(s) imported.")
                    st.rerun()
            else:
                save_profile(active_profile, profile_state)
                st.info("No open positions found in your IBKR account.")

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
        _eck_desc = {
            "day":      "Checks each position using **intraday rules** — did it hit your +1.5% target or -0.75% stop? "
                        "Day traders should not hold positions overnight unless intentional.",
            "swing":    "Checks each stock you own using **technical indicators** (RSI, MACD) and your "
                        "take-profit/stop-loss levels. Tells you: keep holding, take profit, or cut your loss.",
            "longterm": "Reviews each position against **fundamental and trend signals** — P/E ratio, analyst rating, "
                        "and the 200-day moving average. Long-term investors should ignore short-term dips unless "
                        "the underlying business has changed.",
        }
        st.caption(_eck_desc.get(_persona_key, _eck_desc["swing"]))

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
# PRE-BELL SCANNER
# ══════════════════════════════════════════════════════════════════════════════
def show_pre_bell_scanner():
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
    now_et = datetime.now(tz=_ET)

    st.title("🌅 Pre-Bell Scanner")

    _prebell_desc = {
        "day": (
            "**Your morning alarm before the market opens.** "
            "The stock market opens at 9:30 AM ET, but prices start moving as early as 4 AM in 'pre-market' trading. "
            "This scanner checks what happened overnight — did a stock gap up or down? Is volume spiking? — "
            "so you're ready to act the moment the bell rings.  \n\n"
            "⚡ **Day traders use this most.** A pre-market gap + high volume = the best intraday setups."
        ),
        "swing": (
            "**A morning briefing before the market opens.** "
            "Checks overnight price movements and whether your planned swing trades are still valid. "
            "A big overnight gap can blow up a setup — this tells you before you commit money.  \n\n"
            "🔄 **Best used 8–9:25 AM ET**, before the regular session begins at 9:30 AM."
        ),
        "longterm": (
            "**Optional morning check for long-term investors.** "
            "This scanner is primarily designed for short-term traders, but long-term investors can use it "
            "to spot big overnight moves in stocks they're watching — earnings surprises, news events, or "
            "major gaps that might create a better entry price.  \n\n"
            "📈 **Long-term tip:** A stock gapping DOWN 5% on temporary bad news is often a *buying opportunity*, "
            "not a reason to panic."
        ),
    }
    st.markdown(_prebell_desc.get(_persona_key, _prebell_desc["swing"]))
    st.caption("Best run **4:00–9:25 AM ET**. After 9:30 AM the market is open and pre-market data becomes stale.")

    # ── Session badge ─────────────────────────────────────────────────────────
    if now_et.hour < 4:
        st.info("⏳ Before 4 AM ET — pre-market hasn't opened yet. Data will be limited.")
    elif now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 30):
        st.success(f"🌅 Pre-market session open — {now_et.strftime('%H:%M ET')}")
    elif now_et.hour < 16:
        st.warning("📈 Regular session is open — pre-market data may be stale.")
    else:
        st.info("🌙 After hours — reviewing EOD data for tomorrow's plan.")

    st.markdown("---")

    # ── Ticker selection ──────────────────────────────────────────────────────
    st.subheader("Select tickers to scan")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        custom_input = st.text_input(
            "Enter tickers (comma-separated) — leave blank to use full TRADING_UNIVERSE",
            placeholder="e.g. AAPL, NVDA, TSLA, MSFT",
        )
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        use_holdings = st.checkbox("My holdings only", value=False)

    if use_holdings:
        tickers_to_scan = list(profile_state.get("holdings", {}).keys()) or mm.TRADING_UNIVERSE
    elif custom_input.strip():
        tickers_to_scan = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    else:
        tickers_to_scan = mm.TRADING_UNIVERSE

    st.caption(f"Scanning **{len(tickers_to_scan)}** tickers: {', '.join(tickers_to_scan)}")

    run_prebell = st.button("▶  Run Pre-Bell Scan", type="primary", use_container_width=True)

    if run_prebell:
        # ── Fetch all data ────────────────────────────────────────────────────
        with st.spinner("Fetching pre-market data..."):
            gaps = fetch_gap_info(tickers_to_scan)

        with st.spinner("Fetching technical indicators..."):
            techs   = {t: mm.get_technical_indicators(t) for t in tickers_to_scan}
            weeklys = {t: mm.get_weekly_return(t) for t in tickers_to_scan}

        # ── Compute signals ───────────────────────────────────────────────────
        signals = []
        for ticker in tickers_to_scan:
            gap    = gaps[ticker]
            tech   = techs[ticker]
            weekly = weeklys[ticker]
            signal, reason = compute_signal(gap, tech)
            signals.append({
                "ticker": ticker, "gap": gap,
                "tech": tech, "weekly": weekly,
                "signal": signal, "reason": reason,
            })

        # Save to session state so Ask MarketMind can reference them
        st.session_state["prebell_signals"] = signals

        # ── Summary metrics ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Summary")

        n_buy   = sum(1 for s in signals if s["signal"] == "BUY")
        n_sell  = sum(1 for s in signals if s["signal"] == "SELL")
        n_watch = sum(1 for s in signals if s["signal"] == "WATCH")
        n_skip  = sum(1 for s in signals if s["signal"] == "SKIP")
        n_high  = sum(1 for s in signals if s["gap"].gap_risk == GapRisk.HIGH)
        n_mod   = sum(1 for s in signals if s["gap"].gap_risk == GapRisk.MODERATE)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("🟢 BUY",    n_buy)
        m2.metric("🔴 SELL",   n_sell)
        m3.metric("🟡 WATCH",  n_watch)
        m4.metric("⛔ SKIP",   n_skip)
        m5.metric("🔴 High Gap Risk",  n_high)
        m6.metric("🟡 Moderate Gap",   n_mod)

        # ── Signal table ──────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("All Signals")

        _RISK_ICONS = {"low": "🟢", "moderate": "🟡", "high": "🔴"}
        _SIG_ICONS  = {"BUY": "🟢", "SELL": "🔴", "WATCH": "🟡", "SKIP": "⛔", "ERROR": "❌"}

        table_rows = []
        for s in signals:
            g  = s["gap"]
            t  = s["tech"]
            wk = s["weekly"]
            table_rows.append({
                "Signal":      f"{_SIG_ICONS.get(s['signal'], '?')} {s['signal']}",
                "Ticker":      s["ticker"],
                "Price":       f"${t.get('current_price', g.prev_close):.2f}" if not t.get("error") else "N/A",
                "7-Day":       f"{wk.get('weekly_return_pct', 0):+.2f}%" if not wk.get("error") else "N/A",
                "PM Gap %":    f"{g.gap_pct:+.2f}%" if g.data_available else "N/A",
                "Gap Risk":    f"{_RISK_ICONS.get(g.gap_risk.value, '')} {g.gap_risk.value}" if g.data_available else "N/A",
                "PM Vol":      f"{g.pm_vol_ratio:.1%}" if g.data_available else "N/A",
                "Conviction":  "⚡ Yes" if g.high_conviction else "—",
                "RSI":         t.get("rsi_14d", "—") if not t.get("error") else "—",
                "MACD":        t.get("macd_signal", "—") if not t.get("error") else "—",
                "BB %":        f"{t.get('bb_price_position_pct', '—'):.0f}%" if not t.get("error") and t.get("bb_price_position_pct") else "—",
                "Reason":      s["reason"],
            })

        # Sort: BUY first, then SELL, WATCH, SKIP, ERROR
        order = {"BUY": 0, "SELL": 1, "WATCH": 2, "SKIP": 3, "ERROR": 4}
        table_rows.sort(key=lambda r: order.get(r["Signal"].split()[-1], 5))
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        # ── Expanded per-ticker cards ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("Detailed View")

        buy_signals  = [s for s in signals if s["signal"] == "BUY"]
        sell_signals = [s for s in signals if s["signal"] == "SELL"]
        watch_signals= [s for s in signals if s["signal"] == "WATCH"]
        skip_signals = [s for s in signals if s["signal"] == "SKIP"]

        def _render_signal_cards(group: list, label: str, colour: str) -> None:
            if not group:
                return
            st.markdown(f"#### {colour} {label}")
            for s in group:
                g  = s["gap"]
                t  = s["tech"]
                wk = s["weekly"]
                price = t.get("current_price", g.prev_close) if not t.get("error") else g.prev_close
                wk_str = f"{wk.get('weekly_return_pct', 0):+.2f}%" if not wk.get("error") else "N/A"

                with st.expander(
                    f"{colour} **{s['ticker']}**  ${price:.2f}  (7d: {wk_str})  —  {s['reason'][:80]}..."
                    if len(s["reason"]) > 80 else f"{colour} **{s['ticker']}**  ${price:.2f}  (7d: {wk_str})  —  {s['reason']}"
                ):
                    c1, c2, c3 = st.columns(3)

                    # Pre-market
                    with c1:
                        st.markdown("**Pre-Market**")
                        if g.data_available:
                            risk_icon = _RISK_ICONS.get(g.gap_risk.value, "")
                            st.metric("Gap %",    f"{g.gap_pct:+.2f}%")
                            st.metric("PM Price", f"${g.pm_last_price:.2f}", f"prev ${g.prev_close:.2f}")
                            st.caption(f"Risk: {risk_icon} {g.gap_risk.value}")
                            st.caption(f"PM vol: {g.pm_volume:,} ({g.pm_vol_ratio:.1%} of daily avg)")
                            if g.high_conviction:
                                st.success("⚡ High conviction volume")
                        else:
                            st.caption("No pre-market data available")

                    # Technical indicators
                    with c2:
                        st.markdown("**Technicals**")
                        if not t.get("error"):
                            st.metric("RSI (14d)", t.get("rsi_14d", "—"), t.get("rsi_signal", ""))
                            st.metric("MACD",      t.get("macd_signal", "—"))
                            st.metric("Stoch %K",  t.get("stoch_k", "—"), t.get("stoch_signal", ""))
                        else:
                            st.caption(f"Error: {t['error']}")

                    # More technicals
                    with c3:
                        st.markdown("**Bands & Volume**")
                        if not t.get("error"):
                            bb_pct = t.get("bb_price_position_pct")
                            st.metric("BB Position", f"{bb_pct:.0f}th %" if bb_pct else "—", t.get("bb_signal", ""))
                            st.metric("Volume",      f"{t.get('volume_ratio', '—')}x avg", t.get("volume_signal", ""))
                            st.caption(t.get("ma_signal", ""))

                    st.caption(f"**Signal reason:** {s['reason']}")

        _render_signal_cards(buy_signals,   "BUY",   "🟢")
        _render_signal_cards(sell_signals,  "SELL",  "🔴")
        _render_signal_cards(watch_signals, "WATCH", "🟡")
        _render_signal_cards(skip_signals,  "SKIP",  "⛔")

        # ── Gap risk warning ──────────────────────────────────────────────────
        high_risk_tickers = [s["ticker"] for s in signals if s["gap"].gap_risk == GapRisk.HIGH]
        moderate_tickers  = [s["ticker"] for s in signals if s["gap"].gap_risk == GapRisk.MODERATE]

        if high_risk_tickers or moderate_tickers:
            st.markdown("---")
            st.subheader("⚠️ Gap Risk Alerts")
            if high_risk_tickers:
                st.error(
                    f"**🔴 High gap risk — signals suppressed for:** {', '.join(high_risk_tickers)}  \n"
                    "These stocks gapped >5% from yesterday's close. Entry price has moved too far — "
                    "risk/reward has collapsed. Wait for price to stabilise after open."
                )
            if moderate_tickers:
                st.warning(
                    f"**🟡 Moderate gap (2–5%) — reduce position size for:** {', '.join(moderate_tickers)}  \n"
                    "Consider using a limit order near yesterday's close rather than buying at market open."
                )


# ══════════════════════════════════════════════════════════════════════════════
# ASK MARKETMIND — conversational follow-up Q&A
# ══════════════════════════════════════════════════════════════════════════════
def show_ask_marketmind():
    st.title("💬 Ask MarketMind")
    st.caption(
        "Ask anything about your portfolio, today's signals, pre-market conditions, "
        "or general trading questions. MarketMind answers with your live data as context."
    )

    holdings = profile_state.get("holdings", {})

    # ── Session state for chat history ────────────────────────────────────────
    chat_key = f"chat_history_{active_profile}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    # ── Suggested starter questions ───────────────────────────────────────────
    if not st.session_state[chat_key]:
        st.markdown("**Try asking:**")
        suggestions = [
            "Which of my holdings should I be watching most closely today?",
            "Explain the pre-market gap risk for my portfolio",
            "Should I take profit on any positions?",
            "What does a MACD bearish crossover mean for my trade?",
            "Which tickers from the scan look most interesting?",
        ]
        cols = st.columns(len(suggestions))
        for col, q in zip(cols, suggestions):
            if col.button(q, use_container_width=True):
                st.session_state[chat_key].append({"role": "user", "content": q})
                st.rerun()
        st.markdown("---")

    # ── Render chat history ───────────────────────────────────────────────────
    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    if user_input := st.chat_input("Ask a follow-up question about your portfolio or signals…"):
        st.session_state[chat_key].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                history = st.session_state[chat_key][:-1]  # exclude the message just appended
                reply   = mm.ask_mastermind(
                    question             = user_input,
                    portfolio_state      = profile_state,
                    conversation_history = history,
                )
            st.markdown(reply)
            st.session_state[chat_key].append({"role": "assistant", "content": reply})

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        if st.button("🗑️ Clear chat history"):
            st.session_state[chat_key] = []
            st.rerun()
        st.caption(f"{len(st.session_state[chat_key])} messages in this session")
        if holdings:
            st.caption(f"Context includes {len(holdings)} holdings")


# ══════════════════════════════════════════════════════════════════════════════
# TAX CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
def show_tax_calculator():
    st.title("🧾 Tax Calculator")
    st.markdown(
        "**Estimates how much tax you'd owe if you sold your positions today.**  \n\n"
        "In the US, profits from selling stocks are called **capital gains** and they're taxable. "
        "The rate depends on how long you held the stock:  \n"
        "- **Short-term** (held less than 1 year): taxed at your regular income rate — same as your salary  \n"
        "- **Long-term** (held 1 year or more): taxed at a lower special rate of 0%, 15%, or 20%  \n\n"
        "This tool shows you which of your positions are short- vs long-term, how close each is to the "
        "1-year threshold, and whether selling a losing position could reduce your overall tax bill "
        "(a strategy called **tax-loss harvesting**)."
    )
    st.caption("⚠️ Federal estimates only. State taxes, AMT, and wash-sale adjustments not included — consult a CPA for your full picture.")

    holdings = profile_state.get("holdings", {})

    if not holdings:
        st.info("No holdings found. Import your Robinhood data in the Portfolio tab first.")
        st.stop()

    # ── Tax bracket override ──────────────────────────────────────────────────
    st.subheader("⚙️ Your Tax Settings")
    col_tx1, col_tx2, col_tx3 = st.columns(3)
    with col_tx1:
        bracket = st.selectbox(
            "Federal income tax bracket",
            [10, 12, 22, 24, 32, 35, 37],
            index=[10, 12, 22, 24, 32, 35, 37].index(int(profile_state.get("tax_bracket_pct", 22))),
            help="Your marginal federal rate. Short-term gains are taxed at this rate."
        )
    with col_tx2:
        lt_rate_display = 0 if bracket <= 12 else (15 if bracket <= 35 else 20)
        st.metric("Long-term gains rate", f"{lt_rate_display}%",
                  help="Applies to positions held ≥ 365 days.")
    with col_tx3:
        st.metric("Short-term gains rate", f"{bracket}%",
                  help="Applies to positions held < 365 days — all typical swing trades.")

    # Recompute with the selected bracket
    tax_data = mm.compute_tax_implications(holdings, float(bracket))

    lt_rate  = bracket / 100 * (0 if bracket <= 12 else (0.15 / (bracket / 100) if bracket <= 35 else 0.20 / (bracket / 100)))
    lt_rate  = 0.0 if bracket <= 12 else (0.15 if bracket <= 35 else 0.20)
    st_rate  = bracket / 100

    # ── Aggregate summary ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Portfolio Tax Summary")

    total_unrealized       = sum(td["unrealized_gain"] for td in tax_data.values())
    total_gains            = sum(td["unrealized_gain"] for td in tax_data.values() if td["unrealized_gain"] > 0)
    total_losses           = sum(td["unrealized_gain"] for td in tax_data.values() if td["unrealized_gain"] < 0)
    total_est_tax          = sum(td["est_tax_if_sold_now"] for td in tax_data.values())
    st_gains               = sum(td["unrealized_gain"] for td in tax_data.values() if td["gain_type"] == "Short-term" and td["unrealized_gain"] > 0)
    lt_gains               = sum(td["unrealized_gain"] for td in tax_data.values() if td["gain_type"] == "Long-term"  and td["unrealized_gain"] > 0)
    tax_if_offset          = max(0, total_gains + total_losses) * st_rate   # rough: losses offset gains
    net_after_tax          = total_unrealized - total_est_tax

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Unrealized P&L",    f"${total_unrealized:+,.2f}")
    s2.metric("Est. Tax (all sold now)", f"${total_est_tax:,.2f}",
              delta=f"-{total_est_tax/total_gains*100:.0f}% of gains" if total_gains > 0 else None,
              delta_color="inverse")
    s3.metric("Keep After Tax",          f"${net_after_tax:,.2f}")
    s4.metric("Losses to Harvest",       f"${abs(total_losses):,.2f}",
              help="Selling losing positions offsets gains and reduces tax bill.")

    # Gain type breakdown
    g1, g2, g3 = st.columns(3)
    g1.metric("Short-term gains",  f"${st_gains:,.2f}",  f"taxed at {bracket}%")
    g2.metric("Long-term gains",   f"${lt_gains:,.2f}",  f"taxed at {lt_rate_display}%")
    g3.metric("Tax if losses offset gains", f"${tax_if_offset:,.2f}",
              delta=f"save ${total_est_tax - tax_if_offset:,.2f}" if total_losses < 0 else "no losses to use",
              delta_color="normal")

    # ── Per-position breakdown ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Per-Position Tax Breakdown")

    rows = []
    for ticker, td in tax_data.items():
        pos         = holdings[ticker]
        gain        = td["unrealized_gain"]
        gain_type   = td["gain_type"]
        holding_days= td.get("holding_days")
        days_to_lt  = td.get("days_to_lt")
        est_tax     = td["est_tax_if_sold_now"]
        rate        = td["effective_rate_pct"]
        harvest     = td["tax_loss_harvest"]

        # Days to LT label
        if gain_type == "Long-term":
            lt_label = "✅ Long-term"
        elif days_to_lt is not None and days_to_lt <= 30:
            lt_label = f"⚠️ {days_to_lt}d to LT"
        elif days_to_lt is not None:
            lt_label = f"🕐 {days_to_lt}d to LT"
        else:
            lt_label = "Unknown"

        rows.append({
            "Ticker":           ticker,
            "Shares":           pos.get("shares", 0),
            "Avg Buy":          f"${pos.get('avg_buy_price', 0):.2f}",
            "Current":          f"${pos.get('current_price', 0):.2f}",
            "Unrealized P&L":   f"${gain:+,.2f}",
            "Held":             f"{holding_days}d" if holding_days else "?",
            "Gain Type":        gain_type,
            "LT Status":        lt_label,
            "Tax Rate":         f"{rate}%",
            "Est. Tax":         f"${est_tax:,.2f}" if gain > 0 else ("Harvest 🎯" if harvest else "—"),
            "Net After Tax":    f"${gain - est_tax:,.2f}" if gain > 0 else f"${gain:,.2f}",
        })

    # Sort: losses first (harvest), then by est tax descending
    rows.sort(key=lambda r: (0 if r["Est. Tax"] == "Harvest 🎯" else 1, r["Ticker"]))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── What-if: wait for long-term ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("💡 What-If: Wait for Long-Term Rate")
    st.caption("Shows how much you'd save in tax by holding each profitable short-term position until it qualifies for long-term rates.")

    waitfor_rows = []
    for ticker, td in tax_data.items():
        if td["gain_type"] != "Short-term" or td["unrealized_gain"] <= 0:
            continue
        gain        = td["unrealized_gain"]
        days_to_lt  = td.get("days_to_lt", 0) or 0
        tax_now     = round(gain * st_rate, 2)
        tax_lt      = round(gain * lt_rate, 2)
        savings     = round(tax_now - tax_lt, 2)
        waitfor_rows.append({
            "Ticker":           ticker,
            "Unrealized Gain":  f"${gain:,.2f}",
            "Days to LT":       days_to_lt,
            "Tax if Sold Now":  f"${tax_now:,.2f}",
            "Tax at LT Rate":   f"${tax_lt:,.2f}",
            "Potential Saving": f"${savings:,.2f}",
            "Worth Waiting?":   "✅ Yes" if savings > 200 else ("⚠️ Marginal" if savings > 50 else "❌ No"),
        })

    if waitfor_rows:
        waitfor_rows.sort(key=lambda r: -float(r["Potential Saving"].replace("$","").replace(",","")))
        st.dataframe(pd.DataFrame(waitfor_rows), use_container_width=True, hide_index=True)

        top_saver = waitfor_rows[0]
        st.info(
            f"**Biggest opportunity:** {top_saver['Ticker']} — waiting {top_saver['Days to LT']} more days "
            f"could save you **{top_saver['Potential Saving']}** in federal tax."
        )
    else:
        st.success("All profitable positions are already long-term — you're at the lowest applicable rate.")

    # ── Tax-loss harvesting ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Tax-Loss Harvesting Opportunities")
    st.caption(
        "Selling losing positions crystallises the loss, which offsets gains elsewhere and reduces your total tax bill. "
        "**Wash-sale rule:** don't buy the same stock back within 30 days or the loss is disallowed."
    )

    harvest_rows = []
    total_harvestable = 0.0
    for ticker, td in tax_data.items():
        if not td["tax_loss_harvest"]:
            continue
        loss        = abs(td["unrealized_gain"])
        tax_saved   = round(loss * st_rate, 2)   # offsets short-term gains first
        total_harvestable += tax_saved
        harvest_rows.append({
            "Ticker":           ticker,
            "Unrealized Loss":  f"-${loss:,.2f}",
            "Est. Tax Saved":   f"${tax_saved:,.2f}",
            "Held":             f"{td.get('holding_days','?')}d",
            "Wash-Sale Window": "30 days — don't rebuy immediately",
        })

    if harvest_rows:
        harvest_rows.sort(key=lambda r: -float(r["Est. Tax Saved"].replace("$","").replace(",","")))
        st.dataframe(pd.DataFrame(harvest_rows), use_container_width=True, hide_index=True)
        st.success(
            f"Harvesting all losing positions could save up to **${total_harvestable:,.2f}** in federal tax "
            f"by offsetting gains. Remember the 30-day wash-sale window."
        )
    else:
        st.info("No losing positions to harvest right now.")

    # ── Tax-optimised exit order ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📌 Tax-Optimised Exit Order")
    st.caption(
        "If you need to raise cash, this is the order to sell your positions to minimise tax impact."
    )

    exit_rows = []
    for ticker, td in tax_data.items():
        gain      = td["unrealized_gain"]
        est_tax   = td["est_tax_if_sold_now"]
        mkt_val   = holdings[ticker].get("market_value", 0)
        tax_drag  = round(est_tax / mkt_val * 100, 1) if mkt_val else 0

        if td["tax_loss_harvest"]:
            priority = 1
            label    = "🥇 Sell first — harvests a loss"
        elif td["gain_type"] == "Long-term":
            priority = 2
            label    = "🥈 Sell second — long-term rate applies"
        elif td.get("days_to_lt") and td["days_to_lt"] <= 30:
            priority = 3
            label    = f"🕐 Wait {td['days_to_lt']}d — almost long-term"
        else:
            priority = 4
            label    = "🥉 Sell last — short-term gain, highest tax"

        exit_rows.append({
            "Priority":         priority,
            "Ticker":           ticker,
            "Market Value":     f"${mkt_val:,.2f}",
            "Unrealized P&L":   f"${gain:+,.2f}",
            "Est. Tax":         f"${est_tax:,.2f}" if gain > 0 else "—",
            "Tax Drag":         f"{tax_drag}%" if gain > 0 else "—",
            "Recommendation":   label,
        })

    exit_rows.sort(key=lambda r: r["Priority"])
    st.dataframe(pd.DataFrame(exit_rows), use_container_width=True, hide_index=True)

    st.caption(
        "⚠️ Federal estimates only. Does not account for state income tax, AMT, NIIT (net investment income tax "
        "for high earners), or wash-sale adjustments from prior periods. Consult a CPA before making tax-driven decisions."
    )


# ══════════════════════════════════════════════════════════════════════════════
# DAILY TRADE SCAN
# ══════════════════════════════════════════════════════════════════════════════
def show_daily_trade_scan():
    _scan_meta = {
        "day": {
            "title":   "⚡ Daily Trade Scan — Day Trading Mode",
            "desc":    (
                "**Finds stocks to buy and sell today.** "
                "MarketMind scans your full watchlist and flags stocks with volume spikes and intraday momentum — "
                "the two things day traders care most about.  \n\n"
                "Day trading means you enter and exit within the same market session (9:30 AM–4 PM ET). "
                "You need **tight stop-losses** (-0.75%) and **quick profit-taking** (+1.5%) — small, fast gains add up. "
                "Any position from today's scan should be **closed before market close**."
            ),
            "filter_note": "📊 *Day trading filter applied: only stocks with volume >1.5× average and clear momentum qualify.*",
        },
        "swing": {
            "title":   "📊 Daily Trade Scan — Swing Trading Mode",
            "desc":    (
                "**Finds stocks to buy and hold for 2–5 days.** "
                "MarketMind scans your watchlist for technical setups — RSI oversold bounces, MACD crossovers, "
                "and Bollinger Band breakouts — then sizes each trade so you never risk more than 2% of your capital.  \n\n"
                "Swing trading means you hold a position overnight and exit when it hits your take-profit (+5%) "
                "or stop-loss (-2%). You don't need to watch it every minute — just check once a day."
            ),
            "filter_note": "📊 *Swing filter applied: RSI, MACD, and momentum signals required.*",
        },
        "longterm": {
            "title":   "📈 Daily Trade Scan — Long-term Investing Mode",
            "desc":    (
                "**Finds quality stocks to hold for months.** "
                "MarketMind scans for stocks trading above their 200-day moving average (a proxy for long-term health) "
                "with solid analyst ratings and fundamental strength.  \n\n"
                "Long-term investing means you ignore short-term dips — you're buying the company, not the chart. "
                "A wide stop-loss (-8%) protects against real deterioration without getting shaken out by noise. "
                "Think of this as a monthly review, not a daily trade."
            ),
            "filter_note": "📊 *Long-term filter applied: stocks must be above 200-day EMA with positive analyst ratings.*",
        },
    }
    _sm = _scan_meta.get(_persona_key, _scan_meta["swing"])
    st.title(_sm["title"])
    st.markdown(_sm["desc"])
    st.caption(f"Scanning {len(mm.TRADING_UNIVERSE)} stocks — {date.today().strftime('%A, %B %d %Y')}")
    st.info(_sm["filter_note"])

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

        # ── New Opportunities shortcut ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("💡 New Opportunities")
        st.caption("Stocks beyond your holdings with bullish signals — scanned across 11 sectors (~165 stocks).")
        if st.button("Find New Opportunities →", key="btn_new_opps"):
            go_to("new_opportunities")

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
def show_stock_forecast():
    _sf_desc = {
        "day": (
            "**Research before you trade.** Heard about a stock on the news or social media? "
            "Drop the ticker in here and get a full breakdown: is the volume spiking? Is momentum building or fading? "
            "Day traders need to act fast — this tool gives you a quick STRONG BUY / AVOID verdict before you commit.  \n"
            "💡 *Tip: Check volume and momentum first. If volume is below average, skip it — day trading low-volume stocks is risky.*"
        ),
        "swing": (
            "**Deep-dive on any stock before you enter.** Heard about a stock and wondering if now is a good time to buy? "
            "Enter the ticker and get a full technical picture: RSI, MACD, Bollinger Bands, sentiment, and a STRONG BUY / AVOID verdict.  \n"
            "💡 *Tip: A stock is only interesting if RSI is below 60 and MACD is turning bullish — don't buy what's already overbought.*"
        ),
        "longterm": (
            "**Research any company before investing.** Thinking about buying a stock for the long haul? "
            "This tool shows you everything that matters for long-term investors: P/E ratio, analyst target price, profit margins, "
            "and whether the stock is in a long-term uptrend (above the 200-day moving average).  \n"
            "💡 *Tip: For long-term investing, fundamentals matter more than chart patterns. Focus on P/E ratio and analyst rating.*"
        ),
    }
    st.title(f"🔍 Stock Forecast — {active_profile}")
    st.markdown(_sf_desc.get(_persona_key, _sf_desc["swing"]))

    tab_recs, tab_analyse = st.tabs(["💡 Stock Recommendations", "🔎 Analyse a Stock"])

    # ── Tab 1: Stock Recommendations shortcut ────────────────────────────────
    with tab_recs:
        st.subheader("💡 Stock Recommendations")
        st.caption("Scans ~165 stocks across 11 sectors for bullish setups you don't already hold.")
        if st.button("Go to New Opportunities →", key="btn_sf_opps", type="primary", use_container_width=True):
            go_to("new_opportunities")

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
def show_top_10_snapshot():
    st.title("🏆 Top 10 US Stocks Snapshot")
    st.markdown(
        "**A live pulse check on the 10 biggest companies in the US stock market.** "
        "This includes Apple, Microsoft, NVIDIA, Amazon, Alphabet (Google), Meta, "
        "Berkshire Hathaway, Eli Lilly, Broadcom, and Tesla.  \n\n"
        "Each row shows the current price, where Wall Street analysts think it will be in 12 months (**analyst target**), "
        "how much it moved this week, and MarketMind's own 7-day rule-based prediction.  \n\n"
        "Think of this as a morning newspaper for the biggest stocks — a quick scan to see what's moving and what analysts expect."
    )
    st.caption("Data from Yahoo Finance + MarketMind's rule-based prediction engine. Predictions are for educational purposes only.")

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
def show_backtest():
    from datetime import timedelta
    st.title("🔬 Backtest MarketMind Predictions")
    st.markdown(
        "**See how accurate MarketMind's predictions were last week.**  \n\n"
        "MarketMind goes back 10 trading days and pretends it's running on that date — "
        "using only data that existed at the time (no cheating). It generates the same predictions "
        "it would have made, then compares those predictions to what actually happened.  \n\n"
        "This tells you: did MarketMind correctly predict whether a stock would go **up or down**? "
        "And by how much did the prediction miss?  \n\n"
        "💡 *No system is right 100% of the time. A 60–70% directional accuracy (up/down correct) "
        "is considered good — even professional fund managers often fall short of that.*"
    )

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


# ══════════════════════════════════════════════════════════════════════════════
# NEW OPPORTUNITIES
# ══════════════════════════════════════════════════════════════════════════════
def show_new_opportunities():
    st.title(f"💡 New Opportunities — {active_profile}")
    _cap = {
        "day":      "Scans ~165 stocks for volume spikes and intraday momentum setups — day-trade candidates.",
        "swing":    "Scans ~165 stocks across 11 sectors for bullish setups you don't already hold.",
        "longterm": "Scans ~165 stocks for quality companies above their 200-day EMA with strong analyst ratings.",
    }
    st.markdown(_cap.get(_persona_key, _cap["swing"]))

    if st.button("← Back to Daily Scan", key="btn_opps_back"):
        go_to("daily_trade_scan")

    st.markdown("---")
    held_tickers = list(profile_state.get("holdings", {}).keys())

    col1, col2 = st.columns([2, 1])
    with col1:
        run_opps = st.button("▶  Find Opportunities", type="primary", use_container_width=True, key="btn_run_opps")
    with col2:
        st.caption("Takes ~2 min (165 stocks)")

    if run_opps:
        with st.spinner("Scanning 165 stocks across 11 sectors..."):
            opps = mm.find_new_opportunities(held_tickers, n=15)

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
            st.info(
                "**Tax note on new buys:** Any position you open today starts the holding-period clock. "
                f"Short-term gains taxed at your rate ({int(tax_bracket)}%). "
                "Holding ≥1 year qualifies for lower long-term rates (0/15/20%)."
            )
        else:
            st.warning("No high-confidence opportunities found today. Market conditions may be unfavorable.")


# ── Router ────────────────────────────────────────────────────────────────────
_ROUTER = {
    "portfolio":         show_portfolio,
    "daily_trade_scan":  show_daily_trade_scan,
    "new_opportunities": show_new_opportunities,
    "pre_bell_scanner":  show_pre_bell_scanner,
    "ask_marketmind":    show_ask_marketmind,
    "tax_calculator":    show_tax_calculator,
    "stock_forecast":    show_stock_forecast,
    "top_10_snapshot":   show_top_10_snapshot,
    "backtest":          show_backtest,
}

_current_page = st.session_state.get("page", "portfolio")
if _current_page in _ROUTER:
    _ROUTER[_current_page]()
else:
    go_to("portfolio")

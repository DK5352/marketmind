"""
MarketMind — AI Trading Agent
==============================
An AI-powered active trading agent that scans the market daily, identifies
short-term trading opportunities, and gives you specific ENTER / EXIT / HOLD
trade instructions with entry price, take-profit target, and stop-loss level.

Trading style: Swing trading (hold 1–5 days per trade)
Focus: Momentum, breakouts, technical setups, news catalysts
Goal: Capture short-term price moves of 2–8% per trade

How it works
------------
- You set your starting capital in TRADING_CONFIG below.
- MarketMind runs daily and scans your trading universe for the best setups.
- For each opportunity it gives you: entry price, take-profit, stop-loss,
  position size, and a plain-English reason why.
- Open trades are tracked in portfolio_state.json. When a trade hits its
  take-profit or stop-loss, MarketMind tells you to exit.
- Profits are recycled back into new trades — capital compounds over time.

Trading rules built in
----------------------
- Never risk more than 2% of total capital on a single trade (stop-loss sizing)
- Never hold more than 5 open trades at once (focus over diversification)
- Exit any trade that drops more than STOP_LOSS_PCT from entry (capital protection)
- Take profits at TAKE_PROFIT_PCT — don't get greedy
- Avoid trading against the market trend (if S&P 500 is down hard, sit tight)

DISCLAIMER
----------
MarketMind provides AI-generated trade suggestions for educational purposes only.
It is NOT a licensed financial advisor. All trading decisions are yours alone.
Trading stocks carries significant risk — you can lose money quickly.
Never trade with money you cannot afford to lose.

How to use
----------
1. Set your STARTING_CAPITAL and trading parameters in TRADING_CONFIG below.
2. Edit TRADING_UNIVERSE to add/remove stocks you want the agent to scan.
3. Add your API keys to a .env file (see Required .env variables below).
4. Run once to set up:  python stock_friend.py
5. Run daily for trade signals (or set up a cron job — see Knowledge_Base.md).

Data Sources
------------
| Tool                     | Source                                           | Cost                     | Auth needed    |
|--------------------------|--------------------------------------------------|--------------------------|----------------|
| get_portfolio_state      | Local file (portfolio_state.json)                | Free                     | None           |
| get_weekly_return        | Yahoo Finance (yfinance)                         | Free                     | None           |
| get_benchmark_comparison | Yahoo Finance (yfinance)                         | Free                     | None           |
| get_technical_indicators | Yahoo Finance (yfinance)                         | Free                     | None           |
| get_sector_breakdown     | Yahoo Finance (yfinance)                         | Free                     | None           |
| get_fundamentals         | Yahoo Finance (yfinance)                         | Free                     | None           |
| get_stock_news           | Finnhub (finnhub.io)                             | Free tier (60 calls/min) | API key (.env) |
| get_stocktwits_sentiment | StockTwits (stocktwits.com)                      | Free                     | None           |
| get_reddit_mentions      | Reddit (r/wallstreetbets, r/stocks, r/investing) | Free                     | None           |
| get_income_statement     | Alpha Vantage (alphavantage.co)                  | Free tier (25 calls/day) | API key (.env) |
| get_balance_sheet        | Alpha Vantage (alphavantage.co)                  | Free tier (25 calls/day) | API key (.env) |
| get_earnings_history     | Alpha Vantage (alphavantage.co)                  | Free tier (25 calls/day) | API key (.env) |
| get_insider_trades       | SEC EDGAR (Form 4 filings)                       | Free                     | None           |
| get_sec_filings          | SEC EDGAR (10-K / 10-Q)                          | Free                     | None           |
| AI analysis              | Anthropic Claude                                 | Pay-per-use              | API key (.env) |

Required .env variables
-----------------------
    ANTHROPIC_API_KEY      — https://console.anthropic.com
    FINNHUB_API_KEY        — https://finnhub.io (free, no credit card)
    ALPHA_VANTAGE_API_KEY  — https://alphavantage.co (free, no credit card, 25 calls/day)

Install dependencies
--------------------
    pip install anthropic yfinance python-dotenv
"""

import json
import os
import random
import urllib.request
import urllib.error
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

import yfinance as yf
try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None  # type: ignore


# ── Ask MarketMind — conversational AI powered by Claude ─────────────────────

def ask_mastermind(
    question: str,
    portfolio_state: dict,
    conversation_history: list[dict] | None = None,
    ticker_context: str | None = None,
) -> str:
    """
    Ask the MarketMind AI a free-form question about markets, your portfolio,
    or any trading topic.

    Args:
        question:             The user's question in plain English.
        portfolio_state:      The active profile's portfolio_state dict.
        conversation_history: List of {"role": "user"|"assistant", "content": str}
                              from earlier in this session (for multi-turn).
        ticker_context:       Optional ticker the user is currently viewing
                              (adds live technicals to the prompt automatically).

    Returns:
        The AI's answer as a plain string.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return (
            "⚠️ ANTHROPIC_API_KEY not found. "
            "Add it to your .env file or Streamlit secrets to enable Ask Mastermind."
        )
    if _anthropic is None:
        return "⚠️ anthropic package not installed. Run: pip install anthropic"

    # ── Build portfolio context ───────────────────────────────────────────────
    holdings    = portfolio_state.get("holdings", {})
    cash        = portfolio_state.get("cash_available", 0)
    total_val   = portfolio_state.get("total_portfolio_value", 0)
    risk_level  = portfolio_state.get("risk_level", "moderate")
    tax_bracket = portfolio_state.get("tax_bracket_pct", 22)
    tp_pct      = portfolio_state.get("take_profit_pct", 5)
    sl_pct      = portfolio_state.get("stop_loss_pct", 2)
    total_ret   = portfolio_state.get("total_return_pct", 0)

    portfolio_lines = [
        f"Portfolio value: ${total_val:,.2f}  |  Cash: ${cash:,.2f}  |  "
        f"Total return: {total_ret:+.2f}%",
        f"Risk level: {risk_level}  |  Take-profit: +{tp_pct}%  |  "
        f"Stop-loss: -{sl_pct}%  |  Tax bracket: {int(tax_bracket)}%",
    ]
    if holdings:
        portfolio_lines.append("Current holdings:")
        for ticker, pos in holdings.items():
            pct = pos.get("unrealized_pct", 0)
            portfolio_lines.append(
                f"  {ticker}: {pos.get('shares', 0):.2f} shares  "
                f"avg ${pos.get('avg_buy_price', 0):.2f}  "
                f"now ${pos.get('current_price', pos.get('avg_buy_price', 0)):.2f}  "
                f"P&L {pct:+.1f}%"
            )
    else:
        portfolio_lines.append("No holdings yet.")

    # ── Optional: live ticker context ─────────────────────────────────────────
    ticker_lines = []
    if ticker_context:
        try:
            tech = get_technical_indicators(ticker_context)
            fund = get_fundamentals(ticker_context)
            if not tech.get("error"):
                ticker_lines = [
                    f"\nLive data for {ticker_context}:",
                    f"  Price: ${tech.get('current_price', 'N/A')}  "
                    f"RSI: {tech.get('rsi_14d', 'N/A')}  "
                    f"MACD: {tech.get('macd_signal', 'N/A')}",
                    f"  EMA200: ${tech.get('ema_200d', 'N/A')}  "
                    f"BB position: {tech.get('bb_price_position_pct', 'N/A')}th %ile",
                ]
            if not fund.get("error"):
                ticker_lines.append(
                    f"  P/E: {fund.get('pe_ratio', 'N/A')}  "
                    f"Analyst target: ${fund.get('analyst_target_price', 'N/A')}  "
                    f"Rating: {(fund.get('analyst_rating') or 'N/A').replace('_', ' ').title()}"
                )
        except Exception:
            pass

    # ── System prompt ─────────────────────────────────────────────────────────
    system_prompt = f"""You are MarketMind — a personal AI wealth manager and trading coach.
You speak like a knowledgeable friend who happens to be a stock market expert:
clear, direct, and jargon-free unless the user clearly knows the terms.

Today's date: {date.today().strftime('%B %d, %Y')}

USER'S PORTFOLIO CONTEXT:
{chr(10).join(portfolio_lines)}
{chr(10).join(ticker_lines)}

Your role:
- Answer questions about the user's portfolio, specific stocks, market conditions,
  trading strategies, options basics, tax implications, and investing concepts.
- Give specific, actionable answers — not generic disclaimers.
- When discussing tax (short-term vs long-term gains, wash-sale rule, etc.),
  use the user's actual tax bracket ({int(tax_bracket)}%).
- When asked about a stock they hold, reference their actual position.
- When asked about buy/sell decisions, frame it in terms of their risk level ({risk_level})
  and their take-profit (+{tp_pct}%) / stop-loss (-{sl_pct}%) thresholds.
- Use bullet points for multi-part answers. Keep responses concise (under 250 words)
  unless a topic genuinely needs depth.
- Always end with one concrete next step or key takeaway.
- Remind the user this is educational, not licensed financial advice — but only once
  per session, not on every reply.

STOCK MARKET EXPERT PERSPECTIVE:
- Pre-market conditions matter: gap risk, futures direction, VIX level.
- Technical signals are guides, not guarantees — always respect the stop-loss.
- Time horizon determines everything: a day trader and a long-term investor
  should react differently to the same signal.
- Tax efficiency is part of the return — factor it in.
- Position sizing (never risk more than 2% of capital per trade) is how
  professionals survive long-term.
"""

    # ── Build messages and call Claude ────────────────────────────────────────
    messages = list(conversation_history or [])
    messages.append({"role": "user", "content": question})

    try:
        client   = _anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 1024,
            system     = system_prompt,
            messages   = messages,
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"⚠️ MarketMind could not answer right now: {e}"


# ── Trading Configuration ─────────────────────────────────────────────────────
# Edit these values before your first run.

TRADING_CONFIG = {
    "starting_capital":       10000.00,  # USD — your starting trading capital
    "take_profit_pct":        5.0,       # Exit a trade when it gains this % (take profits)
    "stop_loss_pct":          2.0,       # Exit a trade when it loses this % (cut losses)
    "max_open_trades":        5,         # Never hold more than this many trades at once
    "max_position_pct":       20.0,      # Max % of total capital in any single trade
    "max_hold_days":          5,         # Exit any trade held longer than this (swing trading)
    "risk_per_trade_pct":     2.0,       # Max % of capital to risk on a single trade
    "trading_style":          "swing",   # "swing" (1-5 days) or "momentum" (days-weeks)
    "risk_level":             "moderate",# "conservative", "moderate", or "aggressive"
    "trading_persona":        "swing",   # "day", "swing", or "longterm"
}
# conservative = only high-predictability stocks, tight stop-losses
# moderate     = mix of setups, standard risk sizing
# aggressive   = momentum plays, wider stop-losses, higher position sizes

# ── Persona Configurations ────────────────────────────────────────────────────
# Each persona sets the default trading parameters and scoring weights.
# Stored per-profile in portfolio_state.json so different profiles can have
# different personas (e.g. one profile for day trading, one for long-term).

PERSONA_CONFIGS = {
    "day": {
        "label":            "Day Trading",
        "emoji":            "⚡",
        "description":      "Same-day exits. Fast, high-frequency setups. Exit before market close.",
        "tagline":          "Hit fast, exit same day. Small gains compound — protect every cent.",
        "take_profit_pct":  1.5,    # Small, quick gains
        "stop_loss_pct":    0.75,   # Very tight — day traders can't afford big losses
        "max_open_trades":  3,      # Fewer trades, more focus
        "max_hold_days":    1,      # Exit same day
        "risk_per_trade_pct": 1.0,  # Smaller risk per trade
        "max_position_pct": 15.0,
        # Indicator weights for scoring (higher = more important)
        "weights": {
            "volume":       3,   # Volume spikes are the #1 day-trading signal
            "momentum":     2,   # Pre-market gap / price momentum
            "rsi":          1,   # Short-term RSI matters less intraday
            "macd":         1,
            "ema200":       0,   # 200-day EMA irrelevant for same-day trades
            "fundamentals": 0,   # Fundamentals don't matter within a day
            "sentiment":    1,
        },
        "focus":   ["volume spike", "pre-market gap", "momentum breakout", "RSI extremes"],
        "avoid":   ["low volume stocks", "stocks below 200-day EMA for multi-day holds"],
        "hold_label": "Same day",
        "target_label": "+1–2% per trade",
    },
    "swing": {
        "label":            "Swing Trading",
        "emoji":            "🔄",
        "description":      "Hold 2–5 days. Capture short-term price moves of 3–8%.",
        "tagline":          "Ride the wave, not the whole ocean. Enter on setups, exit with discipline.",
        "take_profit_pct":  5.0,
        "stop_loss_pct":    2.0,
        "max_open_trades":  5,
        "max_hold_days":    5,
        "risk_per_trade_pct": 2.0,
        "max_position_pct": 20.0,
        "weights": {
            "volume":       1,
            "momentum":     2,
            "rsi":          2,   # RSI setups are the core of swing trading
            "macd":         2,   # MACD crossovers signal swing entries
            "ema200":       1,
            "fundamentals": 1,
            "sentiment":    1,
        },
        "focus":   ["RSI oversold bounce", "MACD bullish crossover", "Bollinger Band squeeze breakout"],
        "avoid":   ["stocks with no clear technical setup", "earnings week entries (gap risk)"],
        "hold_label": "2–5 days",
        "target_label": "+3–8% per trade",
    },
    "longterm": {
        "label":            "Long-term Investing",
        "emoji":            "📈",
        "description":      "Hold 3–12 months. Focus on fundamentals, sector trends, and analyst consensus.",
        "tagline":          "Buy quality, hold with conviction. Ignore short-term noise — let compounding work.",
        "take_profit_pct":  25.0,   # Large target — let winners run
        "stop_loss_pct":    8.0,    # Wider stop — normal pullbacks won't stop you out
        "max_open_trades":  10,     # More diversification
        "max_hold_days":    180,    # 6-month default hold window
        "risk_per_trade_pct": 5.0,  # Larger position sizes — fewer, higher-conviction trades
        "max_position_pct": 15.0,   # Slightly lower per-stock cap for diversification
        "weights": {
            "volume":       0,   # Volume noise irrelevant over months
            "momentum":     1,
            "rsi":          0,   # Short-term RSI is noise for long-term investors
            "macd":         1,
            "ema200":       3,   # 200-day EMA is the most important long-term trend signal
            "fundamentals": 3,   # P/E, analyst rating, profit margin drive long-term returns
            "sentiment":    1,
        },
        "focus":   ["above 200-day EMA", "strong analyst rating", "low P/E relative to sector",
                    "positive revenue growth", "high profit margin"],
        "avoid":   ["overvalued stocks (P/E > 50)", "companies with negative profit margins",
                    "stocks in long-term downtrends (below 200-day EMA)"],
        "hold_label": "3–12 months",
        "target_label": "+15–30% per position",
    },
}


def apply_persona(persona_key: str) -> None:
    """
    Applies a persona's settings to TRADING_CONFIG.
    Called when loading a profile or changing persona.
    """
    cfg = PERSONA_CONFIGS.get(persona_key, PERSONA_CONFIGS["swing"])
    TRADING_CONFIG["trading_persona"]    = persona_key
    TRADING_CONFIG["take_profit_pct"]    = cfg["take_profit_pct"]
    TRADING_CONFIG["stop_loss_pct"]      = cfg["stop_loss_pct"]
    TRADING_CONFIG["max_open_trades"]    = cfg["max_open_trades"]
    TRADING_CONFIG["max_hold_days"]      = cfg["max_hold_days"]
    TRADING_CONFIG["risk_per_trade_pct"] = cfg["risk_per_trade_pct"]
    TRADING_CONFIG["max_position_pct"]   = cfg["max_position_pct"]

# Keep a reference to the original name for backward compatibility
INVESTMENT_CONFIG = TRADING_CONFIG

# ── Trading Universe ──────────────────────────────────────────────────────────
# Stocks the agent is allowed to trade.
# Chosen for liquidity (easy to buy and sell quickly) and volatility
# (enough price movement to make short-term trades worthwhile).
# Edit this list to add or remove candidates.

TRADING_UNIVERSE = [
    # High momentum / volatile — good for swing trades
    "NVDA",   # NVIDIA       — high beta, strong momentum
    "TSLA",   # Tesla        — volatile, news-driven
    "META",   # Meta         — earnings catalyst plays
    "AMZN",   # Amazon       — liquid, range-bound swings
    "GOOGL",  # Alphabet     — stable momentum
    # Technology
    "AAPL",   # Apple        — liquid, predictable patterns
    "MSFT",   # Microsoft    — reliable trend-following
    "CRM",    # Salesforce   — earnings volatility
    # Finance
    "JPM",    # JPMorgan     — macro-sensitive
    "BAC",    # Bank of America — rate-sensitive swings
    "V",      # Visa         — steady momentum
    # Consumer
    "WMT",    # Walmart      — defensive, breakout plays
    "COST",   # Costco       — strong trend stock
    # Healthcare
    "UNH",    # UnitedHealth — strong trending
    "PFE",    # Pfizer       — news/catalyst driven
]

# Backward compatibility alias
INVESTMENT_UNIVERSE = TRADING_UNIVERSE

# ── Broad opportunity universe (sector → tickers) ─────────────────────────────
# Used by find_new_opportunities() to scan beyond the user's current holdings.
BROAD_UNIVERSE: dict[str, list[str]] = {
    "Technology":            ["AAPL","MSFT","NVDA","GOOGL","META","CRM","ORCL","ADBE","AMD","QCOM","AVGO","SNOW","PLTR","NOW","PANW"],
    "Healthcare":            ["UNH","JNJ","LLY","ABBV","MRK","PFE","TMO","ABT","AMGN","GILD","ISRG","DXCM","VRTX","REGN","ZTS"],
    "Finance":               ["JPM","BAC","V","MA","GS","MS","BLK","AXP","COF","SCHW","PNC","TFC","ICE","MCO","SPGI"],
    "Consumer Discretionary":["AMZN","TSLA","HD","MCD","NKE","SBUX","BKNG","LOW","TGT","COST","MAR","HLT","ETSY","RCL","YUM"],
    "Energy":                ["XOM","CVX","COP","SLB","OXY","EOG","PSX","VLO","MPC","KMI","DVN","HES","BKR","HAL","OKE"],
    "Industrials":           ["HON","GE","CAT","DE","BA","LMT","RTX","UPS","FDX","WM","CTAS","MMM","ITW","EMR","ETN"],
    "Communication":         ["GOOGL","META","NFLX","DIS","CMCSA","T","VZ","TMUS","EA","TTWO","SNAP","PINS","ZM","MTCH","WBD"],
    "Consumer Staples":      ["PG","KO","PEP","WMT","MO","PM","CL","KMB","GIS","K","CHD","CLX","HRL","CAG","SJM"],
    "Materials":             ["LIN","APD","SHW","ECL","NEM","FCX","NUE","STLD","VMC","MLM","DOW","DD","PPG","ALB","CF"],
    "Utilities":             ["NEE","DUK","SO","D","AEP","EXC","SRE","XEL","ED","ES","WEC","ETR","FE","PPL","CMS"],
    "Real Estate":           ["AMT","PLD","CCI","EQIX","PSA","O","WELL","DLR","AVB","EQR","SPG","VTR","SBA","INVH","ARE"],
}

# Path to the trading state file (auto-created on first run)
PORTFOLIO_STATE_FILE = os.path.join(os.path.dirname(__file__), "portfolio_state.json")


# ── Portfolio state management ────────────────────────────────────────────────

def _load_state() -> dict:
    """Load portfolio state from disk. Returns default state if file doesn't exist."""
    if os.path.exists(PORTFOLIO_STATE_FILE):
        with open(PORTFOLIO_STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_state(state: dict) -> None:
    """Persist portfolio state to disk."""
    with open(PORTFOLIO_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def initialize_portfolio() -> dict:
    """
    Creates a fresh portfolio_state.json with the user's initial investment.
    Called automatically on first run if no state file exists.
    """
    cfg = TRADING_CONFIG
    capital = cfg["starting_capital"]
    today = date.today().isoformat()
    end_date = (date.today() + relativedelta(months=6)).isoformat()

    persona_key = cfg.get("trading_persona", "swing")
    persona_cfg = PERSONA_CONFIGS.get(persona_key, PERSONA_CONFIGS["swing"])

    state = {
        "initial_investment": capital,
        "cash_available": capital,
        "total_portfolio_value": capital,
        "start_date": today,
        "end_date": end_date,
        "investment_period_months": 6,
        "weekly_return_target_pct": 1.0,
        "risk_level": cfg["risk_level"],
        "trading_persona": persona_key,
        "persona_label": persona_cfg["label"],
        "holdings": {},
        "transaction_history": [],
        "realized_gains": 0.0,
        "unrealized_gains": 0.0,
        "total_return_pct": 0.0,
    }
    _save_state(state)
    return state


# ── Tool implementations ────────────────────────────────────────────────────


def get_weekly_return(ticker: str) -> dict:
    """
    SOURCE: Yahoo Finance — via yfinance (free, no API key required)
    Calculates the 7-day price return for a single ticker.
    Fetches 10 days of history to account for weekends and market holidays.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="10d")
        if len(hist) < 2:
            return {"error": f"Insufficient price data for {ticker}"}

        current_price = hist["Close"].iloc[-1]
        week_ago_price = hist["Close"].iloc[0]
        weekly_return_pct = ((current_price - week_ago_price) / week_ago_price) * 100

        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "price_7d_ago": round(week_ago_price, 2),
            "weekly_return_pct": round(weekly_return_pct, 2),
            "period_start": str(hist.index[0].date()),
            "period_end": str(hist.index[-1].date()),
        }
    except Exception as e:
        return {"error": str(e)}


def get_benchmark_comparison() -> dict:
    """
    SOURCE: Yahoo Finance — via yfinance (free, no API key required)
    Fetches the 7-day return of major market indices:
      ^GSPC = S&P 500
      ^IXIC = NASDAQ Composite
      ^DJI  = Dow Jones Industrial Average
    Used to put your portfolio's weekly performance in market context.
    """
    results = {}
    for name, symbol in [("S&P 500", "^GSPC"), ("NASDAQ", "^IXIC"), ("DOW", "^DJI")]:
        try:
            hist = yf.Ticker(symbol).history(period="10d")
            if len(hist) < 2:
                continue
            start = hist["Close"].iloc[0]
            end = hist["Close"].iloc[-1]
            results[name] = {
                "weekly_return_pct": round(((end - start) / start) * 100, 2),
                "current_value": round(end, 2),
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    return results


def get_technical_indicators(ticker: str) -> dict:
    """
    SOURCE: Yahoo Finance — via yfinance (free, no API key required)
    Calculates technical signals using 6 months of daily closing prices:

      - RSI (14-day): momentum oscillator.
          >70 = overbought, <30 = oversold, 30-70 = neutral

      - 50-day MA & 200-day MA: trend direction indicators.
          Price above MA = bullish, below = bearish

      - MACD (Moving Average Convergence Divergence):
          MACD line   = 12-day EMA minus 26-day EMA
          Signal line = 9-day EMA of MACD line
          Histogram   = MACD minus Signal (positive = bullish momentum)

      - Bollinger Bands (20-day, 2 std deviations):
          Upper = 20-day SMA + 2x std dev
          Lower = 20-day SMA - 2x std dev
          Price near upper = potentially overbought
          Price near lower = potentially oversold

      - Stochastic Oscillator (14-period):
          %K = (Close - 14-period Low) / (14-period High - Low) * 100
          %D = 3-day SMA of %K
          >80 = overbought, <20 = oversold

      - Volume Spike Detection:
          Compares latest volume to 20-day average
          >2x average = major spike, >1.5x = moderate spike

      - Beta & Average Volume: sourced from Yahoo Finance info endpoint
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        info = stock.info

        if len(hist) < 26:
            return {"error": "Not enough data for technical indicators (need 26+ trading days)"}

        close = hist["Close"]
        volume = hist["Volume"]
        current = round(float(close.iloc[-1]), 2)

        # ── RSI (14-day) ──────────────────────────────────────────────────────
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi = round(float(100 - (100 / (1 + rs.iloc[-1]))), 1)
        rsi_signal = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"

        # ── Moving Averages (SMA) ─────────────────────────────────────────────
        ma50 = round(float(close.rolling(50).mean().iloc[-1]), 2)
        ma200 = round(float(close.rolling(200).mean().iloc[-1]), 2) if len(hist) >= 200 else None
        ma_signal = "above 50-day avg (bullish)" if current > ma50 else "below 50-day avg (bearish)"

        # ── EMAs (Exponential Moving Averages) ────────────────────────────────
        # EMA gives more weight to recent prices than a simple moving average.
        # 20-day EMA: short-term trend — traders watch for price crossing above/below
        # 200-day EMA: long-term trend — price above = bull market, below = bear market
        ema20_val = round(float(close.ewm(span=20, adjust=False).mean().iloc[-1]), 2)
        ema200_val = round(float(close.ewm(span=200, adjust=False).mean().iloc[-1]), 2) if len(hist) >= 200 else None
        ema20_signal = "price above EMA20 (short-term bullish)" if current > ema20_val else "price below EMA20 (short-term bearish)"
        ema200_signal = (
            "price above EMA200 (long-term bullish)" if ema200_val and current > ema200_val
            else "price below EMA200 (long-term bearish)" if ema200_val
            else "insufficient data for EMA200"
        )

        # ── MACD ─────────────────────────────────────────────────────────────
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        macd_val = round(float(macd_line.iloc[-1]), 4)
        signal_val = round(float(signal_line.iloc[-1]), 4)
        histogram_val = round(float(histogram.iloc[-1]), 4)
        macd_signal = (
            "bullish crossover" if macd_val > signal_val and histogram_val > 0
            else "bearish crossover" if macd_val < signal_val and histogram_val < 0
            else "neutral"
        )

        # ── Bollinger Bands (20-day, 2 std dev) ──────────────────────────────
        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = round(float((bb_sma + 2 * bb_std).iloc[-1]), 2)
        bb_middle = round(float(bb_sma.iloc[-1]), 2)
        bb_lower = round(float((bb_sma - 2 * bb_std).iloc[-1]), 2)
        bb_bandwidth = round((bb_upper - bb_lower) / bb_middle * 100, 2)
        bb_pct = round((current - bb_lower) / (bb_upper - bb_lower) * 100, 1) if bb_upper != bb_lower else 50.0
        bb_signal = (
            "near upper band (overbought)" if bb_pct > 80
            else "near lower band (oversold)" if bb_pct < 20
            else "within bands (neutral)"
        )

        # ── Stochastic Oscillator (14-period) ────────────────────────────────
        low14 = hist["Low"].rolling(14).min()
        high14 = hist["High"].rolling(14).max()
        pct_k = round(float(((close - low14) / (high14 - low14) * 100).iloc[-1]), 1)
        pct_d = round(float(((close - low14) / (high14 - low14) * 100).rolling(3).mean().iloc[-1]), 1)
        stoch_signal = "overbought" if pct_k > 80 else "oversold" if pct_k < 20 else "neutral"

        # ── Volume Spike Detection ────────────────────────────────────────────
        avg_vol_20d = volume.rolling(20).mean().iloc[-1]
        latest_vol = volume.iloc[-1]
        volume_ratio = round(float(latest_vol / avg_vol_20d), 2) if avg_vol_20d else None
        volume_signal = (
            "major spike (>2x avg)" if volume_ratio and volume_ratio > 2
            else "moderate spike (>1.5x avg)" if volume_ratio and volume_ratio > 1.5
            else "normal"
        )

        return {
            "ticker": ticker,
            "current_price": current,
            # RSI
            "rsi_14d": rsi,
            "rsi_signal": rsi_signal,
            # Simple Moving Averages (SMA)
            "ma_50d": ma50,
            "ma_200d": ma200,
            "ma_signal": ma_signal,
            # Exponential Moving Averages (EMA)
            "ema_20d": ema20_val,
            "ema_200d": ema200_val,
            "ema_20d_signal": ema20_signal,
            "ema_200d_signal": ema200_signal,
            # MACD
            "macd": macd_val,
            "macd_signal_line": signal_val,
            "macd_histogram": histogram_val,
            "macd_signal": macd_signal,
            # Bollinger Bands
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "bb_bandwidth_pct": bb_bandwidth,
            "bb_price_position_pct": bb_pct,
            "bb_signal": bb_signal,
            # Stochastic Oscillator
            "stoch_k": pct_k,
            "stoch_d": pct_d,
            "stoch_signal": stoch_signal,
            # Volume
            "latest_volume": int(latest_vol),
            "avg_volume_20d": int(avg_vol_20d) if avg_vol_20d else None,
            "volume_ratio": volume_ratio,
            "volume_signal": volume_signal,
            # From Yahoo Finance info endpoint
            "beta": info.get("beta"),
            "avg_volume": info.get("averageVolume"),
        }
    except Exception as e:
        return {"error": str(e)}


def get_sector_breakdown(tickers: list) -> dict:
    """
    SOURCE: Yahoo Finance — via yfinance (free, no API key required)
    Looks up the sector for each ticker (e.g. Technology, Healthcare) and
    shows how the watchlist is distributed across sectors.
    Helps beginners understand industry diversification — spreading stocks
    across different sectors reduces risk if one industry has a bad week.
    """
    try:
        sectors = {}
        total = len(tickers)

        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                sector = info.get("sector", "Unknown")
                industry = info.get("industry", "Unknown")
            except Exception:
                sector = "Unknown"
                industry = "Unknown"

            if sector not in sectors:
                sectors[sector] = {"tickers": [], "industries": []}
            sectors[sector]["tickers"].append(ticker)
            if industry not in sectors[sector]["industries"]:
                sectors[sector]["industries"].append(industry)

        for sector in sectors:
            count = len(sectors[sector]["tickers"])
            sectors[sector]["stock_count"] = count
            sectors[sector]["watchlist_pct"] = round(count / total * 100, 1)

        return {"total_stocks_in_watchlist": total, "sectors": sectors}
    except Exception as e:
        return {"error": str(e)}


def get_fundamentals(ticker: str) -> dict:
    """
    SOURCE: Yahoo Finance — via yfinance (free, no API key required)
    Pulls company fundamentals from Yahoo Finance's info endpoint.
    All fields are as reported by Yahoo Finance; some may be delayed or
    estimated (e.g. forwardPE uses analyst consensus estimates).

    Fields returned:
      Valuation   — P/E, forward P/E, price-to-book, price-to-sales
      Size        — market cap, trailing 12-month revenue
      Profitability — profit margin, EPS, return on equity
      Dividends   — yield and annual dividend rate
      Price range — 52-week high/low, 50-day and 200-day moving averages
      Growth      — YoY earnings and revenue growth
      Analysts    — consensus rating (strong_buy/buy/hold/sell), mean price target,
                    number of analyst opinions
      Earnings    — next earnings date (Unix timestamp from Yahoo)
    """
    try:
        info = yf.Ticker(ticker).info

        def safe(key, default=None):
            val = info.get(key, default)
            return None if val in (None, "N/A", float("inf"), float("-inf")) else val

        return {
            "ticker": ticker,
            "company_name": safe("longName"),
            "sector": safe("sector"),
            "industry": safe("industry"),
            # Valuation — sourced from Yahoo Finance trailing/forward estimates
            "pe_ratio": safe("trailingPE"),
            "forward_pe": safe("forwardPE"),
            "price_to_book": safe("priceToBook"),
            "price_to_sales": safe("priceToSalesTrailing12Months"),
            # Size & profitability
            "market_cap": safe("marketCap"),
            "revenue_ttm": safe("totalRevenue"),
            "profit_margin": safe("profitMargins"),
            "earnings_per_share": safe("trailingEps"),
            "return_on_equity": safe("returnOnEquity"),
            # Dividends
            "dividend_yield": safe("dividendYield"),
            "dividend_rate": safe("dividendRate"),
            # 52-week price context
            "52w_high": safe("fiftyTwoWeekHigh"),
            "52w_low": safe("fiftyTwoWeekLow"),
            "50d_avg": safe("fiftyDayAverage"),
            "200d_avg": safe("twoHundredDayAverage"),
            # Balance sheet ratios
            "debt_to_equity": safe("debtToEquity"),
            "total_debt": safe("totalDebt"),
            "total_cash": safe("totalCash"),
            # YoY growth rates (trailing)
            "earnings_growth_yoy": safe("earningsGrowth"),
            "revenue_growth_yoy": safe("revenueGrowth"),
            # Analyst consensus — aggregated by Yahoo from sell-side analysts
            "analyst_rating": safe("recommendationKey"),
            "analyst_target_price": safe("targetMeanPrice"),
            "num_analyst_opinions": safe("numberOfAnalystOpinions"),
            # Next earnings date (Unix timestamp — convert with datetime.fromtimestamp)
            "next_earnings_date": safe("earningsTimestamp"),
        }
    except Exception as e:
        return {"error": str(e)}


def compute_tax_implications(holdings: dict, tax_bracket_pct: float = 22.0) -> dict:
    """
    For each holding, compute:
      - holding_days       : days held so far
      - gain_type          : "Long-term" (≥365 days) or "Short-term" (<365 days)
      - days_to_lt         : days remaining until long-term threshold (0 if already LT)
      - unrealized_gain    : current gain ($)
      - est_tax_if_sold_now: estimated tax owed if sold today
      - effective_rate     : rate used (long-term = min(20%, bracket), short-term = bracket)
      - tax_loss_harvest   : True if position has a loss (candidate for harvesting)

    tax_bracket_pct is the user's marginal income tax rate (e.g. 22, 32, 37).
    Long-term capital gains rate: 0% if bracket ≤ 12%, 15% if ≤ 35%, 20% if > 35%.
    Short-term capital gains are taxed as ordinary income at tax_bracket_pct.
    """
    lt_rate = 0.0 if tax_bracket_pct <= 12 else (0.15 if tax_bracket_pct <= 35 else 0.20)
    st_rate = tax_bracket_pct / 100.0
    today   = date.today()

    result = {}
    for ticker, pos in holdings.items():
        buy_date_str  = pos.get("buy_date", "")
        unrealized    = pos.get("unrealized_pnl", 0.0)
        unrealized_pct= pos.get("unrealized_pct", 0.0)

        holding_days = None
        days_to_lt   = None
        gain_type    = "Unknown"

        if buy_date_str:
            try:
                bd = date.fromisoformat(buy_date_str)
                holding_days = (today - bd).days
                if holding_days >= 365:
                    gain_type  = "Long-term"
                    days_to_lt = 0
                else:
                    gain_type  = "Short-term"
                    days_to_lt = 365 - holding_days
            except Exception:
                pass

        rate = lt_rate if gain_type == "Long-term" else st_rate
        est_tax = round(max(0.0, unrealized * rate), 2)

        result[ticker] = {
            "holding_days":        holding_days,
            "gain_type":           gain_type,
            "days_to_lt":          days_to_lt,
            "unrealized_gain":     round(unrealized, 2),
            "unrealized_pct":      round(unrealized_pct, 2),
            "est_tax_if_sold_now": est_tax,
            "effective_rate_pct":  round(rate * 100, 1),
            "tax_loss_harvest":    unrealized < 0,
        }
    return result


def get_stock_news(ticker: str, days: int = 7) -> dict:
    """
    SOURCE: Finnhub (finnhub.io) — free tier, 60 API calls/minute
    Requires FINNHUB_API_KEY in .env. Get a free key at https://finnhub.io.

    Fetches up to 5 recent company news articles for the past `days` days.
    Returns headline, truncated summary (200 chars), source outlet, and date.
    Used by Claude to explain *why* a stock moved up or down during the week.
    """
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        return {"error": "FINNHUB_API_KEY not set in .env"}

    date_to = datetime.today().strftime("%Y-%m-%d")
    date_from = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")

    url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={ticker}&from={date_from}&to={date_to}&token={api_key}"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            articles = json.loads(resp.read().decode())

        top = articles[:5]
        return {
            "ticker": ticker,
            "period": f"{date_from} to {date_to}",
            "articles": [
                {
                    "headline": a.get("headline", ""),
                    "summary": a.get("summary", "")[:200],
                    "source": a.get("source", ""),
                    "date": datetime.fromtimestamp(a.get("datetime", 0)).strftime("%Y-%m-%d"),
                }
                for a in top
            ],
        }
    except Exception as e:
        return {"error": str(e)}


def get_income_statement(ticker: str) -> dict:
    """
    SOURCE: Alpha Vantage (alphavantage.co) — free tier, 25 calls/day
    Requires ALPHA_VANTAGE_API_KEY in .env. Get a free key at https://alphavantage.co.

    Returns the two most recent annual income statements:
      - Total revenue, gross profit, operating income, net income, EBITDA
    Use alongside Yahoo Finance fundamentals for deeper profitability analysis.
    """
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        return {"error": "ALPHA_VANTAGE_API_KEY not set in .env"}

    url = (
        f"https://www.alphavantage.co/query"
        f"?function=INCOME_STATEMENT&symbol={ticker}&apikey={api_key}"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        reports = data.get("annualReports", [])[:2]
        if not reports:
            return {"error": f"No income statement data for {ticker}"}

        def _num(val):
            try:
                return int(val) if val not in (None, "None", "N/A") else None
            except (ValueError, TypeError):
                return None

        result = []
        for r in reports:
            result.append({
                "fiscal_year_end": r.get("fiscalDateEnding"),
                "total_revenue": _num(r.get("totalRevenue")),
                "gross_profit": _num(r.get("grossProfit")),
                "operating_income": _num(r.get("operatingIncome")),
                "net_income": _num(r.get("netIncome")),
                "ebitda": _num(r.get("ebitda")),
                "research_and_development": _num(r.get("researchAndDevelopment")),
            })

        return {"ticker": ticker, "annual_income_statements": result}
    except Exception as e:
        return {"error": str(e)}


def get_balance_sheet(ticker: str) -> dict:
    """
    SOURCE: Alpha Vantage (alphavantage.co) — free tier, 25 calls/day
    Requires ALPHA_VANTAGE_API_KEY in .env.

    Returns the two most recent annual balance sheets:
      - Total assets, total liabilities, total shareholder equity
      - Total debt, cash and equivalents, retained earnings
    Useful for assessing financial health, leverage, and liquidity.
    """
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        return {"error": "ALPHA_VANTAGE_API_KEY not set in .env"}

    url = (
        f"https://www.alphavantage.co/query"
        f"?function=BALANCE_SHEET&symbol={ticker}&apikey={api_key}"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        reports = data.get("annualReports", [])[:2]
        if not reports:
            return {"error": f"No balance sheet data for {ticker}"}

        def _num(val):
            try:
                return int(val) if val not in (None, "None", "N/A") else None
            except (ValueError, TypeError):
                return None

        result = []
        for r in reports:
            result.append({
                "fiscal_year_end": r.get("fiscalDateEnding"),
                "total_assets": _num(r.get("totalAssets")),
                "total_liabilities": _num(r.get("totalLiabilities")),
                "shareholder_equity": _num(r.get("totalShareholderEquity")),
                "total_debt": _num(r.get("shortLongTermDebtTotal")),
                "cash_and_equivalents": _num(r.get("cashAndCashEquivalentsAtCarryingValue")),
                "retained_earnings": _num(r.get("retainedEarnings")),
                "long_term_debt": _num(r.get("longTermDebtNoncurrent")),
            })

        return {"ticker": ticker, "annual_balance_sheets": result}
    except Exception as e:
        return {"error": str(e)}


def get_earnings_history(ticker: str) -> dict:
    """
    SOURCE: Alpha Vantage (alphavantage.co) — free tier, 25 calls/day
    Requires ALPHA_VANTAGE_API_KEY in .env.

    Returns the 8 most recent quarterly earnings reports:
      - Reported EPS vs estimated EPS, surprise amount and surprise %
    A consistent earnings beat pattern is a bullish signal.
    A miss, especially with guidance cuts, is typically bearish.
    """
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        return {"error": "ALPHA_VANTAGE_API_KEY not set in .env"}

    url = (
        f"https://www.alphavantage.co/query"
        f"?function=EARNINGS&symbol={ticker}&apikey={api_key}"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        reports = data.get("quarterlyEarnings", [])[:8]
        if not reports:
            return {"error": f"No earnings data for {ticker}"}

        def _flt(val):
            try:
                return float(val) if val not in (None, "None", "N/A") else None
            except (ValueError, TypeError):
                return None

        result = []
        for r in reports:
            result.append({
                "fiscal_quarter_end": r.get("fiscalDateEnding"),
                "reported_date": r.get("reportedDate"),
                "reported_eps": _flt(r.get("reportedEPS")),
                "estimated_eps": _flt(r.get("estimatedEPS")),
                "surprise": _flt(r.get("surprise")),
                "surprise_pct": _flt(r.get("surprisePercentage")),
            })

        return {"ticker": ticker, "quarterly_earnings": result}
    except Exception as e:
        return {"error": str(e)}


def get_stocktwits_sentiment(ticker: str) -> dict:
    """
    SOURCE: StockTwits (stocktwits.com) — completely free, no API key required
    Public API endpoint: https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json

    Returns the 30 most recent StockTwits posts for a ticker.
    Users self-tag posts as Bullish or Bearish, giving us real sentiment signals.

    Returns:
      - bullish_count / bearish_count / untagged_count across the 30 most recent messages
      - bullish_pct — % of tagged posts that are bullish
      - sentiment_label — "Bullish", "Bearish", or "Neutral" based on ratio
      - top_messages — up to 5 recent message bodies with their sentiment tag
    """
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "StockFriend/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        messages = data.get("messages", [])
        bullish = sum(1 for m in messages if m.get("entities", {}).get("sentiment", {}) and m["entities"]["sentiment"].get("basic") == "Bullish")
        bearish = sum(1 for m in messages if m.get("entities", {}).get("sentiment", {}) and m["entities"]["sentiment"].get("basic") == "Bearish")
        untagged = len(messages) - bullish - bearish
        tagged = bullish + bearish
        bullish_pct = round(bullish / tagged * 100, 1) if tagged else None

        if bullish_pct is None:
            sentiment_label = "Neutral"
        elif bullish_pct >= 60:
            sentiment_label = "Bullish"
        elif bullish_pct <= 40:
            sentiment_label = "Bearish"
        else:
            sentiment_label = "Neutral"

        top_messages = []
        for m in messages[:5]:
            sentiment_tag = None
            if m.get("entities", {}).get("sentiment"):
                sentiment_tag = m["entities"]["sentiment"].get("basic")
            top_messages.append({
                "body": m.get("body", "")[:280],
                "sentiment": sentiment_tag,
                "created_at": m.get("created_at", ""),
                "username": m.get("user", {}).get("username", ""),
            })

        return {
            "ticker": ticker,
            "total_messages_sampled": len(messages),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "untagged_count": untagged,
            "bullish_pct": bullish_pct,
            "sentiment_label": sentiment_label,
            "top_messages": top_messages,
        }
    except Exception as e:
        return {"error": str(e)}


def get_reddit_mentions(ticker: str) -> dict:
    """
    SOURCE: Reddit — public JSON search API, no API key required
    Searches r/wallstreetbets, r/stocks, and r/investing for recent posts
    mentioning the ticker symbol.

    Uses Reddit's public search endpoint:
      https://www.reddit.com/r/{subreddit}/search.json?q={ticker}&sort=new&limit=10

    Returns:
      - mention_count — total posts found across all three subreddits
      - per-subreddit breakdown with top post titles, scores, and comment counts
    High mention counts (especially in r/wallstreetbets) can signal retail momentum.
    """
    subreddits = ["wallstreetbets", "stocks", "investing"]
    headers = {"User-Agent": "StockFriend/1.0"}
    results = {}
    total_mentions = 0

    for sub in subreddits:
        url = (
            f"https://www.reddit.com/r/{sub}/search.json"
            f"?q={ticker}&sort=new&limit=10&restrict_sr=1"
        )
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            posts = data.get("data", {}).get("children", [])
            top_posts = []
            for p in posts[:3]:
                pd = p.get("data", {})
                top_posts.append({
                    "title": pd.get("title", "")[:200],
                    "score": pd.get("score", 0),
                    "num_comments": pd.get("num_comments", 0),
                    "created_utc": datetime.utcfromtimestamp(pd.get("created_utc", 0)).strftime("%Y-%m-%d"),
                    "url": f"https://reddit.com{pd.get('permalink', '')}",
                })

            count = len(posts)
            total_mentions += count
            results[f"r/{sub}"] = {
                "mention_count": count,
                "top_posts": top_posts,
            }
        except Exception as e:
            results[f"r/{sub}"] = {"error": str(e)}

    return {
        "ticker": ticker,
        "total_mentions_across_subreddits": total_mentions,
        "subreddits": results,
    }


def _get_cik(ticker: str) -> str | None:
    """
    SOURCE: SEC EDGAR — https://www.sec.gov/files/company_tickers.json
    Looks up the CIK (Central Index Key) for a ticker symbol.
    The CIK is required for all subsequent EDGAR API calls.
    No API key needed; SEC asks for a descriptive User-Agent header.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "StockFriend/1.0 contact@example.com"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            tickers_data = json.loads(resp.read().decode())

        ticker_upper = ticker.upper()
        for entry in tickers_data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                # CIK must be zero-padded to 10 digits for EDGAR submission endpoint
                return str(entry["cik_str"]).zfill(10)
    except Exception:
        pass
    return None


def get_insider_trades(ticker: str) -> dict:
    """
    SOURCE: SEC EDGAR (sec.gov) — completely free, no API key required
    SEC requests a descriptive User-Agent header (set below).

    Fetches the most recent Form 4 filings (insider buy/sell transactions).
    Form 4 must be filed within 2 business days of any trade by officers,
    directors, or >10% shareholders.

    Returns up to 10 recent filings with: filer name, form type, date filed.
    A cluster of insider buys is often a bullish signal; heavy selling can be bearish
    (though insiders sell for many reasons — diversification, taxes, etc.).
    """
    cik = _get_cik(ticker)
    if not cik:
        return {"error": f"Could not find CIK for {ticker} on SEC EDGAR"}

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "StockFriend/1.0 contact@example.com"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        descriptions = filings.get("primaryDocument", [])

        # Filter to Form 4 filings only
        form4_filings = []
        for i, form in enumerate(forms):
            if form == "4":
                acc = accessions[i].replace("-", "")
                form4_filings.append({
                    "form": form,
                    "filed_date": dates[i],
                    "accession_number": accessions[i],
                    "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{descriptions[i]}",
                })
                if len(form4_filings) >= 10:
                    break

        return {
            "ticker": ticker,
            "cik": cik,
            "company_name": data.get("name", ""),
            "recent_form4_filings": form4_filings,
            "note": "Form 4 = insider transaction (officer/director buy or sell within 2 business days)",
        }
    except Exception as e:
        return {"error": str(e)}


def get_sec_filings(ticker: str) -> dict:
    """
    SOURCE: SEC EDGAR (sec.gov) — completely free, no API key required
    SEC requests a descriptive User-Agent header (set below).

    Returns the 5 most recent 10-K (annual) and 10-Q (quarterly) filings
    with direct links to the SEC EDGAR viewer.

    10-K = annual report — audited financials, risk factors, business overview
    10-Q = quarterly report — unaudited quarterly financials and MD&A
    These are the primary source of truth for a company's financial condition.
    """
    cik = _get_cik(ticker)
    if not cik:
        return {"error": f"Could not find CIK for {ticker} on SEC EDGAR"}

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "StockFriend/1.0 contact@example.com"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        descriptions = filings.get("primaryDocument", [])
        report_dates = filings.get("reportDate", [])

        annual_quarterly = []
        for i, form in enumerate(forms):
            if form in ("10-K", "10-Q"):
                acc = accessions[i].replace("-", "")
                annual_quarterly.append({
                    "form": form,
                    "filed_date": dates[i],
                    "period": report_dates[i] if i < len(report_dates) else None,
                    "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{descriptions[i]}",
                })
                if len(annual_quarterly) >= 5:
                    break

        return {
            "ticker": ticker,
            "cik": cik,
            "company_name": data.get("name", ""),
            "recent_10k_10q_filings": annual_quarterly,
        }
    except Exception as e:
        return {"error": str(e)}


def get_portfolio_state() -> dict:
    """
    SOURCE: Local file — portfolio_state.json (created on first run)
    Returns the current state of the portfolio:
      - cash_available: uninvested cash ready to deploy
      - holdings: each stock currently held with shares, avg buy price, and current value
      - total_portfolio_value: cash + market value of all holdings
      - realized_gains: profit locked in from past sales
      - unrealized_gains: paper gains/losses on current holdings
      - total_return_pct: overall return since start
      - days_remaining: days left in the 6-month investment period
    Always call this first so you know exactly what we have before making recommendations.
    """
    state = _load_state()
    if not state:
        return {"error": "Portfolio not initialised. Run initialize_portfolio() first."}

    # Enrich holdings with live prices and unrealized P&L
    holdings_enriched = {}
    total_market_value = 0.0
    total_cost_basis = 0.0

    for ticker, pos in state.get("holdings", {}).items():
        try:
            current_price = float(yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1])
        except Exception:
            current_price = pos.get("avg_buy_price", 0)

        shares = pos.get("shares", 0)
        avg_buy = pos.get("avg_buy_price", 0)
        cost_basis = shares * avg_buy
        market_value = shares * current_price
        unrealized_pnl = market_value - cost_basis
        unrealized_pct = (unrealized_pnl / cost_basis * 100) if cost_basis else 0

        holdings_enriched[ticker] = {
            "shares": shares,
            "avg_buy_price": round(avg_buy, 2),
            "current_price": round(current_price, 2),
            "cost_basis": round(cost_basis, 2),
            "market_value": round(market_value, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "unrealized_pct": round(unrealized_pct, 2),
            "buy_date": pos.get("buy_date", ""),
        }
        total_market_value += market_value
        total_cost_basis += cost_basis

    cash = state.get("cash_available", 0)
    total_value = cash + total_market_value
    initial = state.get("initial_investment", total_value)
    total_return_pct = ((total_value - initial) / initial * 100) if initial else 0

    # Days remaining in investment period
    try:
        end = date.fromisoformat(state["end_date"])
        days_remaining = max(0, (end - date.today()).days)
        weeks_remaining = round(days_remaining / 7, 1)
    except Exception:
        days_remaining = None
        weeks_remaining = None

    return {
        "initial_investment": state.get("initial_investment"),
        "cash_available": round(cash, 2),
        "total_portfolio_value": round(total_value, 2),
        "total_return_pct": round(total_return_pct, 2),
        "realized_gains": round(state.get("realized_gains", 0), 2),
        "unrealized_gains": round(total_market_value - total_cost_basis, 2),
        "holdings": holdings_enriched,
        "start_date": state.get("start_date"),
        "end_date": state.get("end_date"),
        "days_remaining": days_remaining,
        "weeks_remaining": weeks_remaining,
        "weekly_return_target_pct": state.get("weekly_return_target_pct"),
        "risk_level": state.get("risk_level"),
        "recent_transactions": state.get("transaction_history", [])[-5:],
    }


def record_transactions(transactions: list) -> dict:
    """
    SOURCE: Local file — portfolio_state.json
    Records the day's executed BUY/SELL transactions into portfolio state.
    Call this after presenting recommendations to the user and they confirm which
    ones they acted on.

    Each transaction in the list should have:
      action      — "BUY" or "SELL"
      ticker      — stock symbol
      shares      — number of shares
      price       — execution price per share
      date        — date executed (YYYY-MM-DD), defaults to today

    The function updates cash_available, holdings, realized_gains, and
    appends to transaction_history.
    """
    state = _load_state()
    if not state:
        return {"error": "Portfolio not initialised."}

    results = []
    for tx in transactions:
        action = tx.get("action", "").upper()
        ticker = tx.get("ticker", "").upper()
        shares = float(tx.get("shares", 0))
        price = float(tx.get("price", 0))
        tx_date = tx.get("date", date.today().isoformat())
        total = round(shares * price, 2)

        if action == "BUY":
            if total > state["cash_available"]:
                results.append({"error": f"Insufficient cash for BUY {ticker}: need ${total}, have ${state['cash_available']:.2f}"})
                continue
            state["cash_available"] = round(state["cash_available"] - total, 2)
            if ticker in state["holdings"]:
                existing = state["holdings"][ticker]
                old_cost = existing["shares"] * existing["avg_buy_price"]
                new_shares = existing["shares"] + shares
                state["holdings"][ticker] = {
                    "shares": round(new_shares, 6),
                    "avg_buy_price": round((old_cost + total) / new_shares, 4),
                    "buy_date": existing["buy_date"],
                }
            else:
                state["holdings"][ticker] = {
                    "shares": round(shares, 6),
                    "avg_buy_price": round(price, 4),
                    "buy_date": tx_date,
                }
            results.append({"status": "ok", "action": "BUY", "ticker": ticker, "shares": shares, "price": price, "total": total})

        elif action == "SELL":
            if ticker not in state["holdings"]:
                results.append({"error": f"Cannot SELL {ticker}: not in holdings"})
                continue
            held = state["holdings"][ticker]
            if shares > held["shares"]:
                results.append({"error": f"Cannot SELL {shares} shares of {ticker}: only hold {held['shares']}"})
                continue
            proceeds = round(shares * price, 2)
            cost = round(shares * held["avg_buy_price"], 2)
            gain = round(proceeds - cost, 2)
            state["cash_available"] = round(state["cash_available"] + proceeds, 2)
            state["realized_gains"] = round(state.get("realized_gains", 0) + gain, 2)
            remaining = round(held["shares"] - shares, 6)
            if remaining <= 0.0001:
                del state["holdings"][ticker]
            else:
                state["holdings"][ticker]["shares"] = remaining
            results.append({"status": "ok", "action": "SELL", "ticker": ticker, "shares": shares, "price": price, "proceeds": proceeds, "realized_gain": gain})

        state["transaction_history"].append({
            "date": tx_date,
            "action": action,
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "total": total,
        })

    _save_state(state)
    return {"transactions_recorded": len(results), "results": results, "cash_after": state["cash_available"]}


# ── Tool definitions (passed to Claude so it knows what tools exist) ─────────

TOOLS = [
    {
        "name": "get_portfolio_state",
        # Data source: local portfolio_state.json file
        "description": (
            "Fetch the current portfolio state: cash available, all holdings with live prices "
            "and unrealized P&L, total portfolio value, total return since start, days remaining "
            "in the investment period, and the last 5 transactions. "
            "Always call this first before making any recommendations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "record_transactions",
        # Data source: local portfolio_state.json file
        "description": (
            "Record BUY or SELL transactions the user has executed into portfolio_state.json. "
            "Call this after the user confirms they have acted on your recommendations. "
            "Updates cash balance, holdings, and realized gains automatically. "
            "Pass a list of transactions, each with: action (BUY/SELL), ticker, shares, price, date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "transactions": {
                    "type": "array",
                    "description": "List of executed transactions to record",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "description": "BUY or SELL"},
                            "ticker": {"type": "string", "description": "Stock ticker symbol"},
                            "shares": {"type": "number", "description": "Number of shares"},
                            "price": {"type": "number", "description": "Execution price per share"},
                            "date": {"type": "string", "description": "Date executed YYYY-MM-DD"},
                        },
                        "required": ["action", "ticker", "shares", "price"],
                    },
                }
            },
            "required": ["transactions"],
        },
    },
    {
        "name": "get_weekly_return",
        # Data source: Yahoo Finance via yfinance
        "description": (
            "Calculate the 7-day price return for a single stock ticker. "
            "Call this for current holdings and top candidates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL or TSLA",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_benchmark_comparison",
        # Data source: Yahoo Finance via yfinance (^GSPC, ^IXIC, ^DJI)
        "description": (
            "Get weekly returns for S&P 500, NASDAQ, and DOW. "
            "Use this to compare the user's portfolio performance against the broader market."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_technical_indicators",
        # Data source: Yahoo Finance via yfinance (price history + info endpoint)
        "description": (
            "Get RSI (14-day), 50-day and 200-day moving averages, beta, and signals "
            "for a single ticker. Use to flag overbought/oversold conditions and trend direction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL or TSLA",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_sector_breakdown",
        # Data source: Yahoo Finance via yfinance (sector field from info endpoint)
        "description": (
            "Given a list of ticker symbols, show how they are distributed across "
            "market sectors (e.g. Technology, Healthcare, Finance). "
            "Helps identify whether the watchlist is diversified or concentrated in one area."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols to analyse, e.g. ['AAPL', 'TSLA']",
                }
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "get_fundamentals",
        # Data source: Yahoo Finance via yfinance (info endpoint — mix of reported
        # and estimated data; forward-looking fields use analyst consensus)
        "description": (
            "Fetch key fundamental data for a stock: P/E ratio, forward P/E, "
            "price-to-book, market cap, profit margin, dividend yield, 52-week "
            "high/low, analyst rating, price target, and next earnings date. "
            "Use this to assess whether a stock is over/undervalued and its overall health."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL or TSLA",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_stocktwits_sentiment",
        # Data source: StockTwits (stocktwits.com) — completely free, no API key required
        "description": (
            "Fetch StockTwits sentiment for a ticker: bullish vs bearish post counts, "
            "bullish %, overall sentiment label (Bullish/Bearish/Neutral), and top 5 recent messages. "
            "Users self-tag posts, making this a reliable retail sentiment signal. "
            "Use to gauge crowd mood and spot momentum shifts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL or TSLA",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_reddit_mentions",
        # Data source: Reddit public JSON API — free, no API key required
        # Covers: r/wallstreetbets, r/stocks, r/investing
        "description": (
            "Search Reddit for recent posts mentioning a ticker across r/wallstreetbets, "
            "r/stocks, and r/investing. Returns mention count per subreddit and top post titles, "
            "scores, and comment counts. High WSB mention counts can signal retail momentum or hype."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL or TSLA",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_insider_trades",
        # Data source: SEC EDGAR Form 4 filings — completely free, no API key required
        "description": (
            "Fetch recent Form 4 insider trading filings from SEC EDGAR for a stock. "
            "Form 4 must be filed within 2 business days of any buy/sell by officers, "
            "directors, or >10% shareholders. "
            "A cluster of insider buys is often bullish; heavy selling can be a warning sign."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL or TSLA",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_sec_filings",
        # Data source: SEC EDGAR — completely free, no API key required
        "description": (
            "Fetch recent 10-K (annual) and 10-Q (quarterly) SEC filings for a stock. "
            "Returns filing dates and direct links to SEC EDGAR. "
            "Use when you need authoritative financial data or risk disclosures."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL or TSLA",
                }
            },
            "required": ["ticker"],
        },
    },
]


# ── Tool router ───────────────────────────────────────────────────────────────

def execute_tool(name: str, tool_input: dict) -> str:
    if name == "get_portfolio_state":
        result = get_portfolio_state()
    elif name == "record_transactions":
        result = record_transactions(tool_input["transactions"])
    elif name == "get_weekly_return":
        result = get_weekly_return(tool_input["ticker"])
    elif name == "get_benchmark_comparison":
        result = get_benchmark_comparison()
    elif name == "get_technical_indicators":
        result = get_technical_indicators(tool_input["ticker"])
    elif name == "get_sector_breakdown":
        result = get_sector_breakdown(tool_input["tickers"])
    elif name == "get_fundamentals":
        result = get_fundamentals(tool_input["ticker"])
    elif name == "get_stock_news":
        result = get_stock_news(tool_input["ticker"])
    elif name == "get_stocktwits_sentiment":
        result = get_stocktwits_sentiment(tool_input["ticker"])
    elif name == "get_reddit_mentions":
        result = get_reddit_mentions(tool_input["ticker"])
    elif name == "get_income_statement":
        result = get_income_statement(tool_input["ticker"])
    elif name == "get_balance_sheet":
        result = get_balance_sheet(tool_input["ticker"])
    elif name == "get_earnings_history":
        result = get_earnings_history(tool_input["ticker"])
    elif name == "get_insider_trades":
        result = get_insider_trades(tool_input["ticker"])
    elif name == "get_sec_filings":
        result = get_sec_filings(tool_input["ticker"])
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result)


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent() -> str:
    """
    Rule-based daily trading scan. Completely free — no API calls.
    Scans the trading universe and outputs ENTER/EXIT/HOLD signals.
    """
    cfg = TRADING_CONFIG
    today_str = datetime.today().strftime("%A, %B %d %Y")

    state = _load_state()
    if not state:
        state = initialize_portfolio()

    total_capital  = state.get("total_portfolio_value", cfg["starting_capital"])
    cash           = state.get("cash_available", total_capital)
    holdings       = state.get("holdings", {})
    take_profit_pct = cfg["take_profit_pct"]
    stop_loss_pct   = cfg["stop_loss_pct"]
    max_open        = cfg["max_open_trades"]
    risk_pct        = cfg["risk_per_trade_pct"]
    max_pos_pct     = cfg["max_position_pct"]

    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  MARKETMIND DAILY TRADE SCAN  —  {today_str}")
    lines.append(f"{'='*70}")

    # Account snapshot
    initial     = state.get("initial_investment", total_capital)
    realized    = state.get("realized_gains", 0)
    total_ret   = ((total_capital - initial) / initial * 100) if initial else 0
    lines.append(f"\n## ACCOUNT SNAPSHOT")
    lines.append(f"  Total capital:      ${total_capital:,.2f}")
    lines.append(f"  Cash available:     ${cash:,.2f}")
    lines.append(f"  Open trades:        {len(holdings)}/{max_open}")
    lines.append(f"  Return since start: {total_ret:+.2f}%  |  Realized gains: ${realized:,.2f}")

    # Market conditions
    print("  Checking market conditions...")
    benchmark   = get_benchmark_comparison()
    market_weak = False
    lines.append(f"\n## MARKET CONDITIONS")
    for name, data in benchmark.items():
        if data.get("error"):
            lines.append(f"  {name}: unavailable")
        else:
            wk    = data.get("weekly_return_pct", 0)
            arrow = "▲" if wk >= 0 else "▼"
            lines.append(f"  {name}: {arrow} {abs(wk):.2f}% this week  (current: {data.get('current_value','N/A')})")
            if name == "S&P 500" and wk < -1.5:
                market_weak = True
    mood = ("⚠  RISK-OFF — market is weak. Be very selective with new entries."
            if market_weak else
            "✓  RISK-ON — market conditions are supportive for new entries.")
    lines.append(f"\n  Market mood: {mood}")

    # Check open trades
    lines.append(f"\n## OPEN TRADES — ACTION REQUIRED")
    if not holdings:
        lines.append("  No open trades.")
    else:
        for ticker, pos in holdings.items():
            current_price = pos.get("current_price", pos["avg_buy_price"])
            avg_buy       = pos["avg_buy_price"]
            shares        = pos["shares"]
            pnl_pct       = ((current_price - avg_buy) / avg_buy) * 100
            pnl_dollar    = (current_price - avg_buy) * shares

            days_held = 0
            if pos.get("buy_date"):
                try:
                    days_held = (date.today() - date.fromisoformat(pos["buy_date"])).days
                except Exception:
                    pass

            if pnl_pct >= take_profit_pct:
                action = f"EXIT — TAKE PROFIT HIT (+{pnl_pct:.1f}%) 🎯  Lock in ${pnl_dollar:,.2f}"
            elif pnl_pct <= -stop_loss_pct:
                action = f"EXIT — STOP LOSS HIT ({pnl_pct:.1f}%) 🛑  Cut loss of ${abs(pnl_dollar):,.2f}"
            elif days_held >= cfg["max_hold_days"]:
                action = f"EXIT — MAX HOLD TIME ({days_held} days) ⏰  P&L: {pnl_pct:+.1f}%"
            else:
                tp_price = round(avg_buy * (1 + take_profit_pct / 100), 2)
                sl_price = round(avg_buy * (1 - stop_loss_pct  / 100), 2)
                action   = (f"HOLD — P&L: {pnl_pct:+.1f}% (${pnl_dollar:+,.2f})  "
                            f"|  TP: ${tp_price:.2f}  |  SL: ${sl_price:.2f}")

            lines.append(f"\n  {ticker}  |  Bought @ ${avg_buy:.2f}  →  Now ${current_price:.2f}")
            lines.append(f"  Action: {action}")

    # Scan for new setups
    lines.append(f"\n## NEW TRADE SETUPS")
    open_tickers = set(holdings.keys())
    candidates   = []
    passed       = []

    if len(holdings) >= max_open:
        lines.append(f"  Already at max open trades ({max_open}). Close a position before entering a new one.")
    elif cash < total_capital * 0.05:
        lines.append(f"  Insufficient cash (${cash:.2f}) for new trades.")
    else:
        scannable = [t for t in TRADING_UNIVERSE if t not in open_tickers]
        print(f"  Scanning {len(scannable)} stocks...")

        for ticker in scannable:
            print(f"    {ticker}...", end=" ", flush=True)
            try:
                tech = get_technical_indicators(ticker)
                if tech.get("error"):
                    passed.append((ticker, f"Data unavailable"))
                    print("skip")
                    continue

                rsi         = tech.get("rsi_14d", 50)
                macd_sig    = tech.get("macd_signal", "neutral")
                macd_hist   = tech.get("macd_histogram", 0)
                bb_pct      = tech.get("bb_price_position_pct", 50)
                ema20       = tech.get("ema_20d")
                ema200      = tech.get("ema_200d")
                vol_ratio   = tech.get("volume_ratio") or 1.0
                cur_price   = tech.get("current_price")

                # Hard avoids
                if rsi > 72:
                    passed.append((ticker, f"RSI overbought ({rsi})"))
                    print("avoid")
                    continue
                if ema200 and cur_price and cur_price < ema200:
                    passed.append((ticker, "Below 200-day EMA (long-term downtrend)"))
                    print("avoid")
                    continue
                if market_weak:
                    passed.append((ticker, "Market risk-off — skipping new entries"))
                    print("skip")
                    continue

                stw = get_stocktwits_sentiment(ticker)
                stw_label   = stw.get("sentiment_label", "Neutral") if not stw.get("error") else "N/A"
                stw_bull    = stw.get("bullish_pct") if not stw.get("error") else None

                signal = None
                reason = []

                # OVERSOLD BOUNCE
                if rsi < 35 and bb_pct < 25:
                    signal = "OVERSOLD BOUNCE"
                    reason = [f"RSI={rsi} (oversold)", f"Near lower Bollinger band ({bb_pct:.0f}%ile)"]

                # MOMENTUM
                if not signal:
                    bull = 0
                    if 40 <= rsi <= 65:              bull += 1
                    if macd_sig == "bullish crossover": bull += 2
                    elif macd_hist > 0:              bull += 1
                    if ema20 and cur_price and cur_price > ema20: bull += 1
                    if vol_ratio >= 1.5:             bull += 1
                    if stw_bull and stw_bull > 60:   bull += 1

                    if bull >= 3:
                        signal = "MOMENTUM"
                        if macd_sig == "bullish crossover": reason.append("MACD bullish crossover")
                        if 40 <= rsi <= 65: reason.append(f"RSI={rsi} (healthy)")
                        if ema20 and cur_price and cur_price > ema20: reason.append("above EMA20")
                        if vol_ratio >= 1.5: reason.append(f"volume {vol_ratio:.1f}x avg")
                        if stw_bull and stw_bull > 60: reason.append(f"StockTwits {stw_bull}% bullish")

                if signal and cur_price:
                    sl_dist      = cur_price * (stop_loss_pct / 100)
                    risk_dollars = total_capital * (risk_pct / 100)
                    shares_risk  = risk_dollars / sl_dist
                    max_pos_val  = total_capital * (max_pos_pct / 100)
                    shares       = max(1, int(min(shares_risk, max_pos_val / cur_price, cash / cur_price)))
                    pos_val      = shares * cur_price
                    tp_price     = round(cur_price * (1 + take_profit_pct / 100), 2)
                    sl_price     = round(cur_price * (1 - stop_loss_pct  / 100), 2)

                    candidates.append({
                        "ticker":   ticker,
                        "signal":   signal,
                        "price":    cur_price,
                        "tp":       tp_price,
                        "sl":       sl_price,
                        "shares":   shares,
                        "pos_val":  pos_val,
                        "pos_pct":  pos_val / total_capital * 100,
                        "reason":   ", ".join(reason),
                        "rsi":      rsi,
                        "macd":     macd_sig,
                        "vol":      vol_ratio,
                        "sentiment": stw_label,
                    })
                    print("SIGNAL ✓")
                else:
                    passed.append((ticker, "No qualifying setup — mixed signals"))
                    print("pass")

            except Exception as e:
                passed.append((ticker, f"Error: {e}"))
                print("error")

        # Best setups first — MOMENTUM before BOUNCE; fill available slots
        candidates.sort(key=lambda x: 0 if x["signal"] == "MOMENTUM" else 1)
        slots      = max_open - len(holdings)
        new_trades = candidates[:slots]

        if new_trades:
            for t in new_trades:
                lines.append(f"\n  ✅ ENTER — {t['ticker']}")
                lines.append(f"  Setup:         {t['signal']}")
                lines.append(f"  Entry price:   ${t['price']:.2f}")
                lines.append(f"  Take-profit:   ${t['tp']:.2f}  (+{take_profit_pct}%)"
                             f"  ← exit here to lock in the gain")
                lines.append(f"  Stop-loss:     ${t['sl']:.2f}  (-{stop_loss_pct}%)"
                             f"  ← exit immediately if it drops here")
                lines.append(f"  Position size: {t['shares']} shares = ${t['pos_val']:,.2f}"
                             f"  ({t['pos_pct']:.1f}% of capital)")
                lines.append(f"  Max hold:      {cfg['max_hold_days']} days")
                lines.append(f"  Why:           {t['reason']}")
                lines.append(f"  RSI: {t['rsi']}  |  MACD: {t['macd']}  "
                             f"|  Volume: {t['vol']:.1f}x  |  Sentiment: {t['sentiment']}")
        else:
            lines.append("  No qualifying setups found today.")
            lines.append("  Cash is a position — sitting tight is a valid choice.")

        # Stocks scanned but skipped
        if passed:
            lines.append(f"\n## PASSED — NO TRADE TODAY")
            for tkr, rsn in passed:
                lines.append(f"  {tkr:<6}  {rsn}")

    # Daily tip
    TIPS = [
        "The stop-loss is your safety net. If a trade moves against you by the stop-loss %, exit immediately — no exceptions.",
        "RSI above 70 means a stock may be overbought and due for a pullback. Avoid buying at these levels.",
        "Volume confirms moves. A price rise on high volume is more convincing than one on low volume.",
        "MACD bullish crossover means short-term momentum is turning positive — a good early entry signal.",
        "Never risk more than 2% of capital on a single trade. Even 10 losses in a row won't wipe you out.",
        "Cash is a position. When conditions are uncertain, staying in cash is a valid strategy.",
        "Bollinger lower band = potential support zone. Upper band = potential resistance zone.",
        "A green day for the S&P 500 makes all swing trades easier — always check the market mood first.",
    ]
    lines.append(f"\n## BEGINNER TIP OF THE DAY")
    lines.append(f"  {random.choice(TIPS)}")

    lines.append(f"\n{'='*70}")
    lines.append("  DISCLAIMER: Signals are for educational purposes only.")
    lines.append("  Always do your own research before placing real trades.")
    lines.append(f"{'='*70}\n")

    return "\n".join(lines)


def _save_daily_report(report: str) -> str:
    """Save the daily report to a dated file in the marketmind directory."""
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    filename = os.path.join(reports_dir, f"report_{date.today().isoformat()}.md")
    with open(filename, "w") as f:
        f.write(f"# MarketMind Daily Report — {date.today().strftime('%B %d, %Y')}\n\n")
        f.write(report)
    return filename


def find_new_opportunities(held_tickers: list[str], n: int = 10) -> list[dict]:
    """
    Scan BROAD_UNIVERSE for new buy candidates beyond the user's current holdings.
    Returns up to `n` opportunities ranked by bull_score descending.

    Each result includes: ticker, sector, signal, bull_score, price, rsi, macd,
    volume_ratio, sentiment, take_profit, stop_loss, reason.
    """
    held = {t.upper() for t in held_tickers}
    cfg  = TRADING_CONFIG

    candidates = []

    for sector, tickers in BROAD_UNIVERSE.items():
        for ticker in tickers:
            if ticker in held:
                continue
            try:
                tech = get_technical_indicators(ticker)
                if tech.get("error"):
                    continue

                rsi       = tech.get("rsi_14d", 50)
                macd_sig  = tech.get("macd_signal", "neutral")
                macd_hist = tech.get("macd_histogram", 0)
                bb_pct    = tech.get("bb_price_position_pct", 50)
                ema20     = tech.get("ema_20d")
                ema200    = tech.get("ema_200d")
                vol_ratio = tech.get("volume_ratio") or 1.0
                cur       = tech.get("current_price")

                if not cur:
                    continue
                if rsi > 75:
                    continue
                if ema200 and cur < ema200:
                    continue

                stw      = get_stocktwits_sentiment(ticker)
                stw_bull = stw.get("bullish_pct") if not stw.get("error") else None
                stw_lbl  = stw.get("sentiment_label", "Neutral") if not stw.get("error") else "N/A"

                bull = 0
                reasons = []

                if rsi < 35 and bb_pct < 25:
                    bull += 3
                    reasons.append(f"oversold (RSI={rsi}, BB={bb_pct:.0f}%ile)")

                if macd_sig == "bullish crossover":
                    bull += 2
                    reasons.append("MACD bullish crossover")
                elif macd_hist > 0:
                    bull += 1
                    reasons.append("MACD positive")

                if 40 <= rsi <= 65:
                    bull += 1
                    reasons.append(f"RSI={rsi} healthy")

                if ema20 and cur > ema20:
                    bull += 1
                    reasons.append("above EMA20")

                if vol_ratio >= 1.5:
                    bull += 1
                    reasons.append(f"volume {vol_ratio:.1f}x avg")

                if stw_bull and stw_bull > 60:
                    bull += 1
                    reasons.append(f"StockTwits {stw_bull}% bullish")

                if bull < 3:
                    continue

                signal = "OVERSOLD BOUNCE" if (rsi < 35 and bb_pct < 25) else "MOMENTUM"
                tp = round(cur * (1 + cfg["take_profit_pct"] / 100), 2)
                sl = round(cur * (1 - cfg["stop_loss_pct"]   / 100), 2)

                candidates.append({
                    "ticker":      ticker,
                    "sector":      sector,
                    "signal":      signal,
                    "bull_score":  bull,
                    "price":       cur,
                    "rsi":         rsi,
                    "macd":        macd_sig,
                    "volume":      vol_ratio,
                    "sentiment":   stw_lbl,
                    "take_profit": tp,
                    "stop_loss":   sl,
                    "reason":      ", ".join(reasons),
                })
            except Exception:
                continue

    candidates.sort(key=lambda x: x["bull_score"], reverse=True)
    return candidates[:n]


def get_portfolio_sector_analysis(held_tickers: list[str]) -> dict:
    """
    Fetch sector for each held ticker and return:
      - sector_weights: {sector: pct_of_portfolio}
      - overweight: sectors at >25% concentration
      - underweight: sectors with 0% representation from BROAD_UNIVERSE keys
      - suggestion: plain-English rebalancing note
    """
    if not held_tickers:
        return {"error": "No holdings to analyse."}

    sector_map: dict[str, list[str]] = {}
    for ticker in held_tickers:
        try:
            info   = yf.Ticker(ticker).info
            sector = info.get("sector") or "Unknown"
        except Exception:
            sector = "Unknown"
        sector_map.setdefault(sector, []).append(ticker)

    total = len(held_tickers)
    sector_weights = {
        s: {"tickers": tickers, "count": len(tickers), "pct": round(len(tickers) / total * 100, 1)}
        for s, tickers in sector_map.items()
    }

    overweight = [s for s, v in sector_weights.items() if v["pct"] > 35]
    represented = set(sector_map.keys())
    broad_sectors = set(BROAD_UNIVERSE.keys())
    underweight = sorted(broad_sectors - represented)

    suggestions = []
    for s in overweight:
        suggestions.append(f"Reduce concentration in {s} ({sector_weights[s]['pct']}% of portfolio).")
    for s in underweight[:3]:
        suggestions.append(f"No exposure to {s} — consider adding diversification.")
    if not suggestions:
        suggestions.append("Portfolio is reasonably diversified across sectors.")

    return {
        "total_holdings":  total,
        "sector_weights":  sector_weights,
        "overweight":      overweight,
        "underweight":     underweight,
        "suggestions":     suggestions,
    }


def run_stock_forecast(ticker: str) -> str:
    """
    Rule-based single-stock analysis. Completely free — no API calls.
    Answers: "Should I buy this stock right now?"
    """
    ticker     = ticker.upper()
    today_str  = datetime.today().strftime("%A, %B %d %Y")
    cfg        = TRADING_CONFIG

    print(f"  Fetching data for {ticker}...")
    tech     = get_technical_indicators(ticker)
    fund     = get_fundamentals(ticker)
    weekly   = get_weekly_return(ticker)
    stw      = get_stocktwits_sentiment(ticker)
    reddit   = get_reddit_mentions(ticker)
    insider  = get_insider_trades(ticker)

    if tech.get("error"):
        return f"Could not fetch data for {ticker}: {tech['error']}"

    cur        = tech.get("current_price")
    rsi        = tech.get("rsi_14d", 50)
    macd_sig   = tech.get("macd_signal", "neutral")
    macd_hist  = tech.get("macd_histogram", 0)
    bb_pct     = tech.get("bb_price_position_pct", 50)
    ema20      = tech.get("ema_20d")
    ema200     = tech.get("ema_200d")
    vol_ratio  = tech.get("volume_ratio") or 1.0
    ma50       = tech.get("ma_50d")
    wkly_ret   = weekly.get("weekly_return_pct", 0) if not weekly.get("error") else 0

    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  STOCK FORECAST: {ticker}  —  {today_str}")
    lines.append(f"{'='*60}")

    name = fund.get("company_name", ticker) if not fund.get("error") else ticker
    lines.append(f"\n  {name}")
    lines.append(f"  Current price:  ${cur:.2f}" if cur else "  Current price:  N/A")
    lines.append(f"  7-day return:   {wkly_ret:+.2f}%")

    lines.append(f"\n## TECHNICAL SIGNALS")
    lines.append(f"  RSI (14d):      {rsi}  →  {tech.get('rsi_signal','N/A')}")
    lines.append(f"  MACD:           {macd_sig}  (histogram: {macd_hist:+.4f})")
    lines.append(f"  Bollinger:      {bb_pct:.0f}th percentile  →  {tech.get('bb_signal','N/A')}")
    lines.append(f"  EMA 20d:        ${ema20:.2f}  →  {tech.get('ema_20d_signal','N/A')}" if ema20 else "  EMA 20d:        N/A")
    lines.append(f"  EMA 200d:       ${ema200:.2f}  →  {tech.get('ema_200d_signal','N/A')}" if ema200 else "  EMA 200d:       N/A")
    lines.append(f"  50d SMA:        ${ma50:.2f}" if ma50 else "  50d SMA:        N/A")
    lines.append(f"  Volume:         {vol_ratio:.1f}x average  →  {tech.get('volume_signal','N/A')}")
    lines.append(f"  Stochastic %K:  {tech.get('stoch_k','N/A')}")

    if not fund.get("error"):
        lines.append(f"\n## FUNDAMENTALS")
        lines.append(f"  Sector:         {fund.get('sector','N/A')}")
        lines.append(f"  P/E:            {fund.get('pe_ratio','N/A')}  |  Forward P/E: {fund.get('forward_pe','N/A')}")
        mcap = fund.get("market_cap")
        lines.append(f"  Market cap:     ${mcap/1e9:.1f}B" if mcap else "  Market cap:     N/A")
        lines.append(f"  Profit margin:  {fund.get('profit_margin','N/A')}")
        lines.append(f"  52w High:       ${fund.get('52w_high','N/A')}  |  52w Low: ${fund.get('52w_low','N/A')}")
        rating = (fund.get("analyst_rating") or "N/A").replace("_"," ").title()
        lines.append(f"  Analyst rating: {rating}  |  Target: ${fund.get('analyst_target_price','N/A')}")

    if not stw.get("error"):
        lines.append(f"\n## SOCIAL SENTIMENT (StockTwits)")
        lines.append(f"  Sentiment:      {stw.get('sentiment_label','N/A')}  ({stw.get('bullish_pct','N/A')}% bullish)")
        lines.append(f"  Posts sampled:  {stw.get('total_messages_sampled',0)}")

    if not reddit.get("error"):
        lines.append(f"\n## REDDIT BUZZ")
        lines.append(f"  Mentions:       {reddit.get('total_mentions_across_subreddits',0)} posts across WSB / r/stocks / r/investing")

    if not insider.get("error"):
        filings = insider.get("recent_form4_filings", [])
        lines.append(f"\n## INSIDER ACTIVITY (SEC Form 4)")
        lines.append(f"  Recent filings: {len(filings)} in the last period")
        if filings:
            lines.append(f"  Latest filing:  {filings[0].get('filed_date','N/A')}")

    # ── Persona-aware scoring ─────────────────────────────────────────────────
    persona_key = cfg.get("trading_persona", "swing")
    persona_cfg = PERSONA_CONFIGS.get(persona_key, PERSONA_CONFIGS["swing"])
    w           = persona_cfg["weights"]

    bull, bear = 0, 0

    # RSI — weighted by persona
    if w["rsi"] > 0:
        if rsi < 35:               bull += 2 * w["rsi"]
        elif rsi > 72:             bear += 2 * w["rsi"]
        elif 40 <= rsi <= 65:      bull += 1 * w["rsi"]

    # MACD — weighted by persona
    if w["macd"] > 0:
        if macd_sig == "bullish crossover":   bull += 2 * w["macd"]
        elif macd_sig == "bearish crossover": bear += 2 * w["macd"]

    # Momentum / EMA 20d
    if w["momentum"] > 0 and ema20 and cur:
        if cur > ema20:   bull += 1 * w["momentum"]
        else:             bear += 1 * w["momentum"]

    # EMA 200d — most important for long-term
    if w["ema200"] > 0 and ema200 and cur:
        if cur > ema200:   bull += 1 * w["ema200"]
        else:              bear += 2 * w["ema200"]

    # Bollinger Bands (counts as momentum signal)
    if w["momentum"] > 0:
        if bb_pct < 20:    bull += 1
        elif bb_pct > 80:  bear += 1

    # Volume spike (critical for day trading, ignored for long-term)
    if w["volume"] > 0 and vol_ratio >= 1.5 and macd_hist > 0:
        bull += 1 * w["volume"]
    elif w["volume"] > 0 and vol_ratio < 0.7:
        bear += 1 * w["volume"]

    # Fundamentals — only weighted for long-term persona
    if w["fundamentals"] > 0 and not fund.get("error"):
        pe = fund.get("pe_ratio")
        margin = fund.get("profit_margin", "")
        rating = (fund.get("analyst_rating") or "").lower()

        if rating in ("strong_buy", "buy"):      bull += 2 * w["fundamentals"]
        elif rating in ("sell", "underperform"): bear += 2 * w["fundamentals"]

        try:
            if pe and float(pe) < 25:            bull += 1 * w["fundamentals"]
            elif pe and float(pe) > 50:          bear += 1 * w["fundamentals"]
        except (TypeError, ValueError):
            pass

        try:
            mp = float(str(margin).replace("%",""))
            if mp > 15:                          bull += 1 * w["fundamentals"]
            elif mp < 0:                         bear += 2 * w["fundamentals"]
        except (TypeError, ValueError):
            pass

    # Sentiment
    if w["sentiment"] > 0:
        stw_bull = stw.get("bullish_pct") if not stw.get("error") else None
        if stw_bull and stw_bull > 65:           bull += 1 * w["sentiment"]
        elif stw_bull and stw_bull < 35:         bear += 1 * w["sentiment"]

    # Verdict thresholds scale with how many weighted signals there are
    net = bull - bear
    if   net >= 6:  verdict, conf = "STRONG BUY",    9
    elif net >= 3:  verdict, conf = "BUY",            7
    elif net >= 1:  verdict, conf = "WAIT",           5
    elif net >= -2: verdict, conf = "WAIT",           4
    elif net >= -5: verdict, conf = "AVOID",          3
    else:           verdict, conf = "STRONG AVOID",   2

    lines.append(f"\n{'='*60}")
    lines.append(f"  PERSONA:        {persona_cfg['emoji']} {persona_cfg['label']}")
    lines.append(f"  VERDICT:        {verdict}")
    lines.append(f"  CONFIDENCE:     {conf}/10  (bull signals: {bull}  |  bear signals: {bear})")
    lines.append(f"{'='*60}")

    if cur and verdict in ("STRONG BUY", "BUY"):
        tp = round(cur * (1 + cfg["take_profit_pct"] / 100), 2)
        sl = round(cur * (1 - cfg["stop_loss_pct"]   / 100), 2)
        lines.append(f"\n  Entry price:    ${cur:.2f}")
        lines.append(f"  Take-profit:    ${tp:.2f}  (+{cfg['take_profit_pct']}%)  ← exit here")
        lines.append(f"  Stop-loss:      ${sl:.2f}  (-{cfg['stop_loss_pct']}%)   ← exit immediately if hit")
        lines.append(f"  Max hold:       {persona_cfg['hold_label']}")
        lines.append(f"  Target range:   {persona_cfg['target_label']}")

    # Persona-specific risks
    risks = []
    if persona_key == "day":
        if vol_ratio < 1.5:
            risks.append(f"Low volume ({vol_ratio:.1f}x avg) — day trades need volume to move")
        if rsi > 70:
            risks.append(f"RSI overbought ({rsi}) — intraday mean-reversion risk")
        risks.append("Day trading risk: must exit before close or hold overnight unintentionally")
    elif persona_key == "swing":
        if rsi > 65:
            risks.append(f"RSI elevated ({rsi}) — possible near-term pullback")
        if ema200 and cur and cur < ema200:
            risks.append("Trading below 200-day EMA — long-term downtrend")
        if vol_ratio < 0.7:
            risks.append("Low volume — breakout may lack conviction")
    else:  # longterm
        if not fund.get("error"):
            pe = fund.get("pe_ratio")
            try:
                if pe and float(pe) > 40:
                    risks.append(f"High P/E ({pe}) — valuation stretched; growth must justify it")
            except (TypeError, ValueError):
                pass
        if ema200 and cur and cur < ema200:
            risks.append("Below 200-day EMA — wait for trend reversal before long-term entry")
        risks.append("Long-term risk: macro shifts (rates, recession) can override fundamentals")

    if not risks:
        risks.append("Standard market risk — no setup is ever risk-free")

    lines.append(f"\n## KEY RISKS")
    for r in risks[:3]:
        lines.append(f"  • {r}")

    # Persona-specific plain English summary
    lines.append(f"\n## PLAIN ENGLISH SUMMARY  ({persona_cfg['label']})")
    if verdict in ("STRONG BUY", "BUY"):
        if persona_key == "day":
            lines.append(f"  {ticker} has a volume spike and momentum signal — good intraday setup.")
            if cur:
                lines.append(f"  Buy near ${cur:.2f}, target ${round(cur*(1+cfg['take_profit_pct']/100),2):.2f}.")
                lines.append(f"  Set a hard stop at ${round(cur*(1-cfg['stop_loss_pct']/100),2):.2f} and EXIT BEFORE MARKET CLOSE.")
        elif persona_key == "swing":
            lines.append(f"  {ticker} is showing bullish technicals — a solid 2–5 day swing setup.")
            if cur:
                lines.append(f"  Buy near ${cur:.2f}, target ${round(cur*(1+cfg['take_profit_pct']/100),2):.2f}, "
                              f"stop at ${round(cur*(1-cfg['stop_loss_pct']/100),2):.2f}.")
        else:  # longterm
            lines.append(f"  {ticker} has strong fundamentals and is in a long-term uptrend.")
            if cur:
                lines.append(f"  Build a position near ${cur:.2f}. Target: ${round(cur*(1+cfg['take_profit_pct']/100),2):.2f} "
                              f"(+{cfg['take_profit_pct']}%). Stop: ${round(cur*(1-cfg['stop_loss_pct']/100),2):.2f}.")
                lines.append(f"  Hold for {persona_cfg['hold_label']} — ignore short-term volatility.")
    elif verdict == "WAIT":
        if persona_key == "day":
            lines.append(f"  No clean intraday setup right now. Volume and momentum aren't aligned.")
            lines.append(f"  Watch for a volume spike (>1.5x) with price breaking above resistance.")
        elif persona_key == "swing":
            lines.append(f"  {ticker} has mixed signals — not a clear swing entry yet.")
            lines.append(f"  Watch for RSI to dip below 65 and MACD to show a bullish crossover.")
        else:
            lines.append(f"  Fundamentals are OK but the technical picture isn't confirmed yet.")
            lines.append(f"  Wait for price to reclaim the 200-day EMA before adding a long-term position.")
    else:
        if persona_key == "day":
            lines.append(f"  {ticker} has no momentum or volume edge today. Skip — protect your capital.")
        elif persona_key == "swing":
            lines.append(f"  {ticker} has bearish signals. Not a swing trade candidate right now.")
        else:
            lines.append(f"  {ticker} has weak fundamentals or is in a downtrend. Avoid for long-term holding.")
        lines.append(f"  Wait for a clearer setup.")

    lines.append(f"\n  DISCLAIMER: Educational purposes only. Always do your own research.")
    return "\n".join(lines)


def _get_historical_stock_data(ticker: str, as_of: date) -> dict:
    """
    Fetches price, technicals, and fundamentals for a ticker using only
    data available up to `as_of` date — simulating what MarketMind would
    have seen if it ran on that day.
    """
    end = as_of + timedelta(days=1)           # yfinance end is exclusive
    start = as_of - timedelta(days=400)       # enough history for 200-day indicators

    stock = yf.Ticker(ticker)
    hist = stock.history(start=start.isoformat(), end=end.isoformat())

    if len(hist) < 26:
        return {"error": f"Not enough historical data for {ticker} as of {as_of}"}

    close  = hist["Close"]
    volume = hist["Volume"]
    current = round(float(close.iloc[-1]), 2)

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    rsi   = round(float(100 - (100 / (1 + rs.iloc[-1]))), 1)
    rsi_signal = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"

    # SMAs
    ma50  = round(float(close.rolling(50).mean().iloc[-1]), 2)
    ma200 = round(float(close.rolling(200).mean().iloc[-1]), 2) if len(hist) >= 200 else None

    # EMAs
    ema20  = round(float(close.ewm(span=20,  adjust=False).mean().iloc[-1]), 2)
    ema200 = round(float(close.ewm(span=200, adjust=False).mean().iloc[-1]), 2) if len(hist) >= 200 else None

    # MACD
    ema12      = close.ewm(span=12, adjust=False).mean()
    ema26      = close.ewm(span=26, adjust=False).mean()
    macd_line  = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram  = macd_line - signal_line
    macd_val   = round(float(macd_line.iloc[-1]), 4)
    signal_val = round(float(signal_line.iloc[-1]), 4)
    hist_val   = round(float(histogram.iloc[-1]), 4)
    macd_signal = (
        "bullish crossover" if macd_val > signal_val and hist_val > 0
        else "bearish crossover" if macd_val < signal_val and hist_val < 0
        else "neutral"
    )

    # Bollinger Bands
    bb_sma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper  = round(float((bb_sma + 2 * bb_std).iloc[-1]), 2)
    bb_middle = round(float(bb_sma.iloc[-1]), 2)
    bb_lower  = round(float((bb_sma - 2 * bb_std).iloc[-1]), 2)
    bb_pct = round((current - bb_lower) / (bb_upper - bb_lower) * 100, 1) if bb_upper != bb_lower else 50.0
    bb_signal = (
        "near upper band (overbought)" if bb_pct > 80
        else "near lower band (oversold)" if bb_pct < 20
        else "within bands (neutral)"
    )

    # Stochastic
    low14  = hist["Low"].rolling(14).min()
    high14 = hist["High"].rolling(14).max()
    pct_k  = round(float(((close - low14) / (high14 - low14) * 100).iloc[-1]), 1)

    # Volume
    avg_vol = volume.rolling(20).mean().iloc[-1]
    vol_ratio = round(float(volume.iloc[-1] / avg_vol), 2) if avg_vol else None
    vol_signal = (
        "major spike (>2x avg)" if vol_ratio and vol_ratio > 2
        else "moderate spike (>1.5x avg)" if vol_ratio and vol_ratio > 1.5
        else "normal"
    )

    # 7-day return as of as_of date
    weekly_return = round(((current - float(close.iloc[0])) / float(close.iloc[0])) * 100, 2) if len(hist) >= 2 else None

    # Fundamentals — static, use current values as approximation
    fund = get_fundamentals(ticker)

    # Predictability score based on 30-day volatility + beta
    beta = fund.get("beta") if fund and not fund.get("error") else None
    predictability = _calc_predictability(close, beta)

    return {
        "current_price":   current,
        "weekly_return":   weekly_return,
        "predictability":  predictability,
        "technicals": {
            "rsi_14d": rsi, "rsi_signal": rsi_signal,
            "ma_50d": ma50, "ma_200d": ma200,
            "ema_20d": ema20, "ema_200d": ema200,
            "ema_20d_signal": "price above EMA20 (short-term bullish)" if current > ema20 else "price below EMA20 (short-term bearish)",
            "ema_200d_signal": ("price above EMA200 (long-term bullish)" if ema200 and current > ema200 else "price below EMA200 (long-term bearish)") if ema200 else "insufficient data",
            "macd": macd_val, "macd_signal_line": signal_val,
            "macd_histogram": hist_val, "macd_signal": macd_signal,
            "bb_upper": bb_upper, "bb_middle": bb_middle, "bb_lower": bb_lower,
            "bb_price_position_pct": bb_pct, "bb_signal": bb_signal,
            "stoch_k": pct_k,
            "volume_ratio": vol_ratio, "volume_signal": vol_signal,
        },
        "fundamentals": fund,
    }


def _calc_predictability(close_series, beta: float | None) -> dict:
    """
    Scores how predictable a stock is for short-term technical forecasting.

    Uses two signals:
      1. 30-day historical volatility — annualised standard deviation of daily returns.
         Low vol stocks follow technical patterns more reliably.
      2. Beta — how much the stock moves relative to the S&P 500.
         Beta > 1.5 means the stock amplifies market moves, making it harder to predict.

    Scoring:
      High   — vol < 20% AND beta < 1.2  → indicators tend to be reliable
      Medium — vol 20–35% OR beta 1.2–1.8 → indicators work but with more noise
      Low    — vol > 35% OR beta > 1.8    → sentiment/news dominates over technicals

    Returns a dict with: score, volatility_30d_pct, beta, reason
    """
    # 30-day annualised volatility
    daily_returns = close_series.pct_change().dropna()
    vol_30d = round(float(daily_returns.tail(30).std() * (252 ** 0.5) * 100), 1)

    b = beta if beta else 1.0  # default to market-neutral if unknown

    if vol_30d < 20 and b < 1.2:
        score  = "High"
        reason = f"Low volatility ({vol_30d}% ann.) and low beta ({b:.2f}) — technicals are reliable"
    elif vol_30d > 35 or b > 1.8:
        score  = "Low"
        reason = f"High volatility ({vol_30d}% ann.) or high beta ({b:.2f}) — sentiment/news dominates technicals"
    else:
        score  = "Medium"
        reason = f"Moderate volatility ({vol_30d}% ann.) and beta ({b:.2f}) — technicals work with some noise"

    return {
        "score":            score,
        "volatility_30d_pct": vol_30d,
        "beta":             round(b, 2),
        "reason":           reason,
    }


def _get_actual_price_on(ticker: str, target_date: date) -> float | None:
    """
    Returns the closing price of a ticker on or nearest to `target_date`.
    Used to look up what actually happened after a prediction.
    """
    try:
        start = target_date - timedelta(days=5)
        end   = target_date + timedelta(days=1)
        hist  = yf.Ticker(ticker).history(start=start.isoformat(), end=end.isoformat())
        if hist.empty:
            return None
        # Get the row closest to target_date
        hist.index = hist.index.date
        if target_date in hist.index:
            return round(float(hist.loc[target_date, "Close"]), 2)
        # Nearest trading day on or before target_date
        available = [d for d in hist.index if d <= target_date]
        if available:
            return round(float(hist.loc[max(available), "Close"]), 2)
        return None
    except Exception:
        return None


def run_backtest_snapshot() -> None:
    """
    Backtests MarketMind's 7-day price prediction accuracy.

    How it works:
    1. Goes back 7 trading days (the 'prediction date')
    2. Fetches only the data that was available on that date
    3. Asks Claude to predict where each stock would be 7 days later (= today/yesterday)
    4. Looks up what the price actually was on the target date
    5. Shows: Predicted vs Actual vs Error %

    This lets you judge how trustworthy the predictions are before
    putting real money behind them.
    """
    TOP10 = [
        ("AAPL",  "Apple"),
        ("MSFT",  "Microsoft"),
        ("NVDA",  "NVIDIA"),
        ("AMZN",  "Amazon"),
        ("GOOGL", "Alphabet (Google)"),
        ("META",  "Meta"),
        ("BRK-B", "Berkshire Hathaway"),
        ("LLY",   "Eli Lilly"),
        ("AVGO",  "Broadcom"),
        ("TSLA",  "Tesla"),
    ]

    # prediction_date: the day we pretend MarketMind ran (10 calendar days ago ≈ 7 trading days)
    # actual_date:     the day the prediction was supposed to land (yesterday's close)
    prediction_date = date.today() - timedelta(days=10)
    actual_date     = date.today() - timedelta(days=1)

    print(f"\n  Backtest: pretending MarketMind ran on {prediction_date.strftime('%B %d, %Y')}")
    print(f"  Predicted target date: {actual_date.strftime('%B %d, %Y')} (yesterday's close)")
    print(f"\nFetching historical data as of {prediction_date.strftime('%b %d')}...\n")

    rows = []
    for ticker, name in TOP10:
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            data = _get_historical_stock_data(ticker, prediction_date)
            if data.get("error"):
                raise ValueError(data["error"])

            actual_price = _get_actual_price_on(ticker, actual_date)

            rows.append({
                "ticker":        ticker,
                "name":          name,
                "price_on_pred_date": data["current_price"],
                "weekly_return": data["weekly_return"],
                "technicals":    data["technicals"],
                "fundamentals":  data["fundamentals"],
                "actual_price":  actual_price,
            })
            print("done")
        except Exception as e:
            rows.append({
                "ticker": ticker, "name": name,
                "price_on_pred_date": None, "weekly_return": None,
                "technicals": {}, "fundamentals": {},
                "actual_price": None,
            })
            print(f"error: {e}")

    # Ask Claude to predict based on historical data
    print(f"\nAsking MarketMind what it would have predicted on {prediction_date.strftime('%b %d')}...",
          end=" ", flush=True)
    predictions = _get_ai_predictions(rows)
    print("done\n")

    # ── Print backtest table ──────────────────────────────────────────────────
    W = 125
    print("=" * W)
    print(f"  MARKETMIND BACKTEST  |  Predicted on: {prediction_date.strftime('%B %d')}  →  Actual on: {actual_date.strftime('%B %d, %Y')}")
    print(f"  Did MarketMind's prediction match reality?")
    print("=" * W)
    pred_date_str   = prediction_date.strftime("%b %d")
    actual_date_str = actual_date.strftime("%b %d")
    print(f"  {'Ticker':<7} {'Company':<22} {f'Price ({pred_date_str})':>14} "
          f"{'Predicted':>11} {f'Actual ({actual_date_str})':>12} "
          f"{'Predict?':>8}  {'Verdict'}")
    print("-" * W)

    errors = []
    for r in rows:
        base   = f"${r['price_on_pred_date']:,.2f}" if r['price_on_pred_date'] else "N/A"
        actual = f"${r['actual_price']:,.2f}"        if r['actual_price']       else "N/A"

        pred = predictions.get(r["ticker"], {})
        pp   = pred.get("predicted_price")

        p      = r.get("predictability", {})
        pscore = p.get("score", "N/A")
        beta   = p.get("beta")
        vol    = p.get("volatility_30d_pct")
        p_str  = f"{pscore}" + (f"(β{beta})" if beta else "")

        if pp and r["actual_price"] and r["price_on_pred_date"]:
            error_pct  = round((pp - r["actual_price"]) / r["actual_price"] * 100, 2)
            pred_str   = f"${pp:,.2f}"
            errors.append(abs(error_pct))

            pred_direction    = "UP"   if pp                > r["price_on_pred_date"] else "DOWN"
            actual_direction  = "UP"   if r["actual_price"] > r["price_on_pred_date"] else "DOWN"
            direction_correct = pred_direction == actual_direction
            verdict = f"{'✓ CORRECT' if direction_correct else '✗ WRONG'} direction | {abs(error_pct):.1f}% off"
        else:
            pred_str  = "N/A"
            error_pct = None
            verdict   = "N/A"

        print(f"  {r['ticker']:<7} {r['name']:<22} {base:>14} {pred_str:>11} {actual:>12} "
              f"{p_str:>10}  {verdict}")

    print("-" * W)

    if errors:
        avg_error = round(sum(errors) / len(errors), 2)
        correct_directions = sum(
            1 for r in rows
            if predictions.get(r["ticker"], {}).get("predicted_price")
            and r["actual_price"] and r["price_on_pred_date"]
            and (
                (predictions[r["ticker"]]["predicted_price"] > r["price_on_pred_date"]) ==
                (r["actual_price"] > r["price_on_pred_date"])
            )
        )
        total_valid = sum(1 for r in rows if predictions.get(r["ticker"], {}).get("predicted_price") and r["actual_price"])
        print(f"\n  BACKTEST SUMMARY")
        print(f"  Average prediction error : {avg_error}%")
        print(f"  Direction accuracy       : {correct_directions}/{total_valid} stocks called correctly")
        print(f"  (Direction = did MarketMind correctly predict UP or DOWN?)")

    print()
    print(f"  Price ({prediction_date.strftime('%b %d')}) = what the stock was trading at when MarketMind 'ran'")
    print(f"  Predicted              = what MarketMind predicted the price would be by {actual_date.strftime('%b %d')}")
    print(f"  Actual ({actual_date.strftime('%b %d')})   = what the price actually was on {actual_date.strftime('%b %d')}")
    print(f"  Error                  = how far off the predicted price was from reality")
    print(f"  Verdict                = did MarketMind at least get the direction right (UP/DOWN)?")
    print()
    print("  NOTE: A small error % and correct direction both matter.")
    print("  Even professional analysts are wrong ~40% of the time.")
    print()

    # Save to file
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, f"backtest_{prediction_date.isoformat()}_to_{actual_date.isoformat()}.txt")
    with open(out_path, "w") as f:
        f.write(f"MARKETMIND BACKTEST\n")
        f.write(f"Predicted on: {prediction_date}  |  Target date: {actual_date}\n\n")
        f.write(f"{'Ticker':<7} {'Company':<22} {'Base Price':>12} {'Predicted':>11} {'Actual':>10} {'Verdict'}\n")
        f.write("-" * 90 + "\n")
        for r in rows:
            base   = f"${r['price_on_pred_date']:,.2f}" if r['price_on_pred_date'] else "N/A"
            actual = f"${r['actual_price']:,.2f}"        if r['actual_price']       else "N/A"
            pred   = predictions.get(r["ticker"], {})
            pp     = pred.get("predicted_price")
            pred_str = f"${pp:,.2f}" if pp else "N/A"
            f.write(f"{r['ticker']:<7} {r['name']:<22} {base:>12} {pred_str:>11} {actual:>10}\n")
        if errors:
            f.write(f"\nAverage error: {avg_error}%  |  Direction accuracy: {correct_directions}/{total_valid}\n")
    print(f"  Backtest saved to: {out_path}\n")


def _get_ai_predictions(stocks_data: list) -> dict:
    """
    Rule-based 7-day price predictions using technical indicators.
    Replaces the Claude API call — completely free.
    """
    predictions = {}
    for s in stocks_data:
        ticker = s.get("ticker")
        tech   = s.get("technicals", {})
        cur    = s.get("current_price") or s.get("price_on_pred_date")

        if not tech or not cur or tech.get("error"):
            continue

        rsi        = tech.get("rsi_14d", 50)
        macd_sig   = tech.get("macd_signal", "neutral")
        macd_hist  = tech.get("macd_histogram", 0)
        bb_pct     = tech.get("bb_price_position_pct", 50)
        ema20      = tech.get("ema_20d")
        ema200     = tech.get("ema_200d")
        vol_ratio  = tech.get("volume_ratio") or 1.0

        bull, bear = 0, 0
        if rsi < 35:                                             bull += 2
        elif rsi > 72:                                           bear += 2
        elif 40 <= rsi <= 65:                                    bull += 1
        if macd_sig == "bullish crossover":                      bull += 2
        elif macd_sig == "bearish crossover":                    bear += 2
        if ema20 and cur > ema20:                                bull += 1
        elif ema20:                                              bear += 1
        if ema200 and cur > ema200:                              bull += 1
        elif ema200:                                             bear += 1
        if bb_pct < 20:                                          bull += 1
        elif bb_pct > 80:                                        bear += 1
        if vol_ratio >= 1.5 and macd_hist > 0:                   bull += 1

        net = bull - bear
        if   net >= 4:  direction, move, confidence = "UP",       0.04,  "High"
        elif net >= 2:  direction, move, confidence = "UP",       0.02,  "Medium"
        elif net <= -4: direction, move, confidence = "DOWN",    -0.04,  "High"
        elif net <= -2: direction, move, confidence = "DOWN",    -0.02,  "Medium"
        else:           direction, move, confidence = "SIDEWAYS", 0.005, "Low"

        reason_parts = []
        if rsi < 35:                   reason_parts.append(f"oversold RSI={rsi}")
        elif rsi > 70:                 reason_parts.append(f"overbought RSI={rsi}")
        if macd_sig != "neutral":      reason_parts.append(f"MACD {macd_sig}")
        if ema20:
            reason_parts.append("above EMA20" if cur > ema20 else "below EMA20")

        predictions[ticker] = {
            "predicted_price": round(cur * (1 + move), 2),
            "predicted_today": round(cur * (1 + move * 0.3), 2),
            "direction":       direction,
            "arrow":           "▲" if direction == "UP" else "▼" if direction == "DOWN" else "→",
            "confidence":      confidence,
            "reason":          "; ".join(reason_parts[:3]) or "mixed signals",
        }
    return predictions


def _get_ai_predictions_UNUSED(stocks_data: list) -> dict:
    """Original Claude-based implementation — kept for reference only."""
    client = None  # anthropic.Anthropic() — requires paid API key

    # Build a compact data summary for each stock to send to Claude
    summary_lines = []
    for s in stocks_data:
        t = s["ticker"]
        lines = [
            f"\n{t} ({s['name']}):",
            f"  Current price: ${s['current_price']}",
            f"  7-day return: {s['weekly_return']}%",
            f"  Analyst target: ${s.get('target_price', 'N/A')}  |  Rating: {s.get('rating', 'N/A')}",
        ]
        tech = s.get("technicals", {})
        if tech and not tech.get("error"):
            lines += [
                f"  RSI (14d): {tech.get('rsi_14d')}  ({tech.get('rsi_signal')})",
                f"  MACD signal: {tech.get('macd_signal')}  |  Histogram: {tech.get('macd_histogram')}",
                f"  EMA 20d: ${tech.get('ema_20d')}  |  EMA 200d: ${tech.get('ema_200d')}",
                f"  EMA 20d signal: {tech.get('ema_20d_signal')}",
                f"  EMA 200d signal: {tech.get('ema_200d_signal')}",
                f"  Bollinger signal: {tech.get('bb_signal')}  |  BB position: {tech.get('bb_price_position_pct')}%",
                f"  Stochastic %K: {tech.get('stoch_k')}  |  Signal: {tech.get('stoch_signal')}",
                f"  Volume signal: {tech.get('volume_signal')}  |  Volume ratio: {tech.get('volume_ratio')}x",
                f"  50d SMA: ${tech.get('ma_50d')}  |  200d SMA: ${tech.get('ma_200d')}",
            ]
        fund = s.get("fundamentals", {})
        if fund and not fund.get("error"):
            lines += [
                f"  P/E ratio: {fund.get('pe_ratio')}  |  Forward P/E: {fund.get('forward_pe')}",
                f"  EPS: {fund.get('earnings_per_share')}  |  Revenue growth: {fund.get('revenue_growth_yoy')}",
                f"  Profit margin: {fund.get('profit_margin')}  |  ROE: {fund.get('return_on_equity')}",
                f"  52w High: ${fund.get('52w_high')}  |  52w Low: ${fund.get('52w_low')}",
            ]
        p = s.get("predictability", {})
        if p:
            lines.append(
                f"  Predictability: {p.get('score', 'N/A')} | "
                f"30d volatility: {p.get('volatility_30d_pct')}% | Beta: {p.get('beta')} | "
                f"{p.get('reason', '')}"
            )
        summary_lines.append("\n".join(lines))

    prompt = (
        f"Today is {date.today().strftime('%B %d, %Y')}.\n\n"
        "You are a quantitative analyst. Below is real market data for 10 stocks. "
        "Based strictly on the technical indicators, price momentum, and fundamental signals provided, "
        "predict TWO prices for each stock:\n"
        "1. predicted_today: where you expect the stock to close TODAY based on current technicals\n"
        "2. predicted_price: where you expect the stock to be 7 days from today\n\n"
        "Use this reasoning framework:\n"
        "- RSI > 70 = overbought (price likely to pull back)\n"
        "- RSI < 30 = oversold (price likely to bounce)\n"
        "- MACD bullish crossover + positive histogram = upward momentum\n"
        "- Price above EMA20 and EMA200 = strong uptrend\n"
        "- Bollinger near upper band = potential resistance\n"
        "- Bollinger near lower band = potential support / bounce\n"
        "- High volume spike on up day = strong buying interest\n"
        "- Stochastic > 80 = overbought, < 20 = oversold\n\n"
        "STOCK DATA:\n"
        + "\n".join(summary_lines) +
        "\n\nFor EACH stock, respond with ONLY a JSON object in this exact format "
        "(no prose, no markdown, just valid JSON):\n"
        "{\n"
        '  "TICKER": {"predicted_today": 121.00, "predicted_price": 123.45, "confidence": "High|Medium|Low", '
        '"direction": "UP|DOWN|SIDEWAYS", "reason": "one sentence max"},\n'
        "  ...\n"
        "}\n\n"
        "Be realistic — small moves (1-5%) are more common than large ones. "
        "Base confidence on how many indicators agree with each other AND on the predictability score:\n"
        "- High predictability = technicals are reliable, you can be more confident\n"
        "- Medium predictability = widen your uncertainty range slightly\n"
        "- Low predictability = technicals may be overridden by news/sentiment; "
        "set confidence to Low regardless of indicator agreement, and keep predicted move conservative."
    )

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if Claude wrapped the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        predictions = json.loads(raw)
        # Attach direction symbol
        for ticker, pred in predictions.items():
            d = pred.get("direction", "SIDEWAYS")
            pred["arrow"] = "▲" if d == "UP" else "▼" if d == "DOWN" else "→"
        return predictions
    except json.JSONDecodeError:
        return {}


def run_top10_snapshot() -> None:
    """
    Fetches live data for the top 10 US stocks by market cap, then calls Claude
    to generate a 7-day price prediction based on technical indicators and fundamentals.

    Columns in the output table:
      Current $   — today's market price
      Target $    — Wall Street analyst consensus price target
      Upside      — gap between current price and analyst target
      7-Day       — price change over the past 7 trading days
      AI Predict  — MarketMind's 7-day price prediction (powered by Claude)
      Confidence  — High / Medium / Low (how many indicators agree)
      Rating      — analyst consensus rating
    """
    # Top 10 US stocks by market cap (approximate as of 2026)
    TOP10 = [
        ("AAPL",  "Apple"),
        ("MSFT",  "Microsoft"),
        ("NVDA",  "NVIDIA"),
        ("AMZN",  "Amazon"),
        ("GOOGL", "Alphabet (Google)"),
        ("META",  "Meta"),
        ("BRK-B", "Berkshire Hathaway"),
        ("LLY",   "Eli Lilly"),
        ("AVGO",  "Broadcom"),
        ("TSLA",  "Tesla"),
    ]

    print("\nFetching data for top 10 US stocks...\n")

    rows = []
    for ticker, name in TOP10:
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            weekly   = get_weekly_return(ticker)
            fund     = get_fundamentals(ticker)
            tech     = get_technical_indicators(ticker)

            current_price = weekly.get("current_price")
            weekly_return = weekly.get("weekly_return_pct")
            target_price  = fund.get("analyst_target_price")
            rating        = (fund.get("analyst_rating") or "N/A").replace("_", " ").title()

            if current_price and target_price:
                upside_pct = round((target_price - current_price) / current_price * 100, 1)
                upside_str = f"{'▲' if upside_pct >= 0 else '▼'} {abs(upside_pct):.1f}%"
            else:
                upside_pct = None
                upside_str = "N/A"

            # Predictability score — fetch 60 days of close prices for volatility calc
            try:
                hist_close = yf.Ticker(ticker).history(period="60d")["Close"]
                beta_val   = fund.get("beta")
                predictability = _calc_predictability(hist_close, beta_val)
            except Exception:
                predictability = {"score": "N/A", "volatility_30d_pct": None, "beta": None, "reason": ""}

            rows.append({
                "ticker":          ticker,
                "name":            name,
                "current_price":   current_price,
                "target_price":    target_price,
                "upside":          upside_str,
                "upside_pct":      upside_pct,
                "weekly_return":   weekly_return,
                "rating":          rating,
                "technicals":      tech,
                "fundamentals":    fund,
                "predictability":  predictability,
            })
            print("done")
        except Exception as e:
            rows.append({
                "ticker": ticker, "name": name,
                "current_price": None, "target_price": None,
                "upside": "N/A", "upside_pct": None,
                "weekly_return": None, "rating": "N/A",
                "technicals": {}, "fundamentals": {},
                "predictability": {"score": "N/A", "volatility_30d_pct": None, "beta": None, "reason": ""},
            })
            print(f"error: {e}")

    # ── Ask Claude for 7-day price predictions ────────────────────────────────
    print("\nAsking MarketMind to predict 7-day prices...", end=" ", flush=True)
    predictions = _get_ai_predictions(rows)
    print("done\n")

    # ── Print table ───────────────────────────────────────────────────────────
    next_week = (date.today() + timedelta(days=7)).strftime("%b %d")
    W = 155
    print("=" * W)
    print(f"  TOP 10 US STOCKS — PRICE SNAPSHOT + AI PREDICTION  |  {date.today().strftime('%B %d, %Y')}")
    print(f"  Actual $ = TODAY'S live price  |  Pred Today $ = MarketMind's call for today's close  |  AI ({next_week}) = 7-day forecast")
    print("=" * W)
    print(f"  {'Ticker':<7} {'Company':<22} {'Actual $':>10} {'Pred Today $':>13} {'vs Actual':>10} "
          f"{'Target $':>10} {'Upside':>9} {'7-Day':>7} {f'AI ({next_week})':>19} {'Conf.':>6}  {'Predict?':>10}  {'Rating'}")
    print("-" * W)

    for r in rows:
        cur  = f"${r['current_price']:,.2f}" if r['current_price'] else "N/A"
        tgt  = f"${r['target_price']:,.2f}"  if r['target_price']  else "N/A"
        wkly = f"{r['weekly_return']:+.2f}%" if r['weekly_return'] is not None else "N/A"

        pred = predictions.get(r["ticker"], {})
        if pred and pred.get("predicted_price"):
            pp      = pred["predicted_price"]
            arrow   = pred.get("arrow", "→")
            pct_chg = ((pp - r["current_price"]) / r["current_price"] * 100) if r["current_price"] else 0
            ai_str  = f"{arrow}${pp:,.2f} ({pct_chg:+.1f}%)"
            conf    = pred.get("confidence", "N/A")
        else:
            ai_str = "N/A"
            conf   = "N/A"

        # vs Actual: compare predicted_today to current live price
        if pred and pred.get("predicted_today") and r["current_price"]:
            pt       = pred["predicted_today"]
            diff     = pt - r["current_price"]
            diff_pct = (diff / r["current_price"]) * 100
            pt_str   = f"${pt:,.2f}"
            sign     = "▲" if diff >= 0 else "▼"
            vs_str   = f"{sign}${abs(diff):,.2f} ({diff_pct:+.1f}%)"
        else:
            pt_str = "N/A"
            vs_str = "N/A"

        p     = r.get("predictability", {})
        pscore = p.get("score", "N/A")
        beta  = p.get("beta")
        predictability_str = f"{pscore}" + (f" (β{beta})" if beta else "")

        print(f"  {r['ticker']:<7} {r['name']:<22} {cur:>10} {pt_str:>13} {vs_str:>10} "
              f"{tgt:>10} {r['upside']:>9} {wkly:>7} {ai_str:>19} {conf:>6}  {predictability_str:>10}  {r['rating']}")

    print("-" * W)
    print()
    print(f"  Actual $      = live market price RIGHT NOW ({date.today().strftime('%B %d, %Y')})")
    print(f"  Pred Today $  = MarketMind's prediction for today's closing price")
    print(f"  vs Actual     = difference between today's prediction and the live price (▲ = AI predicts higher close)")
    print(f"  AI ({next_week})  = MarketMind's predicted price in 7 days")
    print(f"  Target $      = where Wall Street analysts think it will be in 12 months")
    print(f"  Upside      = gap between today's price and analyst 12-month target (▲ = room to grow)")
    print(f"  7-Day       = how much the price actually moved over the past 7 trading days")
    print(f"  Conf.       = how many technical indicators agree with the prediction (High/Medium/Low)")
    print(f"  Predict?    = how reliable is this prediction? High/Medium/Low based on 30-day volatility + beta")
    print(f"              High  = stable stock, technicals are reliable, trust the prediction more")
    print(f"              Medium = moderate volatility, prediction is a good guide but watch for surprises")
    print(f"              Low   = high volatility/beta (like Tesla), news/sentiment can override technicals")
    print(f"  Rating      = Wall Street analyst consensus (Strong Buy → Buy → Hold → Sell)")
    print()
    print("  DISCLAIMER: AI predictions are for educational purposes only.")
    print("  Past patterns do not guarantee future results. Always do your own research.")
    print()

    # ── Save to file ──────────────────────────────────────────────────────────
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, f"top10_snapshot_{date.today().isoformat()}.txt")
    with open(out_path, "w") as f:
        f.write(f"TOP 10 US STOCKS — SNAPSHOT + AI PREDICTION\n")
        f.write(f"{date.today().strftime('%B %d, %Y')}\n")
        f.write(f"Pred Today = MarketMind's predicted close for today  |  AI {next_week} = 7-day forecast\n\n")
        f.write(f"{'Ticker':<7} {'Company':<22} {'Actual $':>10} {'Pred Today':>13} {'vs Actual':>12} "
                f"{'Target':>10} {'Upside':>9} {'7-Day':>7} {f'AI {next_week}':>19} {'Conf':>6}  Rating\n")
        f.write("-" * 125 + "\n")
        for r in rows:
            cur  = f"${r['current_price']:,.2f}" if r['current_price'] else "N/A"
            tgt  = f"${r['target_price']:,.2f}"  if r['target_price']  else "N/A"
            wkly = f"{r['weekly_return']:+.2f}%" if r['weekly_return'] is not None else "N/A"
            pred = predictions.get(r["ticker"], {})
            if pred and pred.get("predicted_price"):
                pp = pred["predicted_price"]
                pct_chg = ((pp - r["current_price"]) / r["current_price"] * 100) if r["current_price"] else 0
                ai_str = f"{pred.get('arrow','→')}${pp:,.2f} ({pct_chg:+.1f}%)"
                conf = pred.get("confidence", "N/A")
            else:
                ai_str = "N/A"
                conf = "N/A"
            if pred and pred.get("predicted_today") and r["current_price"]:
                pt = pred["predicted_today"]
                diff = pt - r["current_price"]
                diff_pct = (diff / r["current_price"]) * 100
                pt_str = f"${pt:,.2f}"
                sign = "▲" if diff >= 0 else "▼"
                vs_str = f"{sign}${abs(diff):,.2f} ({diff_pct:+.1f}%)"
            else:
                pt_str = "N/A"
                vs_str = "N/A"
            f.write(f"{r['ticker']:<7} {r['name']:<22} {cur:>10} {pt_str:>13} {vs_str:>12} "
                    f"{tgt:>10} {r['upside']:>9} {wkly:>7} {ai_str:>19} {conf:>6}  {r['rating']}\n")
            if pred.get("reason"):
                f.write(f"         AI reason: {pred['reason']}\n")
    print(f"  Snapshot saved to: {out_path}\n")


def _save_forecast(ticker: str, report: str) -> str:
    """Save a stock forecast to a dated file."""
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    filename = os.path.join(
        reports_dir,
        f"forecast_{ticker.upper()}_{date.today().isoformat()}.md"
    )
    with open(filename, "w") as f:
        f.write(report)
    return filename


# ── Entry point ───────────────────────────────────────────────────────────────

# ── Broker connection (multi-broker) ─────────────────────────────────────────

ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL  = "https://api.alpaca.markets"


# ── Shared helpers ────────────────────────────────────────────────────────────

def _test_alpaca_connection(api_key: str, api_secret: str, base_url: str) -> dict:
    """
    Public helper — tests Alpaca credentials by hitting /v2/account.
    Returns the account dict on success, or {"error": "..."} on failure.
    Called by both the CLI flow and the Streamlit app.
    """
    url = f"{base_url}/v2/account"
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret,
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()}"}
    except Exception as e:
        return {"error": str(e)}


def _fetch_alpaca_positions(api_key: str, api_secret: str, base_url: str) -> list:
    """
    Public helper — fetches open positions from Alpaca /v2/positions.
    Returns a list of position dicts, or [] on error.
    Called by both the CLI flow and the Streamlit app.
    """
    url = f"{base_url}/v2/positions"
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret,
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception:
        return []


def _save_env_keys(new_values: dict, section_comment: str) -> None:
    """
    Upserts key=value pairs into the .env file.
    Existing matching keys are overwritten in-place; new keys are appended
    under section_comment. All writes stay local — never committed to git.
    """
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    lines = open(env_path).readlines() if os.path.exists(env_path) else []

    found, updated = set(), []
    for line in lines:
        k = line.split("=")[0].strip()
        if k in new_values:
            updated.append(f"{k}={new_values[k]}\n")
            found.add(k)
        else:
            updated.append(line)

    if not found:
        updated.append(f"\n{section_comment}\n")
    for k, v in new_values.items():
        if k not in found:
            updated.append(f"{k}={v}\n")

    with open(env_path, "w") as f:
        f.writelines(updated)


def _import_positions(state: dict, positions: list, broker_tag: str) -> dict:
    """
    Shared helper: offers to import a list of positions into portfolio state.
    positions — list of dicts with keys: symbol, qty, avg_buy_price,
                market_value, unrealized_pl, cash (account-level cash).
    """
    if not positions:
        print("  No open positions found in your account.")
        return state

    print(f"\n  Found {len(positions)} open position(s):")
    for p in positions:
        sym     = p.get("symbol", "?")
        qty     = p.get("qty", "?")
        cost    = float(p.get("avg_buy_price", 0))
        mval    = float(p.get("market_value",  0))
        pl      = float(p.get("unrealized_pl", 0))
        pl_sign = "+" if pl >= 0 else ""
        print(f"    {sym:<6}  {qty} shares @ ${cost:.2f}"
              f"  | Market value: ${mval:,.2f}  | P&L: {pl_sign}${pl:.2f}")

    ans = input("\n  Import these positions into MarketMind? (y/n): ").strip().lower()
    if ans == "y":
        holdings = state.setdefault("holdings", {})
        for p in positions:
            sym  = p.get("symbol", "")
            qty  = float(p.get("qty", 0))
            cost = float(p.get("avg_buy_price", 0))
            if sym and qty > 0:
                holdings[sym] = {
                    "shares":        qty,
                    "avg_buy_price": cost,
                    "source":        broker_tag,
                    "import_date":   date.today().isoformat(),
                }
        cash   = float(p.get("cash", state.get("cash_available", 0)))
        equity = sum(float(p.get("market_value", 0)) for p in positions) + cash
        state["cash_available"]        = cash
        state["total_portfolio_value"] = equity
        print(f"  ✓ {len(positions)} position(s) imported.")
    return state


# ── Alpaca ────────────────────────────────────────────────────────────────────

def _connect_alpaca(state: dict) -> dict:
    """
    Connects to Alpaca (paper or live) using API key + secret.
    Official API — no extra dependencies beyond stdlib urllib.
    """
    print("\n  ── Alpaca ──────────────────────────────────────────────────")
    print("  Official API  |  Free paper + live trading  |  No credit card")
    print("  Credentials: API Key ID + Secret Key (not your login password)")
    print("  Security: Keys are read-only by default. MarketMind cannot")
    print("            place orders without your explicit action.\n")

    print("  Account type:")
    print("  1. Paper  — Simulated money, zero real risk. Best for getting started.")
    print("  2. Live   — Real money. Only if your Alpaca account is funded.\n")
    while True:
        c = input("  Enter 1 or 2: ").strip()
        if c == "1":
            base_url, acct_type = ALPACA_PAPER_URL, "paper"
            break
        elif c == "2":
            base_url, acct_type = ALPACA_LIVE_URL, "live"
            print("  ⚠  Live selected — no orders placed automatically.")
            break
        print("  Please enter 1 or 2.")

    print("\n  Get your free keys (≈1 min):")
    print("  1. Sign up at https://alpaca.markets")
    if acct_type == "paper":
        print("  2. Dashboard → Paper Trading → API Keys → Generate New Key")
    else:
        print("  2. Dashboard → Live Trading → API Keys → Generate New Key")
    print("  3. Copy the Key ID and Secret (secret shown only once)\n")

    api_key    = input("  API Key ID : ").strip()
    api_secret = input("  Secret Key : ").strip()
    if not api_key or not api_secret:
        print("  No credentials entered — skipped.")
        return state

    print("\n  Testing connection...", end=" ", flush=True)
    url = f"{base_url}/v2/account"
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret,
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            account = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print("FAILED")
        print(f"  ✗ HTTP {e.code}: {e.read().decode()}")
        return state
    except Exception as e:
        print("FAILED")
        print(f"  ✗ {e}")
        return state

    equity = float(account.get("equity", 0))
    cash   = float(account.get("cash",   0))
    print("OK")
    print(f"  ✓ Connected  |  Status: {account.get('status','?').upper()}"
          f"  |  Equity: ${equity:,.2f}  |  Cash: ${cash:,.2f}")

    _save_env_keys(
        {"ALPACA_API_KEY": api_key, "ALPACA_API_SECRET": api_secret,
         "ALPACA_BASE_URL": base_url},
        "# Alpaca brokerage — https://alpaca.markets"
    )
    print("  ✓ Credentials saved to .env (local only).")

    state.update({"broker": "alpaca", "broker_url": base_url,
                  "broker_acct_type": acct_type})

    # Fetch and offer to import positions
    try:
        pos_req = urllib.request.Request(
            f"{base_url}/v2/positions",
            headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret,
                     "Accept": "application/json"},
        )
        with urllib.request.urlopen(pos_req, timeout=10) as r:
            raw = json.loads(r.read().decode())
        positions = [
            {"symbol": p["symbol"], "qty": p["qty"],
             "avg_buy_price": p["avg_entry_price"],
             "market_value": p["market_value"],
             "unrealized_pl": p["unrealized_pl"],
             "cash": cash}
            for p in raw
        ]
    except Exception:
        positions = []

    return _import_positions(state, positions, "alpaca_import")


# ── Robinhood ─────────────────────────────────────────────────────────────────

def _connect_robinhood(state: dict) -> dict:
    """
    Connects to Robinhood via the robin_stocks library (unofficial API).
    Uses username + password + MFA code — no API key exists for Robinhood.
    """
    print("\n  ── Robinhood ────────────────────────────────────────────────")
    print("  ⚠  IMPORTANT: Robinhood has NO official API for third-party")
    print("     apps. This connection uses 'robin_stocks', an unofficial")
    print("     library that reverse-engineers Robinhood's mobile app.")
    print()
    print("     What this means for you:")
    print("     • It uses your Robinhood username & password (not an API key)")
    print("     • It may break if Robinhood updates their app without notice")
    print("     • Robinhood's Terms of Service technically prohibit this")
    print("     • MarketMind reads your positions only — it cannot place trades")
    print()
    print("  If you prefer a safer, officially supported connection,")
    print("  go back and choose Alpaca (free, takes 1 minute to set up).\n")

    ans = input("  Understood — continue with Robinhood? (y/n): ").strip().lower()
    if ans != "y":
        print("  Robinhood connection cancelled.")
        return state

    # Check dependency
    try:
        import robin_stocks.robinhood as rh
    except ImportError:
        print("\n  robin_stocks is not installed. Run:")
        print("    pip install robin_stocks")
        print("  Then re-run MarketMind to connect your Robinhood account.")
        return state

    print("\n  Enter your Robinhood login credentials.")
    print("  These are stored only in memory during this session —")
    print("  they are NOT written to .env or any file.\n")

    import getpass
    username = input("  Robinhood email: ").strip()
    password = getpass.getpass("  Password (hidden): ")

    print("\n  Logging in...", end=" ", flush=True)
    try:
        login = rh.login(username, password, store_session=False)
        if not login:
            print("FAILED")
            print("  ✗ Login failed. Check your credentials and try again.")
            return state
    except Exception as e:
        print("FAILED")
        print(f"  ✗ {e}")
        return state

    print("OK")

    # Fetch account info
    try:
        profile  = rh.load_account_profile()
        cash     = float(profile.get("cash", 0) or 0)
        equity   = float(profile.get("equity", 0) or 0)
        print(f"  ✓ Connected to Robinhood")
        print(f"    Equity: ${equity:,.2f}  |  Cash: ${cash:,.2f}")
    except Exception:
        cash, equity = 0.0, 0.0
        print("  ✓ Logged in (could not fetch account summary)")

    state.update({"broker": "robinhood"})

    # Note: Robinhood credentials are intentionally NOT saved to .env
    # (passwords should never be stored in plain text files)
    print("\n  Note: Robinhood credentials are not saved to disk for security.")
    print("  You will be asked to log in again on the next session.")

    # Fetch and offer to import positions
    try:
        raw = rh.get_open_stock_positions()
        positions = []
        for p in raw:
            instrument_url = p.get("instrument", "")
            # Resolve ticker symbol from instrument URL
            try:
                inst_req = urllib.request.Request(
                    instrument_url,
                    headers={"Accept": "application/json"}
                )
                with urllib.request.urlopen(inst_req, timeout=5) as r:
                    inst = json.loads(r.read().decode())
                symbol = inst.get("symbol", "")
            except Exception:
                symbol = ""

            qty  = float(p.get("quantity", 0) or 0)
            cost = float(p.get("average_buy_price", 0) or 0)

            if symbol and qty > 0:
                # Get current price for market value estimate
                try:
                    quote = rh.get_latest_price(symbol)
                    price = float(quote[0]) if quote else cost
                except Exception:
                    price = cost
                mval = round(qty * price, 2)
                pl   = round(mval - qty * cost, 2)
                positions.append({
                    "symbol": symbol, "qty": qty,
                    "avg_buy_price": cost,
                    "market_value": mval,
                    "unrealized_pl": pl,
                    "cash": cash,
                })
    except Exception:
        positions = []

    try:
        rh.logout()
    except Exception:
        pass

    return _import_positions(state, positions, "robinhood_import")


# ── Interactive Brokers ───────────────────────────────────────────────────────

def _connect_ibkr(state: dict) -> dict:
    """
    Connects to Interactive Brokers via ib_insync (official API).
    Requires TWS or IB Gateway to be running on the local machine.
    """
    print("\n  ── Interactive Brokers ─────────────────────────────────────")
    print("  Official API  |  Extremely powerful  |  Free for account holders")
    print()
    print("  Requirements before connecting:")
    print("  1. An Interactive Brokers account (min. $0 for IBKR Lite)")
    print("  2. Trader Workstation (TWS) or IB Gateway installed and running")
    print("     Download: https://www.interactivebrokers.com/en/trading/tws.php")
    print("  3. In TWS: Edit → Global Configuration → API → Settings")
    print("     ✓ Enable ActiveX and Socket Clients")
    print("     ✓ Socket port: 7497 (paper) or 7496 (live)")
    print("     ✓ Allow connections from localhost only\n")

    ans = input("  Is TWS or IB Gateway running right now? (y/n): ").strip().lower()
    if ans != "y":
        print("  Please start TWS first, then re-run this setup.")
        return state

    # Check dependency
    try:
        from ib_insync import IB, util
        util.startLoop()
    except ImportError:
        print("\n  ib_insync is not installed. Run:")
        print("    pip install ib_insync")
        print("  Then re-run MarketMind to connect your IBKR account.")
        return state

    print("\n  Account type:")
    print("  1. Paper  — TWS paper trading (port 7497)")
    print("  2. Live   — TWS live trading   (port 7496)\n")
    while True:
        c = input("  Enter 1 or 2: ").strip()
        if c == "1":
            port, acct_type = 7497, "paper"
            break
        elif c == "2":
            port, acct_type = 7496, "live"
            print("  ⚠  Live selected — no orders placed automatically.")
            break
        print("  Please enter 1 or 2.")

    print(f"\n  Connecting to TWS on port {port}...", end=" ", flush=True)
    ib = IB()
    try:
        ib.connect("127.0.0.1", port, clientId=10)
    except Exception as e:
        print("FAILED")
        print(f"  ✗ Could not connect: {e}")
        print("  Ensure TWS is running and API connections are enabled.")
        return state

    print("OK")

    # Account summary
    try:
        summary = {s.tag: s.value for s in ib.accountSummary()}
        equity  = float(summary.get("NetLiquidation", 0))
        cash    = float(summary.get("TotalCashValue",  0))
        acct_id = summary.get("AccountId", "unknown")
        print(f"  ✓ Connected to IBKR  |  Account: {acct_id}"
              f"  |  Equity: ${equity:,.2f}  |  Cash: ${cash:,.2f}")
    except Exception:
        equity, cash, acct_id = 0.0, 0.0, "unknown"
        print("  ✓ Connected (could not fetch account summary)")

    _save_env_keys(
        {"IBKR_PORT": str(port), "IBKR_ACCT_TYPE": acct_type},
        "# Interactive Brokers — requires TWS or IB Gateway running locally"
    )
    print("  ✓ Port and account type saved to .env.")

    state.update({"broker": "ibkr", "broker_port": port,
                  "broker_acct_type": acct_type})

    # Fetch and offer to import positions
    try:
        from ib_insync import Stock
        raw = ib.positions()
        positions = []
        for p in raw:
            symbol = p.contract.symbol
            qty    = float(p.position)
            cost   = float(p.avgCost) / qty if qty else 0
            ticker = ib.reqMktData(p.contract, "", True, False)
            ib.sleep(1)
            price  = ticker.last or ticker.close or cost
            mval   = round(qty * price, 2)
            pl     = round(mval - qty * cost, 2)
            positions.append({
                "symbol": symbol, "qty": qty,
                "avg_buy_price": round(cost, 4),
                "market_value": mval,
                "unrealized_pl": pl,
                "cash": cash,
            })
    except Exception:
        positions = []

    ib.disconnect()
    return _import_positions(state, positions, "ibkr_import")


# ── Broker menu ───────────────────────────────────────────────────────────────

def _ask_broker_connection(state: dict) -> dict:
    """
    Shows a broker selection menu and routes to the chosen broker's
    connection flow. All broker flows write credentials to .env (except
    Robinhood, which never stores a password). Returns updated state.
    """
    print("\n┌─ CONNECT YOUR BROKERAGE ACCOUNT (optional) ─────────────────┐")
    print("│ Link MarketMind to your real brokerage so your Daily Trade  │")
    print("│ Scan reflects what you actually own.                        │")
    print("│                                                             │")
    print("│ Security: Credentials are stored only in your local .env   │")
    print("│ file and never uploaded or shared. MarketMind reads your   │")
    print("│ positions — it cannot place or cancel orders for you.      │")
    print("└─────────────────────────────────────────────────────────────┘")

    ans = input("\nWould you like to connect a brokerage account? (y/n): ").strip().lower()
    if ans != "y":
        print("  Skipped. You can connect a broker later by re-running setup.")
        return state

    print("\n  Choose your broker:\n")
    print("  1. Alpaca          — Official API. Free paper + live trading.")
    print("                       Best choice if you don't have a broker yet.")
    print("                       Needs: API Key ID + Secret Key\n")
    print("  2. Robinhood       — Unofficial (no public API exists).")
    print("                       Works today but may break with app updates.")
    print("                       Needs: Robinhood username + password\n")
    print("  3. Interactive     — Official API. Very powerful, all asset classes.")
    print("     Brokers (IBKR)    Needs: TWS desktop app running + ib_insync\n")
    print("  4. Skip for now    — Set up a connection later.\n")

    handlers = {"1": _connect_alpaca, "2": _connect_robinhood, "3": _connect_ibkr}
    while True:
        c = input("  Enter 1, 2, 3, or 4: ").strip()
        if c in handlers:
            return handlers[c](state)
        if c == "4":
            print("  Skipped. Run setup again any time to connect a broker.")
            return state
        print("  Please enter 1, 2, 3, or 4.")


def _ask_profile_name() -> str:
    """
    Asks the user to give their profile a name.
    Returns the name as a string.
    """
    print("\n┌─ PROFILE NAME ──────────────────────────────────────────────┐")
    print("│ Give your trading profile a name so you can identify it     │")
    print("│ later. This is just a label — it doesn't affect any trades. │")
    print("│ Examples: 'My First Portfolio', 'Divya - Growth Fund'       │")
    print("└─────────────────────────────────────────────────────────────┘")
    while True:
        name = input("Profile name: ").strip()
        if name:
            print(f"  ✓ Profile name set to: \"{name}\"")
            return name
        print("  Please enter a name for your profile.")


def _ask_risk_appetite() -> str:
    """
    Interactively asks the user for their risk appetite and returns
    one of: "conservative", "moderate", "aggressive".
    Also previews the take-profit and stop-loss percentages each option sets.
    """
    print("\n┌─ RISK APPETITE ─────────────────────────────────────────────┐")
    print("│ Your risk appetite tells MarketMind how aggressively to     │")
    print("│ trade. It sets two key limits automatically:                │")
    print("│                                                             │")
    print("│  • Take-profit %  — When a trade gains this %, MarketMind  │")
    print("│                     tells you to sell and lock in profits.  │")
    print("│  • Stop-loss %    — When a trade loses this %, MarketMind  │")
    print("│                     tells you to exit to protect capital.   │")
    print("└─────────────────────────────────────────────────────────────┘\n")
    print("  1. Conservative  — Safety first. Stable, dividend-paying stocks.")
    print("                     Take-profit: +3%  |  Stop-loss: -1.5%")
    print("                     Best for: beginners or anyone who cannot afford big losses.\n")
    print("  2. Moderate      — Balanced. Mix of stable and growth stocks.")
    print("                     Take-profit: +5%  |  Stop-loss: -2%")
    print("                     Best for: most investors wanting steady, moderate growth.\n")
    print("  3. Aggressive    — Growth-focused. High-volatility momentum stocks.")
    print("                     Take-profit: +8%  |  Stop-loss: -3%")
    print("                     Best for: experienced traders comfortable with bigger swings.\n")

    options = {"1": "conservative", "2": "moderate", "3": "aggressive",
               "conservative": "conservative", "moderate": "moderate", "aggressive": "aggressive"}

    while True:
        choice = input("Enter your choice (1, 2, or 3): ").strip().lower()
        if choice in options:
            selected = options[choice]
            print(f"\n  ✓ Risk appetite set to: {selected.upper()}")
            return selected
        print("  Please enter 1, 2, or 3.")


def _ask_initial_investment() -> float:
    """
    Asks the user how much money they want to start with.
    Returns the amount as a float.
    """
    print("\n┌─ STARTING BUDGET ───────────────────────────────────────────┐")
    print("│ This is the total amount of money you want MarketMind to    │")
    print("│ manage. Think of it as your trading 'pot'.                  │")
    print("│                                                             │")
    print("│  • No new money is added for 6 months after setup.         │")
    print("│  • Profits from trades are reinvested automatically        │")
    print("│    (compounding), so your pot can grow over time.          │")
    print("│  • Only invest what you can afford to lose entirely.       │")
    print("│                                                             │")
    print("│ Examples: 1000 (starter), 5000 (typical), 10000 (advanced) │")
    print("└─────────────────────────────────────────────────────────────┘")
    while True:
        raw = input("Enter amount (USD): ").strip().replace(",", "").replace("$", "")
        try:
            amount = float(raw)
            if amount < 100:
                print("  Minimum budget is $100. Please enter a higher amount.")
                continue
            print(f"\n  ✓ Starting budget set to: ${amount:,.2f}")
            return amount
        except ValueError:
            print("  Please enter a valid number, e.g. 5000")


def _print_profile_summary(profile_name: str, capital: float, risk_level: str,
                            take_profit: float, stop_loss: float,
                            start_date: str, end_date: str) -> None:
    """
    Prints a clear summary of every profile field with a brief definition,
    so the user understands exactly what has been set up.
    """
    print("\n" + "=" * 62)
    print("  YOUR TRADING PROFILE — SUMMARY")
    print("=" * 62)
    print(f"  Profile Name      : {profile_name}")
    print(f"    └─ Your label for this account.\n")
    print(f"  Starting Budget   : ${capital:,.2f}")
    print(f"    └─ Total money MarketMind will trade with.\n")
    print(f"  Risk Appetite     : {risk_level.upper()}")
    print(f"    └─ How aggressively to trade (Conservative / Moderate / Aggressive).\n")
    print(f"  Take-Profit       : +{take_profit}%")
    print(f"    └─ Exit a winning trade when it gains this % to lock in profits.\n")
    print(f"  Stop-Loss         : -{stop_loss}%")
    print(f"    └─ Exit a losing trade when it drops this % to protect your capital.\n")
    print(f"  Max Open Trades   : 5")
    print(f"    └─ Never hold more than 5 trades at the same time (keeps you focused).\n")
    print(f"  Max Per Trade     : 20% of total capital")
    print(f"    └─ No single trade can use more than 20% of your budget.\n")
    print(f"  Risk Per Trade    : 2% of total capital")
    print(f"    └─ The most you can lose on any one trade (used to size positions).\n")
    print(f"  Trading Period    : {start_date}  →  {end_date}")
    print(f"    └─ Your 6-month window. Results are tracked over this period.\n")
    print("=" * 62)


if __name__ == "__main__":
    print("=" * 60)
    print("  Welcome to MarketMind!")
    print("  Your Personal AI Trading Agent")
    print("=" * 60)

    # First run: ask questions and initialise trading account
    if not os.path.exists(PORTFOLIO_STATE_FILE):
        print("\nLooks like this is your first time running MarketMind.")
        print("Let's set up your trading profile. We'll walk through each")
        print("setting one at a time and explain what it means.\n")

        # Ask for profile name
        profile_name = _ask_profile_name()

        # Ask for starting capital (budget)
        initial_investment = _ask_initial_investment()
        TRADING_CONFIG["starting_capital"]    = initial_investment
        TRADING_CONFIG["initial_investment"]  = initial_investment

        # Ask for risk appetite
        risk_level = _ask_risk_appetite()
        TRADING_CONFIG["risk_level"] = risk_level

        # Set take-profit / stop-loss based on risk level
        take_profit_map = {"conservative": 3.0, "moderate": 5.0, "aggressive": 8.0}
        stop_loss_map   = {"conservative": 1.5, "moderate": 2.0, "aggressive": 3.0}
        TRADING_CONFIG["take_profit_pct"] = take_profit_map[risk_level]
        TRADING_CONFIG["stop_loss_pct"]   = stop_loss_map[risk_level]

        print(f"\nCreating your trading account...")
        state = initialize_portfolio()
        state["profile_name"] = profile_name
        _save_state(state)

        # Print a clear summary of every setting with its definition
        _print_profile_summary(
            profile_name   = profile_name,
            capital        = initial_investment,
            risk_level     = risk_level,
            take_profit    = take_profit_map[risk_level],
            stop_loss      = stop_loss_map[risk_level],
            start_date     = state["start_date"],
            end_date       = state["end_date"],
        )

        # Offer optional brokerage connection
        state = _ask_broker_connection(state)
        _save_state(state)

        print(f"\n  ✓ Profile saved to portfolio_state.json\n")

    else:
        # Returning user — ask if they want to update their risk appetite
        state = _load_state()
        current_risk = state.get("risk_level", TRADING_CONFIG["risk_level"])
        print(f"\nWelcome back! Current risk level: {current_risk.upper()}")
        change = input("Would you like to change your risk appetite? (y/n): ").strip().lower()
        if change == "y":
            risk_level = _ask_risk_appetite()
            take_profit = {"conservative": 3.0, "moderate": 5.0, "aggressive": 8.0}
            stop_loss   = {"conservative": 1.5, "moderate": 2.0, "aggressive": 3.0}
            state["risk_level"]       = risk_level
            state["take_profit_pct"]  = take_profit[risk_level]
            state["stop_loss_pct"]    = stop_loss[risk_level]
            _save_state(state)
            TRADING_CONFIG["risk_level"]      = risk_level
            TRADING_CONFIG["take_profit_pct"] = take_profit[risk_level]
            TRADING_CONFIG["stop_loss_pct"]   = stop_loss[risk_level]
            print(f"Risk updated to {risk_level.upper()} — "
                  f"Take-profit: +{take_profit[risk_level]}%  |  Stop-loss: -{stop_loss[risk_level]}%")
        print()

    # ── Main menu ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  MARKETMIND — AI TRADING AGENT")
    print(f"  {datetime.today().strftime('%A, %B %d %Y')}")
    print("=" * 60)
    print("\nWhat would you like to do?\n")
    print("  1. Daily Trade Scan")
    print("     Scan the market for setups, check open trades, get ENTER/EXIT signals\n")
    print("  2. Stock Forecast")
    print("     Heard news about a stock? Get a trade decision with entry, take-profit,")
    print("     and stop-loss levels\n")
    print("  3. Top 10 US Stocks Snapshot")
    print("     Live prices, analyst targets, and MarketMind's 7-day prediction\n")
    print("  4. Backtest MarketMind Predictions")
    print("     See how accurate last week's predictions were vs what actually happened\n")

    while True:
        choice = input("Enter 1, 2, 3, or 4: ").strip()
        if choice in ("1", "2", "3", "4"):
            break
        print("Please enter 1, 2, 3, or 4.")

    print()

    if choice == "1":
        print("=" * 60)
        print("  DAILY TRADE SCAN")
        print("=" * 60 + "\n")
        report = run_agent()
        print("\n" + report)
        saved_path = _save_daily_report(report)
        print(f"\n\nReport saved to: {saved_path}")

    elif choice == "2":
        print("=" * 60)
        print("  STOCK FORECAST")
        print("=" * 60)
        print("\nEnter the ticker symbol of the stock you want to analyse.")
        print("Examples: AAPL (Apple), TSLA (Tesla), NVDA (NVIDIA)\n")
        ticker = input("Ticker symbol: ").strip().upper()
        if not ticker:
            print("No ticker entered. Exiting.")
        else:
            print()
            report = run_stock_forecast(ticker)
            print("\n" + report)
            saved_path = _save_forecast(ticker, report)
            print(f"\n\nForecast saved to: {saved_path}")

    elif choice == "3":
        print("=" * 60)
        print("  TOP 10 US STOCKS SNAPSHOT")
        print("=" * 60)
        run_top10_snapshot()

    elif choice == "4":
        print("=" * 60)
        print("  BACKTEST MARKETMIND PREDICTIONS")
        print("=" * 60)
        run_backtest_snapshot()

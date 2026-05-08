"""
MarketMind — Pre-Bell Scanner
==============================
A morning tool that runs BEFORE the market opens (ideally 8:00–9:25 AM ET).

For each ticker it shows:
  - Pre-market price movement + gap risk (from pre_market.py)
  - Key technical indicators: RSI, MACD, Bollinger, Stochastic, Volume (from marketmind.py)
  - A combined quick signal: BUY / SELL / WATCH / SKIP

Signal logic (stock market perspective):
  SKIP  — Gap risk is HIGH (entry price has moved >5% from plan; risk/reward collapsed)
  BUY   — Gap is flat/moderate-up + RSI not overbought + MACD bullish momentum
  SELL  — Gap is flat/moderate-down + RSI not oversold + MACD bearish momentum
  WATCH — Mixed or unclear signals; wait for first 15 min of trading before acting

Usage:
    python prebell.py                          # scans TRADING_UNIVERSE from marketmind.py
    python prebell.py AAPL NVDA TSLA MSFT      # scan specific tickers
    python prebell.py --universe               # explicitly use full TRADING_UNIVERSE
"""

from __future__ import annotations

import sys
from datetime import datetime
from zoneinfo import ZoneInfo

# Reuse functions from existing marketmind.py and pre_market.py
from marketmind import get_technical_indicators, get_weekly_return, TRADING_UNIVERSE
from pre_market import fetch_gap_info, GapRisk, GapType, GapInfo

_ET = ZoneInfo("America/New_York")


# ── Signal engine ─────────────────────────────────────────────────────────────

def compute_signal(gap: GapInfo, tech: dict) -> tuple[str, str]:
    """
    Combine pre-market gap data with technical indicators to produce a signal.

    Returns:
        (signal, reason)  where signal ∈ {"BUY", "SELL", "WATCH", "SKIP", "ERROR"}
    """
    # ── SKIP: gap risk too high — thesis invalidated before open ─────────────
    if gap.gap_risk == GapRisk.HIGH:
        return "SKIP", (
            f"Gap {gap.gap_pct:+.2f}% exceeds threshold — "
            f"entry price has moved too far {'up' if gap.gap_pct > 0 else 'down'} from plan"
        )

    if "error" in tech:
        return "ERROR", f"Technical data unavailable: {tech['error']}"

    rsi       = tech.get("rsi_14d", 50)
    macd_sig  = tech.get("macd_signal", "neutral")
    bb_pct    = tech.get("bb_price_position_pct", 50)
    stoch_k   = tech.get("stoch_k", 50)
    vol_ratio = tech.get("volume_ratio") or 1.0

    # Score bullish vs bearish signals (simple point system)
    bull_score = 0
    bear_score = 0
    reasons: list[str] = []

    # RSI
    if rsi < 35:
        bull_score += 2
        reasons.append(f"RSI {rsi} oversold (bounce candidate)")
    elif rsi > 65:
        bear_score += 2
        reasons.append(f"RSI {rsi} overbought (pullback risk)")
    else:
        reasons.append(f"RSI {rsi} neutral")

    # MACD
    if "bullish" in macd_sig:
        bull_score += 2
        reasons.append("MACD bullish crossover")
    elif "bearish" in macd_sig:
        bear_score += 2
        reasons.append("MACD bearish crossover")

    # Bollinger band position
    if bb_pct < 20:
        bull_score += 1
        reasons.append(f"BB {bb_pct:.0f}% (near lower band — oversold)")
    elif bb_pct > 80:
        bear_score += 1
        reasons.append(f"BB {bb_pct:.0f}% (near upper band — overbought)")

    # Stochastic
    if stoch_k < 20:
        bull_score += 1
        reasons.append(f"Stoch %K {stoch_k} oversold")
    elif stoch_k > 80:
        bear_score += 1
        reasons.append(f"Stoch %K {stoch_k} overbought")

    # Pre-market gap direction adds conviction
    if gap.gap_type == GapType.UP:
        bull_score += 1
        reasons.append(f"PM gap up {gap.gap_pct:+.2f}%")
    elif gap.gap_type == GapType.DOWN:
        bear_score += 1
        reasons.append(f"PM gap down {gap.gap_pct:+.2f}%")

    # High pre-market volume = stronger conviction in direction
    if gap.high_conviction:
        if gap.gap_type == GapType.UP:
            bull_score += 1
        elif gap.gap_type == GapType.DOWN:
            bear_score += 1
        reasons.append(f"High PM volume ({gap.pm_vol_ratio:.1%} of daily avg)")

    # Volume spike on last close day
    if vol_ratio > 1.5:
        reasons.append(f"Volume spike {vol_ratio:.1f}x avg at close")

    # ── Moderate gap adjustment: reduce signal confidence ─────────────────────
    if gap.gap_risk == GapRisk.MODERATE:
        reasons.append(f"⚠ Moderate gap — consider half size, limit near ${gap.prev_close:.2f}")

    # ── Decision ──────────────────────────────────────────────────────────────
    reason_str = " | ".join(reasons)

    if bull_score >= 4 and bull_score > bear_score + 1:
        return "BUY", reason_str
    elif bear_score >= 4 and bear_score > bull_score + 1:
        return "SELL", reason_str
    else:
        return "WATCH", reason_str


# ── Display ───────────────────────────────────────────────────────────────────

_SIGNAL_ICONS = {
    "BUY":   "🟢 BUY  ",
    "SELL":  "🔴 SELL ",
    "WATCH": "🟡 WATCH",
    "SKIP":  "⛔ SKIP ",
    "ERROR": "❌ ERROR",
}

_RISK_ICONS = {
    "low":      "🟢",
    "moderate": "🟡",
    "high":     "🔴",
}


def _print_ticker_block(
    ticker: str,
    gap: GapInfo,
    tech: dict,
    weekly: dict,
    signal: str,
    reason: str,
) -> None:
    """Print a clean per-ticker block to the terminal."""
    icon = _SIGNAL_ICONS.get(signal, "❓")
    risk_icon = _RISK_ICONS.get(gap.gap_risk.value, "")

    # Header
    price = tech.get("current_price") or gap.prev_close
    week_ret = weekly.get("weekly_return_pct")
    week_str = f"{week_ret:+.2f}%" if week_ret is not None else "N/A"
    print(f"\n  ┌─ {ticker:<6}  ${price:.2f}  (7d: {week_str})")

    # Pre-market gap
    if gap.data_available:
        conv = " ⚡ HIGH CONVICTION" if gap.high_conviction else ""
        print(
            f"  │  PRE-MARKET  {gap.gap_pct:+.2f}%  "
            f"(${gap.prev_close:.2f} → ${gap.pm_last_price:.2f})  "
            f"risk={risk_icon}{gap.gap_risk.value}  "
            f"PM vol={gap.pm_volume:,}{conv}"
        )
    else:
        print(f"  │  PRE-MARKET  no data available")

    # Technical indicators (compact)
    if "error" not in tech:
        rsi       = tech.get("rsi_14d", "N/A")
        macd_s    = tech.get("macd_signal", "N/A")
        bb_pct    = tech.get("bb_price_position_pct", "N/A")
        stoch     = tech.get("stoch_k", "N/A")
        vol_ratio = tech.get("volume_ratio", "N/A")
        ma_s      = tech.get("ma_signal", "")
        print(
            f"  │  TECHNICALS  RSI {rsi}  |  MACD {macd_s}  |  "
            f"BB {bb_pct:.0f}%  |  Stoch %K {stoch}  |  Vol {vol_ratio}x"
        )
        print(f"  │              {ma_s}")
    else:
        print(f"  │  TECHNICALS  error: {tech['error']}")

    # Signal
    print(f"  └─ {icon}  {reason}")


def run_scan(tickers: list[str]) -> None:
    """Fetch all data and print the pre-bell scan report."""
    now = datetime.now(tz=_ET)

    print("\n" + "═" * 72)
    print("  MarketMind — Pre-Bell Scanner")
    print("═" * 72)
    print(f"  {now.strftime('%A, %B %d, %Y  %H:%M:%S ET')}  |  {len(tickers)} tickers")

    # Market session context
    if now.hour < 4:
        session = "⏳ Pre-pre-market (before 4 AM ET) — limited data"
    elif now.hour < 9 or (now.hour == 9 and now.minute < 30):
        session = "🌅 Pre-market session (4:00–9:30 AM ET) — ideal window"
    elif now.hour < 16:
        session = "📈 Regular session open — pre-market data may be stale"
    else:
        session = "🌙 After hours — reviewing EOD data"
    print(f"  {session}")
    print("─" * 72)

    # ── Fetch all data ────────────────────────────────────────────────────────
    print(f"\n  Fetching pre-market data...", end="", flush=True)
    gaps = fetch_gap_info(tickers)
    print(" ✓")

    print(f"  Fetching technical indicators...", end="", flush=True)
    techs   = {t: get_technical_indicators(t) for t in tickers}
    weeklys = {t: get_weekly_return(t) for t in tickers}
    print(" ✓\n")

    # ── Signals + output ──────────────────────────────────────────────────────
    signals: list[tuple[str, str, str]] = []  # (ticker, signal, reason)

    for ticker in tickers:
        gap    = gaps[ticker]
        tech   = techs[ticker]
        weekly = weeklys[ticker]
        signal, reason = compute_signal(gap, tech)
        signals.append((ticker, signal, reason))
        _print_ticker_block(ticker, gap, tech, weekly, signal, reason)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  SUMMARY")
    print("─" * 72)

    for label, category in [
        ("🟢 BUY  ", "BUY"),
        ("🔴 SELL ", "SELL"),
        ("🟡 WATCH", "WATCH"),
        ("⛔ SKIP ", "SKIP"),
        ("❌ ERROR", "ERROR"),
    ]:
        group = [t for t, s, _ in signals if s == category]
        if group:
            print(f"  {label}  {', '.join(group)}")

    # Gap risk summary
    high_risk  = [t for t in tickers if gaps[t].gap_risk.value == "high"]
    moderate   = [t for t in tickers if gaps[t].gap_risk.value == "moderate"]
    conviction = [t for t in tickers if gaps[t].high_conviction]

    print()
    if high_risk:
        print(f"  🔴 High gap risk (signals suppressed): {', '.join(high_risk)}")
    if moderate:
        print(f"  🟡 Moderate gap (half-size recommended): {', '.join(moderate)}")
    if conviction:
        print(f"  ⚡ High PM conviction volume: {', '.join(conviction)}")

    print(f"\n  Scan complete — {now.strftime('%H:%M:%S ET')}")
    print("═" * 72 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or "--universe" in args:
        tickers = TRADING_UNIVERSE
    else:
        tickers = [t.upper() for t in args if not t.startswith("--")]

    run_scan(tickers)

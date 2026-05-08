"""
MarketMind — Pre-Market Gap Detection
======================================
Fetches pre-market (4:00–9:30 AM ET) price and volume data using yfinance,
computes gap metrics, and provides a gap filter to annotate or suppress
swing signals before the next session opens.

Key concepts (stock market perspective):
- Gap %       = (pre-market last price − previous close) / previous close × 100
- Gap type    = UP (bullish extension) | DOWN (bearish extension) | FLAT (< threshold)
- Gap risk    = LOW / MODERATE / HIGH — governs whether to act on an EOD signal
- PM vol ratio= pre-market volume / 20-day average daily volume
              A ratio > 0.10 (10% of daily vol before open) signals conviction.

Usage (drop-in with marketmind.py or volatile_predict.py):
    from pre_market import fetch_gap_info, apply_gap_filter, GapRisk

    gaps   = fetch_gap_info(["AAPL", "NVDA", "TSLA"])
    result = apply_gap_filter(my_signals, gaps)
    for r in result:
        if r.suppressed:
            print(f"SKIP {r.signal['ticker']}: {r.suppress_reason}")

Run standalone for a morning snapshot:
    python pre_market.py AAPL NVDA TSLA MSFT AMZN
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# ── Thresholds (tunable) ──────────────────────────────────────────────────────

# Gap % beyond which a signal is suppressed (entry thesis invalidated)
HIGH_GAP_SUPPRESS_PCT: float = 5.0

# Gap % that warrants a warning annotation but doesn't suppress
MODERATE_GAP_WARN_PCT: float = 2.0

# Pre-market volume as fraction of 20-day avg daily volume — "conviction" threshold
PM_VOL_CONVICTION_RATIO: float = 0.10


# ── Data types ────────────────────────────────────────────────────────────────

class GapType(str, Enum):
    UP   = "gap_up"
    DOWN = "gap_down"
    FLAT = "flat"


class GapRisk(str, Enum):
    LOW      = "low"       # < MODERATE_GAP_WARN_PCT  — proceed normally
    MODERATE = "moderate"  # MODERATE ≤ gap < HIGH    — annotate, reduce size
    HIGH     = "high"      # ≥ HIGH_GAP_SUPPRESS_PCT  — suppress signal


@dataclass
class GapInfo:
    """All pre-market gap metrics for a single ticker."""
    ticker: str

    # Core metrics
    prev_close: float             # Previous session's closing price
    pm_last_price: float          # Last pre-market trade price (or prev_close if no PM data)
    gap_pct: float                # Signed gap percentage (positive = gap up)
    gap_type: GapType
    gap_risk: GapRisk

    # Volume
    pm_volume: int                # Total pre-market share volume
    avg_daily_volume: float       # 20-day average daily volume
    pm_vol_ratio: float           # pm_volume / avg_daily_volume

    # Conviction flag — True when pre-market volume suggests strong directional interest
    high_conviction: bool

    # Metadata
    as_of: datetime               # Timestamp of pre-market snapshot (ET)
    data_available: bool = True   # False if yfinance returned no pre-market bars
    error: Optional[str] = None   # Non-None if fetch failed

    def summary(self) -> str:
        """Human-readable one-liner — used in terminal output and reports."""
        if not self.data_available:
            return f"{self.ticker}: no pre-market data"
        conv = "  ⚡ HIGH CONVICTION" if self.high_conviction else ""
        return (
            f"{self.ticker}: {self.gap_type.value} {self.gap_pct:+.2f}% "
            f"(prev close ${self.prev_close:.2f} → PM ${self.pm_last_price:.2f}) | "
            f"risk={self.gap_risk.value} | "
            f"PM vol={self.pm_volume:,} ({self.pm_vol_ratio:.1%} of avg daily){conv}"
        )


# ── Core classification ───────────────────────────────────────────────────────

def _classify_gap(gap_pct: float) -> tuple[GapType, GapRisk]:
    """Classify gap direction and risk level from signed gap %."""
    abs_gap = abs(gap_pct)

    if abs_gap >= HIGH_GAP_SUPPRESS_PCT:
        risk = GapRisk.HIGH
    elif abs_gap >= MODERATE_GAP_WARN_PCT:
        risk = GapRisk.MODERATE
    else:
        risk = GapRisk.LOW

    if gap_pct >= MODERATE_GAP_WARN_PCT:
        gap_type = GapType.UP
    elif gap_pct <= -MODERATE_GAP_WARN_PCT:
        gap_type = GapType.DOWN
    else:
        gap_type = GapType.FLAT

    return gap_type, risk


# ── Main fetch ────────────────────────────────────────────────────────────────

def fetch_gap_info(
    tickers: list[str],
    pm_lookback_days: int = 1,
    avg_vol_lookback: int = 20,
) -> dict[str, GapInfo]:
    """
    Fetch pre-market gap information for a list of tickers.

    Strategy:
      1. Download 1-minute bars with prepost=True (covers 4:00–9:30 AM ET today).
      2. Isolate only pre-market bars (before 9:30 AM ET).
      3. Use the last close from regular-hours history as prev_close reference.
      4. Compute gap %, classify, and return GapInfo per ticker.

    Args:
        tickers:           List of ticker symbols (e.g. TRADING_UNIVERSE from marketmind.py).
        pm_lookback_days:  Days of 1-min data to pull (1 = today only).
        avg_vol_lookback:  Days of daily data for average volume baseline.

    Returns:
        Dict mapping ticker → GapInfo (always present, even on fetch failure).
    """
    now_et = datetime.now(tz=_ET)
    results: dict[str, GapInfo] = {}

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)

            # ── 1. Get previous close + avg volume from daily history ──────────
            hist = t.history(period=f"{avg_vol_lookback + 5}d", auto_adjust=True)
            if hist.empty:
                raise ValueError("No daily history available")

            prev_close = float(hist["Close"].iloc[-1])
            avg_daily_vol = float(hist["Volume"].tail(avg_vol_lookback).mean())

            # ── 2. Fetch 1-minute bars including pre/post market ──────────────
            pm_data = t.history(
                period=f"{pm_lookback_days}d",
                interval="1m",
                prepost=True,
                auto_adjust=True,
            )

            if pm_data.empty:
                logger.warning(f"{ticker}: no intraday data returned")
                results[ticker] = GapInfo(
                    ticker=ticker,
                    prev_close=prev_close,
                    pm_last_price=prev_close,
                    gap_pct=0.0,
                    gap_type=GapType.FLAT,
                    gap_risk=GapRisk.LOW,
                    pm_volume=0,
                    avg_daily_volume=avg_daily_vol,
                    pm_vol_ratio=0.0,
                    high_conviction=False,
                    as_of=now_et,
                    data_available=False,
                )
                continue

            # ── 3. Isolate pre-market bars (4:00–9:29 AM ET) ─────────────────
            pm_data.index = pm_data.index.tz_convert(_ET)
            market_open = time(9, 30)
            today = now_et.date()

            premarket_mask = (
                (pm_data.index.date == today) &
                (pm_data.index.time < market_open)
            )
            premarket_bars = pm_data.loc[premarket_mask]

            if premarket_bars.empty:
                logger.info(f"{ticker}: no pre-market bars yet today")
                pm_last_price = prev_close
                pm_volume = 0
            else:
                pm_last_price = float(premarket_bars["Close"].iloc[-1])
                pm_volume = int(premarket_bars["Volume"].sum())

            # ── 4. Compute gap metrics ────────────────────────────────────────
            gap_pct = (pm_last_price - prev_close) / prev_close * 100
            gap_type, gap_risk = _classify_gap(gap_pct)
            pm_vol_ratio = pm_volume / avg_daily_vol if avg_daily_vol > 0 else 0.0
            high_conviction = pm_vol_ratio >= PM_VOL_CONVICTION_RATIO

            gap_info = GapInfo(
                ticker=ticker,
                prev_close=prev_close,
                pm_last_price=pm_last_price,
                gap_pct=gap_pct,
                gap_type=gap_type,
                gap_risk=gap_risk,
                pm_volume=pm_volume,
                avg_daily_volume=avg_daily_vol,
                pm_vol_ratio=pm_vol_ratio,
                high_conviction=high_conviction,
                as_of=now_et,
            )
            results[ticker] = gap_info

        except Exception as e:
            logger.error(f"{ticker}: gap fetch failed — {e}")
            results[ticker] = GapInfo(
                ticker=ticker,
                prev_close=0.0,
                pm_last_price=0.0,
                gap_pct=0.0,
                gap_type=GapType.FLAT,
                gap_risk=GapRisk.LOW,
                pm_volume=0,
                avg_daily_volume=0.0,
                pm_vol_ratio=0.0,
                high_conviction=False,
                as_of=datetime.now(tz=_ET),
                data_available=False,
                error=str(e),
            )

    return results


# ── Gap filter ────────────────────────────────────────────────────────────────

@dataclass
class SignalWithGap:
    """
    Wraps an existing signal dict with gap annotation.

    Added fields:
      gap_info       — the GapInfo object for this ticker
      suppressed     — True if gap risk is HIGH (signal should be skipped)
      suppress_reason— plain-English explanation when suppressed
      size_scalar    — position size multiplier (1.0 = full, 0.5 = half, 0.0 = none)
      notes          — list of informational strings (conviction, moderate warnings)
    """
    signal: dict
    gap_info: GapInfo
    suppressed: bool = False
    suppress_reason: str = ""
    size_scalar: float = 1.0
    notes: list[str] = field(default_factory=list)


def apply_gap_filter(
    signals: list[dict],
    gap_data: dict[str, GapInfo],
    ticker_key: str = "ticker",
    direction_key: str = "direction",
) -> list[SignalWithGap]:
    """
    Annotate and optionally suppress swing signals based on pre-market gaps.

    Suppression logic:
      - LONG  + HIGH gap UP   → suppress (chasing — entry has moved 5%+ above plan)
      - SHORT + HIGH gap DOWN → suppress (chasing — same logic in reverse)
      - LONG  + HIGH gap DOWN → suppress (adverse gap invalidates bullish thesis)
      - SHORT + HIGH gap UP   → suppress (gap-up risks short squeeze)
      - MODERATE gap (2–5%)  → annotate, halve position size
      - HIGH conviction vol  → informational note only

    Args:
        signals:       List of signal dicts (same format as marketmind.py trade suggestions).
        gap_data:      Output of fetch_gap_info().
        ticker_key:    Key in signal dict for the ticker symbol.
        direction_key: Key in signal dict for "long" / "short".

    Returns:
        List of SignalWithGap — one entry per input signal, same order.
    """
    output: list[SignalWithGap] = []

    for signal in signals:
        ticker = signal.get(ticker_key, "")
        direction = str(signal.get(direction_key, "long")).lower()
        gap_info = gap_data.get(ticker)

        if gap_info is None or not gap_info.data_available:
            output.append(SignalWithGap(
                signal=signal,
                gap_info=gap_info or GapInfo(
                    ticker=ticker, prev_close=0, pm_last_price=0, gap_pct=0,
                    gap_type=GapType.FLAT, gap_risk=GapRisk.LOW,
                    pm_volume=0, avg_daily_volume=0, pm_vol_ratio=0,
                    high_conviction=False, as_of=datetime.now(tz=_ET),
                    data_available=False,
                ),
                notes=["pre-market data unavailable — signal passed through unmodified"],
            ))
            continue

        notes: list[str] = []
        suppressed = False
        suppress_reason = ""
        size_scalar = 1.0

        if gap_info.gap_risk == GapRisk.HIGH:
            suppressed = True
            if gap_info.gap_type == GapType.UP and direction == "long":
                suppress_reason = (
                    f"Gap UP {gap_info.gap_pct:+.2f}% exceeds {HIGH_GAP_SUPPRESS_PCT}% — "
                    f"long entry has moved too far; skip to avoid chasing."
                )
            elif gap_info.gap_type == GapType.DOWN and direction == "short":
                suppress_reason = (
                    f"Gap DOWN {gap_info.gap_pct:+.2f}% exceeds -{HIGH_GAP_SUPPRESS_PCT}% — "
                    f"short entry has moved too far; skip to avoid chasing."
                )
            elif gap_info.gap_type == GapType.DOWN and direction == "long":
                suppress_reason = (
                    f"Gap DOWN {gap_info.gap_pct:+.2f}% — large adverse gap invalidates "
                    f"bullish thesis; wait for price discovery."
                )
            elif gap_info.gap_type == GapType.UP and direction == "short":
                suppress_reason = (
                    f"Gap UP {gap_info.gap_pct:+.2f}% — large gap up risks short squeeze; "
                    f"do not short into momentum."
                )

        elif gap_info.gap_risk == GapRisk.MODERATE:
            size_scalar = 0.5
            notes.append(
                f"Moderate gap {gap_info.gap_pct:+.2f}% — position size reduced to 50%. "
                f"Consider limit order near prev close ${gap_info.prev_close:.2f}."
            )

        if gap_info.high_conviction:
            notes.append(
                f"High pre-market conviction: {gap_info.pm_vol_ratio:.1%} of avg daily volume "
                f"traded before open ({gap_info.pm_volume:,} shares)."
            )

        output.append(SignalWithGap(
            signal=signal,
            gap_info=gap_info,
            suppressed=suppressed,
            suppress_reason=suppress_reason,
            size_scalar=0.0 if suppressed else size_scalar,
            notes=notes,
        ))

    suppressed_count = sum(1 for s in output if s.suppressed)
    logger.info(
        f"Gap filter: {suppressed_count}/{len(output)} signals suppressed, "
        f"{sum(1 for s in output if s.gap_info.gap_risk == GapRisk.MODERATE)} flagged moderate"
    )
    return output


# ── Convenience: standalone snapshot table ────────────────────────────────────

def premarket_snapshot(tickers: list[str]) -> pd.DataFrame:
    """
    Return a sorted DataFrame of pre-market conditions.
    Useful for a quick morning review before the bell.

    Columns: ticker, prev_close, pm_price, gap_pct, gap_type, gap_risk,
             pm_volume, pm_vol_ratio, high_conviction, as_of
    """
    gaps = fetch_gap_info(tickers)
    rows = []
    for g in gaps.values():
        rows.append({
            "ticker":          g.ticker,
            "prev_close":      g.prev_close,
            "pm_price":        g.pm_last_price,
            "gap_pct":         round(g.gap_pct, 3),
            "gap_type":        g.gap_type.value,
            "gap_risk":        g.gap_risk.value,
            "pm_volume":       g.pm_volume,
            "pm_vol_ratio":    round(g.pm_vol_ratio, 4),
            "high_conviction": g.high_conviction,
            "data_available":  g.data_available,
            "as_of":           g.as_of.strftime("%Y-%m-%d %H:%M ET"),
        })
    return pd.DataFrame(rows).sort_values("gap_pct", ascending=False).reset_index(drop=True)


# ── Standalone terminal runner ────────────────────────────────────────────────

def _print_snapshot(tickers: list[str]) -> None:
    """Pretty-print a pre-market snapshot table to stdout."""
    print("\n" + "=" * 75)
    print("  MarketMind — Pre-Market Gap Snapshot")
    print("=" * 75)

    gaps = fetch_gap_info(tickers)
    sorted_gaps = sorted(gaps.values(), key=lambda g: g.gap_pct, reverse=True)

    print(f"\n  {'Ticker':<6}  {'Prev Close':>10}  {'PM Price':>9}  {'Gap %':>7}  "
          f"{'Type':<10}  {'Risk':<8}  {'PM Vol':>10}  {'Vol Ratio':>9}  Conviction")
    print("  " + "-" * 85)

    for g in sorted_gaps:
        if not g.data_available:
            print(f"  {g.ticker:<6}  {'N/A':>10}  {'N/A':>9}  {'N/A':>7}  {'no data':<10}")
            continue
        conv = "YES ⚡" if g.high_conviction else "-"
        risk_icon = "🔴" if g.gap_risk == GapRisk.HIGH else "🟡" if g.gap_risk == GapRisk.MODERATE else "🟢"
        print(
            f"  {g.ticker:<6}  ${g.prev_close:>9.2f}  ${g.pm_last_price:>8.2f}  "
            f"{g.gap_pct:>+7.2f}%  {g.gap_type.value:<10}  {risk_icon} {g.gap_risk.value:<6}  "
            f"{g.pm_volume:>10,}  {g.pm_vol_ratio:>8.1%}   {conv}"
        )

    print("\n  " + "-" * 85)
    high_risk = [g.ticker for g in sorted_gaps if g.gap_risk == GapRisk.HIGH]
    moderate  = [g.ticker for g in sorted_gaps if g.gap_risk == GapRisk.MODERATE]
    if high_risk:
        print(f"  🔴 HIGH RISK (suppress signals): {', '.join(high_risk)}")
    if moderate:
        print(f"  🟡 MODERATE (halve size):        {', '.join(moderate)}")
    print(f"\n  Snapshot time: {datetime.now(tz=_ET).strftime('%Y-%m-%d %H:%M:%S ET')}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    # Usage: python pre_market.py AAPL NVDA TSLA MSFT AMZN
    # Falls back to MarketMind's default universe if no args given
    DEFAULT_TICKERS = [
        "TSLA", "NVDA", "AMD", "MSTR", "COIN",
        "MRNA", "META", "SNAP", "GME", "AMC",
    ]
    tickers_to_check = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS
    _print_snapshot(tickers_to_check)

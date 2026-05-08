"""
Unit tests for pre_market.py (older version — flat module, no package prefix)

Run from the 'older version' folder:
    pip install pytest yfinance pandas
    python -m pytest test_pre_market.py -v

Tests:
- Gap classification across all threshold combinations
- apply_gap_filter suppresses HIGH risk signals correctly
- apply_gap_filter halves size for MODERATE risk
- apply_gap_filter passes through when no gap data available
- HIGH conviction flag set correctly by volume ratio
- GapInfo.summary() produces readable strings
- premarket_snapshot() returns a proper DataFrame
- fetch_gap_info() handles yfinance errors gracefully
"""

from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from pre_market import (
    HIGH_GAP_SUPPRESS_PCT,
    MODERATE_GAP_WARN_PCT,
    PM_VOL_CONVICTION_RATIO,
    GapInfo,
    GapRisk,
    GapType,
    SignalWithGap,
    _classify_gap,
    apply_gap_filter,
    fetch_gap_info,
    premarket_snapshot,
)

_ET = ZoneInfo("America/New_York")
_NOW = datetime(2024, 6, 3, 8, 0, tzinfo=_ET)  # 8 AM ET — pre-market hour


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_gap_info(
    ticker: str = "AAPL",
    gap_pct: float = 0.0,
    gap_type: GapType = GapType.FLAT,
    gap_risk: GapRisk = GapRisk.LOW,
    pm_vol_ratio: float = 0.0,
    data_available: bool = True,
) -> GapInfo:
    return GapInfo(
        ticker=ticker,
        prev_close=100.0,
        pm_last_price=100.0 * (1 + gap_pct / 100),
        gap_pct=gap_pct,
        gap_type=gap_type,
        gap_risk=gap_risk,
        pm_volume=int(pm_vol_ratio * 1_000_000),
        avg_daily_volume=1_000_000.0,
        pm_vol_ratio=pm_vol_ratio,
        high_conviction=pm_vol_ratio >= PM_VOL_CONVICTION_RATIO,
        as_of=_NOW,
        data_available=data_available,
    )


def make_daily_history(close: float = 100.0, volume: int = 1_000_000, n: int = 25) -> pd.DataFrame:
    idx = pd.date_range("2024-05-01", periods=n, freq="B", tz=_ET)
    return pd.DataFrame({
        "Open":   [close * 0.99] * n,
        "High":   [close * 1.02] * n,
        "Low":    [close * 0.98] * n,
        "Close":  [close] * n,
        "Volume": [volume] * n,
    }, index=idx)


def make_intraday_history(
    date_str: str = "2024-06-03",
    pm_price: float = 105.0,
    pm_volume: int = 90_000,
) -> pd.DataFrame:
    """Simulate yfinance 1m prepost=True bars: 3 pre-market + 1 regular."""
    pm_times = pd.DatetimeIndex([
        pd.Timestamp(f"{date_str} 04:00", tz=_ET),
        pd.Timestamp(f"{date_str} 05:00", tz=_ET),
        pd.Timestamp(f"{date_str} 08:00", tz=_ET),
    ])
    reg_times = pd.DatetimeIndex([
        pd.Timestamp(f"{date_str} 09:30", tz=_ET),
    ])
    idx = pm_times.append(reg_times)
    n = len(idx)
    return pd.DataFrame({
        "Open":   [pm_price] * n,
        "High":   [pm_price * 1.005] * n,
        "Low":    [pm_price * 0.995] * n,
        "Close":  [pm_price] * n,
        "Volume": [pm_volume // 3] * 3 + [200_000],
    }, index=idx)


# ── _classify_gap ─────────────────────────────────────────────────────────────

class TestClassifyGap:
    def test_flat_small_positive(self):
        gt, gr = _classify_gap(0.5)
        assert gt == GapType.FLAT
        assert gr == GapRisk.LOW

    def test_flat_small_negative(self):
        gt, gr = _classify_gap(-0.5)
        assert gt == GapType.FLAT
        assert gr == GapRisk.LOW

    def test_moderate_gap_up(self):
        gt, gr = _classify_gap(3.0)
        assert gt == GapType.UP
        assert gr == GapRisk.MODERATE

    def test_moderate_gap_down(self):
        gt, gr = _classify_gap(-3.0)
        assert gt == GapType.DOWN
        assert gr == GapRisk.MODERATE

    def test_high_gap_up(self):
        gt, gr = _classify_gap(HIGH_GAP_SUPPRESS_PCT + 1)
        assert gt == GapType.UP
        assert gr == GapRisk.HIGH

    def test_high_gap_down(self):
        gt, gr = _classify_gap(-(HIGH_GAP_SUPPRESS_PCT + 1))
        assert gt == GapType.DOWN
        assert gr == GapRisk.HIGH

    def test_exactly_at_moderate_threshold(self):
        gt, gr = _classify_gap(MODERATE_GAP_WARN_PCT)
        assert gt == GapType.UP
        assert gr == GapRisk.MODERATE

    def test_exactly_at_high_threshold(self):
        _, gr = _classify_gap(HIGH_GAP_SUPPRESS_PCT)
        assert gr == GapRisk.HIGH

    def test_zero_gap(self):
        gt, gr = _classify_gap(0.0)
        assert gt == GapType.FLAT
        assert gr == GapRisk.LOW


# ── apply_gap_filter suppression logic ───────────────────────────────────────

class TestApplyGapFilter:

    def _signal(self, ticker: str, direction: str) -> dict:
        return {"ticker": ticker, "direction": direction, "entry": 100.0}

    def test_high_gap_up_suppresses_long(self):
        gap = make_gap_info("AAPL", gap_pct=6.0, gap_type=GapType.UP, gap_risk=GapRisk.HIGH)
        result = apply_gap_filter([self._signal("AAPL", "long")], {"AAPL": gap})
        assert result[0].suppressed is True
        assert result[0].size_scalar == 0.0
        assert "chasing" in result[0].suppress_reason

    def test_high_gap_down_suppresses_short(self):
        gap = make_gap_info("NVDA", gap_pct=-6.0, gap_type=GapType.DOWN, gap_risk=GapRisk.HIGH)
        result = apply_gap_filter([self._signal("NVDA", "short")], {"NVDA": gap})
        assert result[0].suppressed is True
        assert result[0].size_scalar == 0.0

    def test_high_gap_down_suppresses_long(self):
        gap = make_gap_info("MSFT", gap_pct=-7.0, gap_type=GapType.DOWN, gap_risk=GapRisk.HIGH)
        result = apply_gap_filter([self._signal("MSFT", "long")], {"MSFT": gap})
        assert result[0].suppressed is True
        assert "invalidates" in result[0].suppress_reason

    def test_high_gap_up_suppresses_short(self):
        gap = make_gap_info("TSLA", gap_pct=8.0, gap_type=GapType.UP, gap_risk=GapRisk.HIGH)
        result = apply_gap_filter([self._signal("TSLA", "short")], {"TSLA": gap})
        assert result[0].suppressed is True
        assert "squeeze" in result[0].suppress_reason

    def test_moderate_gap_halves_size(self):
        gap = make_gap_info("AMD", gap_pct=3.0, gap_type=GapType.UP, gap_risk=GapRisk.MODERATE)
        result = apply_gap_filter([self._signal("AMD", "long")], {"AMD": gap})
        assert result[0].suppressed is False
        assert result[0].size_scalar == 0.5
        assert any("50%" in n for n in result[0].notes)

    def test_low_risk_passes_full_size(self):
        gap = make_gap_info("AMZN", gap_pct=0.3, gap_type=GapType.FLAT, gap_risk=GapRisk.LOW)
        result = apply_gap_filter([self._signal("AMZN", "long")], {"AMZN": gap})
        assert result[0].suppressed is False
        assert result[0].size_scalar == 1.0

    def test_no_gap_data_passes_through(self):
        result = apply_gap_filter([self._signal("XYZ", "long")], {})
        assert result[0].suppressed is False
        assert "unavailable" in result[0].notes[0]

    def test_unavailable_gap_data_passes_through(self):
        gap = make_gap_info("XYZ", data_available=False)
        result = apply_gap_filter([self._signal("XYZ", "long")], {"XYZ": gap})
        assert result[0].suppressed is False

    def test_multiple_signals_mixed_outcomes(self):
        gaps = {
            "TSLA": make_gap_info("TSLA", gap_pct=6.0, gap_type=GapType.UP, gap_risk=GapRisk.HIGH),
            "MSFT": make_gap_info("MSFT", gap_pct=0.1, gap_type=GapType.FLAT, gap_risk=GapRisk.LOW),
            "NVDA": make_gap_info("NVDA", gap_pct=3.0, gap_type=GapType.UP, gap_risk=GapRisk.MODERATE),
        }
        signals = [
            self._signal("TSLA", "long"),
            self._signal("MSFT", "long"),
            self._signal("NVDA", "long"),
        ]
        result = apply_gap_filter(signals, gaps)
        assert result[0].suppressed is True    # TSLA: high gap up, long → suppressed
        assert result[1].suppressed is False   # MSFT: flat → pass
        assert result[2].size_scalar == 0.5    # NVDA: moderate → half size


# ── Conviction flag ───────────────────────────────────────────────────────────

class TestConviction:
    def test_high_conviction_when_above_threshold(self):
        gap = make_gap_info(pm_vol_ratio=PM_VOL_CONVICTION_RATIO + 0.01)
        assert gap.high_conviction is True

    def test_no_conviction_below_threshold(self):
        gap = make_gap_info(pm_vol_ratio=PM_VOL_CONVICTION_RATIO - 0.01)
        assert gap.high_conviction is False

    def test_conviction_note_added_to_signal(self):
        gap = make_gap_info("AAPL", gap_pct=0.5, gap_type=GapType.FLAT,
                            gap_risk=GapRisk.LOW, pm_vol_ratio=0.15)
        result = apply_gap_filter([{"ticker": "AAPL", "direction": "long"}], {"AAPL": gap})
        assert any("conviction" in n.lower() for n in result[0].notes)


# ── GapInfo.summary() ─────────────────────────────────────────────────────────

class TestGapInfoSummary:
    def test_summary_contains_ticker(self):
        gap = make_gap_info("AAPL", gap_pct=3.5)
        assert "AAPL" in gap.summary()

    def test_summary_unavailable(self):
        gap = make_gap_info("XYZ", data_available=False)
        assert "no pre-market data" in gap.summary()

    def test_summary_shows_conviction(self):
        gap = make_gap_info("TSLA", pm_vol_ratio=0.20)
        assert "CONVICTION" in gap.summary()


# ── fetch_gap_info error handling ─────────────────────────────────────────────

class TestFetchGapInfoErrors:
    def test_yfinance_error_returns_unavailable_gap_info(self):
        with patch("pre_market.yf.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.history.side_effect = Exception("network error")
            mock_ticker_cls.return_value = mock_ticker

            result = fetch_gap_info(["FAIL"])
            assert "FAIL" in result
            assert result["FAIL"].data_available is False
            assert result["FAIL"].error is not None

    def test_empty_history_returns_unavailable(self):
        with patch("pre_market.yf.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame()
            mock_ticker_cls.return_value = mock_ticker

            result = fetch_gap_info(["EMPTY"])
            assert result["EMPTY"].data_available is False


# ── fetch_gap_info happy path ─────────────────────────────────────────────────

class TestFetchGapInfoHappyPath:
    def test_gap_up_detected(self):
        daily = make_daily_history(close=100.0)
        intraday = make_intraday_history(pm_price=107.0)

        with patch("pre_market.yf.Ticker") as mock_ticker_cls:
            with patch("pre_market.datetime") as mock_dt:
                mock_dt.now.return_value = _NOW
                mock_ticker = MagicMock()
                mock_ticker.history.side_effect = [daily, intraday]
                mock_ticker_cls.return_value = mock_ticker

                result = fetch_gap_info(["AAPL"])

        gap = result["AAPL"]
        assert gap.data_available is True
        assert gap.gap_pct > 0
        assert gap.gap_type == GapType.UP

    def test_no_premarket_bars_returns_flat_gap(self):
        daily = make_daily_history(close=100.0)
        reg_only = pd.DataFrame({
            "Open": [100.0], "High": [101.0], "Low": [99.0],
            "Close": [100.5], "Volume": [500_000],
        }, index=pd.DatetimeIndex([pd.Timestamp("2024-06-03 09:30", tz=_ET)]))

        with patch("pre_market.yf.Ticker") as mock_ticker_cls:
            with patch("pre_market.datetime") as mock_dt:
                mock_dt.now.return_value = _NOW
                mock_ticker = MagicMock()
                mock_ticker.history.side_effect = [daily, reg_only]
                mock_ticker_cls.return_value = mock_ticker

                result = fetch_gap_info(["AAPL"])

        gap = result["AAPL"]
        assert gap.gap_pct == 0.0
        assert gap.pm_volume == 0


# ── premarket_snapshot DataFrame ─────────────────────────────────────────────

class TestPremarketSnapshot:
    def test_snapshot_returns_dataframe(self):
        daily = make_daily_history(close=100.0)
        intraday = make_intraday_history(pm_price=103.0)

        with patch("pre_market.yf.Ticker") as mock_ticker_cls:
            with patch("pre_market.datetime") as mock_dt:
                mock_dt.now.return_value = _NOW
                mock_ticker = MagicMock()
                mock_ticker.history.side_effect = [daily, intraday]
                mock_ticker_cls.return_value = mock_ticker

                df = premarket_snapshot(["AAPL"])

        assert isinstance(df, pd.DataFrame)
        assert "ticker" in df.columns
        assert "gap_pct" in df.columns
        assert "gap_risk" in df.columns
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "AAPL"

    def test_snapshot_sorted_by_gap_pct_descending(self):
        gaps_mock = {
            "A": make_gap_info("A", gap_pct=-3.0, gap_type=GapType.DOWN, gap_risk=GapRisk.MODERATE),
            "B": make_gap_info("B", gap_pct=6.0, gap_type=GapType.UP, gap_risk=GapRisk.HIGH),
            "C": make_gap_info("C", gap_pct=0.5, gap_type=GapType.FLAT, gap_risk=GapRisk.LOW),
        }
        with patch("pre_market.fetch_gap_info", return_value=gaps_mock):
            df = premarket_snapshot(["A", "B", "C"])

        assert df.iloc[0]["ticker"] == "B"   # highest gap first
        assert df.iloc[-1]["ticker"] == "A"  # most negative last

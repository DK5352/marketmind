"""
MarketMind — Volatile Stocks Today Predictor
=============================================
1. Measures 30-day annualised volatility for a broad candidate universe.
2. Picks the 10 most volatile US stocks.
3. Uses yesterday's close + technical indicators to predict TODAY's price.
4. Compares the prediction against the actual opening price (or latest price if market is open).

Run:
    cd /Users/divyakapoor/marketmind
    .venv/bin/python volatile_predict.py
"""

import json
import os
import sys
from datetime import date, datetime, timedelta

import anthropic
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ── Candidate universe ────────────────────────────────────────────────────────
# A broad list of US stocks known for high volatility.
# We'll rank them by 30-day realised volatility and keep the top 10.

CANDIDATES = [
    # Meme / high-beta tech
    "TSLA", "NVDA", "AMD", "MSTR", "COIN",
    # Biotech / pharma (huge volatility on clinical trial news)
    "MRNA", "BNTX", "NVAX", "SAVA", "ACMR",
    # Small-cap tech / EV
    "RIVN", "LCID", "NKLA", "SOFI", "UPST",
    # Crypto-adjacent
    "MARA", "RIOT", "HUT", "CLSK", "BTBT",
    # High-beta large cap
    "META", "SNAP", "RBLX", "HOOD", "DKNG",
    # Volatile ETFs as fillers if needed
    "GME", "AMC", "BBBY",
]

# Remove dupes while preserving order
CANDIDATES = list(dict.fromkeys(CANDIDATES))


# ── Helper: get 30-day annualised volatility ──────────────────────────────────

def get_volatility(ticker: str) -> float | None:
    """Returns annualised 30-day volatility (%) or None on failure."""
    try:
        hist = yf.Ticker(ticker).history(period="60d")
        if len(hist) < 20:
            return None
        returns = hist["Close"].pct_change().dropna()
        vol = float(returns.tail(30).std() * (252 ** 0.5) * 100)
        return round(vol, 1)
    except Exception:
        return None


# ── Helper: get today's open price ───────────────────────────────────────────

def get_todays_open(ticker: str) -> float | None:
    """Returns today's opening price. Falls back to latest price if open not available."""
    try:
        data = yf.Ticker(ticker).history(period="2d", interval="1d")
        if data.empty:
            return None
        today_str = date.today().isoformat()
        for idx in reversed(data.index):
            row_date = idx.date() if hasattr(idx, "date") else idx
            if str(row_date) == today_str:
                open_price = float(data.loc[idx, "Open"])
                return round(open_price, 2)
        # If today's row not present (pre-market, weekend), return yesterday's close
        latest = float(data["Close"].iloc[-1])
        return round(latest, 2)
    except Exception:
        return None


# ── Helper: build a concise data summary for one stock ───────────────────────

def get_stock_summary(ticker: str) -> dict | None:
    """
    Pulls yesterday's close, recent technicals (RSI, MACD, EMA, Bollinger,
    volume ratio, stochastic), 30d volatility, and beta.
    Returns a dict suitable for passing to the AI prompt.
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1y")
        if len(hist) < 30:
            return None

        info = {}
        try:
            info = t.info or {}
        except Exception:
            pass

        close = hist["Close"]
        volume = hist["Volume"]

        # Prices
        prev_close = round(float(close.iloc[-1]), 2)

        # RSI (14)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi_series = 100 - (100 / (1 + rs))
        rsi = round(float(rsi_series.iloc[-1]), 1)
        rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        macd_signal = "Bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] else "Bearish"
        hist_val = round(float(histogram.iloc[-1]), 4)

        # EMAs
        ema20 = round(float(close.ewm(span=20, adjust=False).mean().iloc[-1]), 2)
        ema200 = None
        if len(hist) >= 200:
            ema200 = round(float(close.ewm(span=200, adjust=False).mean().iloc[-1]), 2)

        # Bollinger Bands (20, 2)
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_pos_pct = round(
            float((close.iloc[-1] - bb_lower.iloc[-1]) /
                  (bb_upper.iloc[-1] - bb_lower.iloc[-1]) * 100), 1
        )
        bb_signal = "Near Upper" if bb_pos_pct > 80 else "Near Lower" if bb_pos_pct < 20 else "Mid Band"

        # Stochastic (14, 3)
        low14 = hist["Low"].rolling(14).min()
        high14 = hist["High"].rolling(14).max()
        stoch_k = round(float((close.iloc[-1] - low14.iloc[-1]) /
                              (high14.iloc[-1] - low14.iloc[-1]) * 100), 1)
        stoch_signal = "Overbought" if stoch_k > 80 else "Oversold" if stoch_k < 20 else "Neutral"

        # Volume ratio (today vs 20-day average)
        vol_avg = float(volume.rolling(20).mean().iloc[-1])
        vol_ratio = round(float(volume.iloc[-1]) / vol_avg, 2) if vol_avg else 1.0
        vol_signal = "Spike" if vol_ratio > 1.5 else "Low" if vol_ratio < 0.7 else "Normal"

        # 30d volatility + beta
        returns = close.pct_change().dropna()
        vol_30d = round(float(returns.tail(30).std() * (252 ** 0.5) * 100), 1)
        beta = info.get("beta", None)
        beta = round(float(beta), 2) if beta else None

        # 7-day return
        seven_day_ret = round(float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100), 2) if len(close) >= 6 else None

        return {
            "ticker": ticker,
            "name": info.get("shortName", ticker),
            "prev_close": prev_close,
            "seven_day_ret": seven_day_ret,
            "rsi": rsi,
            "rsi_signal": rsi_signal,
            "macd_signal": macd_signal,
            "macd_histogram": hist_val,
            "ema20": ema20,
            "ema200": ema200,
            "bb_pos_pct": bb_pos_pct,
            "bb_signal": bb_signal,
            "stoch_k": stoch_k,
            "stoch_signal": stoch_signal,
            "vol_ratio": vol_ratio,
            "vol_signal": vol_signal,
            "vol_30d": vol_30d,
            "beta": beta,
        }
    except Exception as e:
        return None


# ── AI: predict today's price for all stocks in one call ─────────────────────

def predict_todays_prices(stocks: list[dict]) -> dict:
    """
    Sends all stock data to Claude and asks it to predict today's closing price
    based on yesterday's close and technical signals.

    Returns { "TICKER": {"predicted_price": 123.45, "direction": "UP/DOWN/SIDEWAYS",
                          "confidence": "High/Medium/Low", "reason": "..."} }
    """
    client = anthropic.Anthropic()

    lines = []
    for s in stocks:
        b = f"Beta: {s['beta']}" if s["beta"] else "Beta: N/A"
        e200 = f"${s['ema200']}" if s["ema200"] else "N/A (insufficient history)"
        block = (
            f"\n{s['ticker']} ({s['name']}):\n"
            f"  Yesterday's close: ${s['prev_close']}\n"
            f"  7-day return: {s['seven_day_ret']}%\n"
            f"  30d annualised volatility: {s['vol_30d']}%  |  {b}\n"
            f"  RSI (14d): {s['rsi']}  ({s['rsi_signal']})\n"
            f"  MACD signal: {s['macd_signal']}  |  Histogram: {s['macd_histogram']}\n"
            f"  EMA 20d: ${s['ema20']}  |  EMA 200d: {e200}\n"
            f"  Bollinger position: {s['bb_pos_pct']}%  ({s['bb_signal']})\n"
            f"  Stochastic %K: {s['stoch_k']}  ({s['stoch_signal']})\n"
            f"  Volume signal: {s['vol_signal']}  |  Ratio vs 20d avg: {s['vol_ratio']}x\n"
        )
        lines.append(block)

    prompt = (
        f"Today is {date.today().strftime('%B %d, %Y')}.\n\n"
        "You are a quantitative trader. Below is technical data for 10 highly volatile US stocks.\n"
        "For each stock, predict where its price will CLOSE today.\n\n"
        "Important context:\n"
        "- These are the 10 MOST VOLATILE stocks from a broad US universe right now.\n"
        "- High volatility means large intraday swings are expected — predictions carry real uncertainty.\n"
        "- Base your prediction on: momentum direction, RSI, MACD, Bollinger position, and volume signals.\n"
        "- For stocks with very high volatility (>60% annualised), widen your uncertainty — small moves\n"
        "  are less likely; but direction is still more predictable than magnitude.\n\n"
        "Framework:\n"
        "- RSI > 70 = overbought → lean bearish for today\n"
        "- RSI < 30 = oversold → lean bullish for today (bounce potential)\n"
        "- MACD Bullish + positive histogram → upward momentum continuation\n"
        "- Price near upper Bollinger band → watch for resistance / pullback\n"
        "- Price near lower Bollinger band → potential bounce\n"
        "- Volume spike on prior day → continuation of prior day's move\n"
        "- Stochastic > 80 → overbought, < 20 → oversold\n\n"
        "STOCK DATA:\n"
        + "".join(lines) +
        "\nRespond with ONLY a valid JSON object, no prose, no markdown:\n"
        "{\n"
        '  "TICKER": {\n'
        '    "predicted_price": 123.45,\n'
        '    "direction": "UP" | "DOWN" | "SIDEWAYS",\n'
        '    "confidence": "High" | "Medium" | "Low",\n'
        '    "reason": "one sentence max"\n'
        "  },\n"
        "  ...\n"
        "}\n"
        "Be realistic — given high volatility, predicted moves of 2-8% are common. "
        "Confidence should mostly be Low or Medium for volatile stocks."
    )

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        preds = json.loads(raw)
        for ticker, p in preds.items():
            d = p.get("direction", "SIDEWAYS")
            p["arrow"] = "▲" if d == "UP" else "▼" if d == "DOWN" else "→"
        return preds
    except json.JSONDecodeError:
        print("[WARN] Could not parse Claude's JSON response.")
        print(raw[:500])
        return {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  MarketMind — 10 Most Volatile US Stocks: Today's Price Prediction")
    print("=" * 70)
    print(f"\n  Date: {date.today().strftime('%A, %B %d, %Y')}")
    print("\n  Step 1/3  Measuring 30-day volatility for candidate universe...")

    vol_map = {}
    for ticker in CANDIDATES:
        v = get_volatility(ticker)
        if v is not None:
            vol_map[ticker] = v
        print(f"    {ticker:<6} {v if v else 'N/A':>7}%", flush=True) if v else print(f"    {ticker:<6} skip")

    # Pick top 10 by volatility
    top10 = sorted(vol_map.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_tickers = [t for t, _ in top10]

    print(f"\n  Top 10 most volatile: {', '.join(top10_tickers)}")
    print("\n  Step 2/3  Fetching detailed technical data...")

    stocks_data = []
    for ticker in top10_tickers:
        print(f"    {ticker}...", end=" ", flush=True)
        summary = get_stock_summary(ticker)
        if summary:
            stocks_data.append(summary)
            print("OK")
        else:
            print("SKIP (data error)")

    if not stocks_data:
        print("\n  [ERROR] Could not retrieve data for any stock. Check network / yfinance.")
        sys.exit(1)

    print(f"\n  Step 3/3  Asking Claude to predict today's prices...")
    predictions = predict_todays_prices(stocks_data)

    # Fetch today's actual open/current price
    print("\n  Fetching today's actual opening prices from Yahoo Finance...")
    actuals = {}
    for s in stocks_data:
        t = s["ticker"]
        actuals[t] = get_todays_open(t)

    # ── Print results table ───────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  RESULTS — Prediction vs Actual (Today's Open)")
    print("=" * 70)
    print(
        f"  {'Ticker':<6}  {'Yest.Close':>10}  {'AI Predicts':>11}  "
        f"{'Actual Open':>11}  {'vs Pred':>8}  {'Conf':>6}  Direction  Reason"
    )
    print("  " + "-" * 90)

    for s in stocks_data:
        ticker = s["ticker"]
        pred_info = predictions.get(ticker, {})
        predicted = pred_info.get("predicted_price")
        arrow = pred_info.get("arrow", "?")
        confidence = pred_info.get("confidence", "?")
        reason = pred_info.get("reason", "")[:55]
        actual_open = actuals.get(ticker)

        prev_close_str = f"${s['prev_close']:.2f}"
        pred_str = f"${predicted:.2f}" if predicted else "N/A"

        if actual_open:
            actual_str = f"${actual_open:.2f}"
            if predicted:
                diff = actual_open - predicted
                diff_pct = diff / predicted * 100
                diff_str = f"{'+' if diff >= 0 else ''}{diff_pct:.1f}%"
            else:
                diff_str = "N/A"
        else:
            actual_str = "N/A"
            diff_str = "N/A"

        print(
            f"  {ticker:<6}  {prev_close_str:>10}  {pred_str:>11}  "
            f"{actual_str:>11}  {diff_str:>8}  {confidence:>6}  "
            f"{arrow} {pred_info.get('direction', '?'):<8}  {reason}"
        )

    print("\n  " + "-" * 90)
    print("  Note: 'vs Pred' = how much the actual open differed from AI prediction")
    print("        Positive = actual was higher than predicted")
    print("        Negative = actual was lower than predicted")

    # ── Volatility ranking summary ────────────────────────────────────────────
    print("\n  VOLATILITY RANKING (30-day annualised)")
    print("  " + "-" * 40)
    for ticker, vol in top10:
        bar = "#" * int(vol / 5)
        print(f"  {ticker:<6}  {vol:>6.1f}%  {bar}")

    print("\n" + "=" * 70 + "\n")

    # ── Save to file ──────────────────────────────────────────────────────────
    os.makedirs("reports", exist_ok=True)
    fname = f"reports/volatile_predict_{date.today().isoformat()}.txt"
    with open(fname, "w") as f:
        f.write(f"MarketMind — Volatile Stocks Prediction\n")
        f.write(f"Date: {date.today().isoformat()}\n")
        f.write(f"Top 10 most volatile: {', '.join(top10_tickers)}\n\n")
        f.write(f"{'Ticker':<6}  {'Yest.Close':>10}  {'AI Predicts':>11}  {'Actual Open':>11}  {'vs Pred':>8}  Confidence  Direction  Reason\n")
        f.write("-" * 90 + "\n")
        for s in stocks_data:
            ticker = s["ticker"]
            pred_info = predictions.get(ticker, {})
            predicted = pred_info.get("predicted_price")
            actual_open = actuals.get(ticker)
            conf = pred_info.get("confidence", "?")
            direction = pred_info.get("direction", "?")
            reason = pred_info.get("reason", "")[:55]
            prev_str = f"${s['prev_close']:.2f}"
            pred_str = f"${predicted:.2f}" if predicted else "N/A"
            actual_str = f"${actual_open:.2f}" if actual_open else "N/A"
            if actual_open and predicted:
                diff_pct = (actual_open - predicted) / predicted * 100
                diff_str = f"{'+' if diff_pct >= 0 else ''}{diff_pct:.1f}%"
            else:
                diff_str = "N/A"
            f.write(f"{ticker:<6}  {prev_str:>10}  {pred_str:>11}  {actual_str:>11}  {diff_str:>8}  {conf:>10}  {direction:<9}  {reason}\n")
        f.write("\nVolatility ranking:\n")
        for ticker, vol in top10:
            f.write(f"  {ticker:<6}  {vol:.1f}%\n")
    print(f"  Saved to {fname}\n")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift()
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_vp = (typical * df["volume"]).cumsum()
    cum_v = df["volume"].cumsum().replace(0, np.nan)
    return cum_vp / cum_v


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(d_period, min_periods=d_period).mean()
    return k, d


def summary(df: pd.DataFrame) -> dict:
    """Compute a rolled-up indicator snapshot from OHLCV history."""
    if df.empty:
        return {}
    close = df["close"]
    last = close.iloc[-1]
    out = {
        "sma20": sma(close, 20).iloc[-1],
        "sma50": sma(close, 50).iloc[-1],
        "sma200": sma(close, 200).iloc[-1],
        "ema20": ema(close, 20).iloc[-1],
        "rsi14": rsi(close, 14).iloc[-1],
        "atr14": atr(df["high"], df["low"], close, 14).iloc[-1],
        "52w_high": df["high"].tail(252).max(),
        "52w_low": df["low"].tail(252).min(),
        "avg_vol_20": df["volume"].tail(20).mean(),
        "last": last,
    }
    macd_line, signal_line, hist = macd(close)
    out.update({
        "macd": macd_line.iloc[-1],
        "macd_signal": signal_line.iloc[-1],
        "macd_hist": hist.iloc[-1],
    })
    k, d = stochastic(df["high"], df["low"], close)
    out["stoch_k"] = k.iloc[-1]
    out["stoch_d"] = d.iloc[-1]
    return out

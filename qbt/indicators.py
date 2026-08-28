"""
テクニカル指標ライブラリ。

すべての関数は pandas.Series を受け取り pandas.Series を返す。
戦略ファイル(YAML)の式の中から、そのままの名前で呼び出せる。

重要な設計方針:
  すべての指標は「その時点までの情報だけ」で計算される（未来を見ない）。
  shift(-n) のような未来参照は絶対に使わない。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 移動平均系

def sma(s: pd.Series, n: int) -> pd.Series:
    """単純移動平均"""
    return s.rolling(int(n), min_periods=int(n)).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    """指数移動平均"""
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    """加重移動平均"""
    n = int(n)
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n, min_periods=n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


# ---------------------------------------------------------------- ボラティリティ系

def stdev(s: pd.Series, n: int) -> pd.Series:
    """標準偏差"""
    return s.rolling(int(n), min_periods=int(n)).std(ddof=0)


def bb_upper(s: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """ボリンジャーバンド上限"""
    return sma(s, n) + k * stdev(s, n)


def bb_lower(s: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """ボリンジャーバンド下限"""
    return sma(s, n) - k * stdev(s, n)


def bb_pctb(s: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """%b: バンド内の相対位置。0=下限, 1=上限"""
    up, lo = bb_upper(s, n, k), bb_lower(s, n, k)
    width = (up - lo).replace(0, np.nan)
    return (s - lo) / width


def zscore(s: pd.Series, n: int = 20) -> pd.Series:
    """移動平均からの乖離を標準偏差で割った値（平均回帰戦略の基本形）"""
    sd = stdev(s, n).replace(0, np.nan)
    return (s - sma(s, n)) / sd


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Average True Range（実効的な値幅）"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / int(n), adjust=False, min_periods=int(n)).mean()


def atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """ATRを終値で割った値（%表記のボラティリティ。銘柄間比較に使える）"""
    return atr(high, low, close, n) / close


def hist_vol(s: pd.Series, n: int = 20) -> pd.Series:
    """年率換算ヒストリカルボラティリティ"""
    r = np.log(s / s.shift(1))
    return r.rolling(int(n), min_periods=int(n)).std(ddof=0) * np.sqrt(252.0)


# ---------------------------------------------------------------- オシレーター系

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """RSI（Wilder方式）"""
    n = int(n)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss が 0（全戻しなし）のときは RSI=100
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def macd(s: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """MACD線"""
    return ema(s, fast) - ema(s, slow)


def macd_signal(s: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    """MACDシグナル線"""
    return ema(macd(s, fast, slow), sig)


def macd_hist(s: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    """MACDヒストグラム"""
    return macd(s, fast, slow) - macd_signal(s, fast, slow, sig)


def stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """ストキャスティクス %K"""
    n = int(n)
    ll = low.rolling(n, min_periods=n).min()
    hh = high.rolling(n, min_periods=n).max()
    return 100.0 * (close - ll) / (hh - ll).replace(0, np.nan)


# ---------------------------------------------------------------- 価格位置・モメンタム系

def highest(s: pd.Series, n: int) -> pd.Series:
    """過去n本の最高値（当日を含む）"""
    return s.rolling(int(n), min_periods=int(n)).max()


def lowest(s: pd.Series, n: int) -> pd.Series:
    """過去n本の最安値（当日を含む）"""
    return s.rolling(int(n), min_periods=int(n)).min()


def roc(s: pd.Series, n: int = 20) -> pd.Series:
    """変化率（モメンタム）。n日前比の騰落率"""
    return s / s.shift(int(n)) - 1.0


def pct_from_high(s: pd.Series, n: int = 252) -> pd.Series:
    """過去n日高値からの下落率（マイナス値。-0.1 なら高値から10%下）"""
    return s / highest(s, n) - 1.0


def slope(s: pd.Series, n: int = 20) -> pd.Series:
    """線形回帰の傾きを価格で正規化した値（トレンドの強さ）"""
    n = int(n)
    x = np.arange(n, dtype=float)
    x_c = x - x.mean()
    denom = (x_c ** 2).sum()

    def _f(y: np.ndarray) -> float:
        return float(np.dot(x_c, y - y.mean()) / denom)

    return s.rolling(n, min_periods=n).apply(_f, raw=True) / s


# ---------------------------------------------------------------- 補助関数

def _as_series(x, like: pd.Series) -> pd.Series:
    """スカラーを Series に揃える（cross_above(rsi(...), 30) のような書き方を許すため）"""
    return x if isinstance(x, pd.Series) else pd.Series(float(x), index=like.index)


def cross_above(a: pd.Series, b) -> pd.Series:
    """aがbを下から上に抜けた瞬間にTrue（ゴールデンクロス）。bは数値でもよい"""
    b = _as_series(b, a)
    return (a > b) & (a.shift(1) <= b.shift(1))


def cross_below(a: pd.Series, b) -> pd.Series:
    """aがbを上から下に抜けた瞬間にTrue（デッドクロス）。bは数値でもよい"""
    b = _as_series(b, a)
    return (a < b) & (a.shift(1) >= b.shift(1))


def rising(s: pd.Series, n: int = 1) -> pd.Series:
    """n本連続で上昇しているか"""
    up = s > s.shift(1)
    return up.rolling(int(n), min_periods=int(n)).sum() == int(n)


def falling(s: pd.Series, n: int = 1) -> pd.Series:
    """n本連続で下落しているか"""
    dn = s < s.shift(1)
    return dn.rolling(int(n), min_periods=int(n)).sum() == int(n)


def shift(s: pd.Series, n: int = 1) -> pd.Series:
    """n本前の値。nは正のみ許可（未来参照を防ぐため）"""
    n = int(n)
    if n < 0:
        raise ValueError("shift() に負の値は使えません（未来を見ることになります）")
    return s.shift(n)


def barssince(cond: pd.Series) -> pd.Series:
    """条件が最後にTrueになってから何本経過したか"""
    c = cond.fillna(False).astype(bool)
    idx = np.arange(len(c))
    last = pd.Series(np.where(c, idx, np.nan), index=c.index).ffill()
    return pd.Series(idx, index=c.index) - last


# 戦略式から呼び出せる関数の一覧（安全な eval のための白リスト）
INDICATOR_NAMESPACE = {
    "sma": sma, "ema": ema, "wma": wma,
    "stdev": stdev, "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_pctb": bb_pctb,
    "zscore": zscore, "atr": atr, "atr_pct": atr_pct, "hist_vol": hist_vol,
    "rsi": rsi, "macd": macd, "macd_signal": macd_signal, "macd_hist": macd_hist,
    "stoch_k": stoch_k,
    "highest": highest, "lowest": lowest, "roc": roc,
    "pct_from_high": pct_from_high, "slope": slope,
    "cross_above": cross_above, "cross_below": cross_below,
    "rising": rising, "falling": falling, "shift": shift, "barssince": barssince,
    "abs": abs, "min": min, "max": max,
}

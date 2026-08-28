"""
成績評価。

「年率何%」だけ見ても戦略の良し悪しは分からない。
ドローダウン、取引回数、期待値、そして「たまたま勝っただけの確率」まで見る。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def volatility(equity: pd.Series) -> float:
    r = _returns(equity)
    return float(r.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else 0.0


def sharpe(equity: pd.Series, rf: float = 0.0) -> float:
    r = _returns(equity)
    if len(r) < 2 or r.std(ddof=0) == 0:
        return 0.0
    excess = r - rf / TRADING_DAYS
    return float(excess.mean() / r.std(ddof=0) * np.sqrt(TRADING_DAYS))


def sortino(equity: pd.Series, rf: float = 0.0) -> float:
    r = _returns(equity)
    downside = r[r < 0]
    if len(downside) < 2 or downside.std(ddof=0) == 0:
        return 0.0
    return float((r.mean() - rf / TRADING_DAYS) / downside.std(ddof=0) * np.sqrt(TRADING_DAYS))


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min()) if len(equity) else 0.0


def max_dd_duration_days(equity: pd.Series) -> int:
    """最大ドローダウンからの回復に要した最長日数（水面下にいた期間）"""
    if len(equity) < 2:
        return 0
    peak = equity.cummax()
    underwater = equity < peak
    longest = cur = 0
    start = None
    for date, uw in underwater.items():
        if uw:
            start = start or date
            cur = (date - start).days
            longest = max(longest, cur)
        else:
            start, cur = None, 0
    return int(longest)


def calmar(equity: pd.Series) -> float:
    mdd = abs(max_drawdown(equity))
    return float(cagr(equity) / mdd) if mdd > 1e-9 else 0.0


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"取引回数": 0, "勝率": 0.0, "平均利益": 0.0, "平均損失": 0.0,
                "ペイオフレシオ": 0.0, "プロフィットファクター": 0.0,
                "期待値(1回あたり%)": 0.0, "平均保有日数": 0.0, "最大連敗": 0}
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    gross_win = float(wins["pnl"].sum())
    gross_loss = float(-losses["pnl"].sum())
    avg_w = float(wins["pnl_pct"].mean()) if len(wins) else 0.0
    avg_l = float(losses["pnl_pct"].mean()) if len(losses) else 0.0

    streak = worst = 0
    for pnl in trades.sort_values("exit_date")["pnl"]:
        streak = streak + 1 if pnl <= 0 else 0
        worst = max(worst, streak)

    return {
        "取引回数": int(len(trades)),
        "勝率": len(wins) / len(trades),
        "平均利益": avg_w,
        "平均損失": avg_l,
        "ペイオフレシオ": abs(avg_w / avg_l) if avg_l else 0.0,
        "プロフィットファクター": gross_win / gross_loss if gross_loss > 1e-9 else float("inf"),
        "期待値(1回あたり%)": float(trades["pnl_pct"].mean()),
        "平均保有日数": float(trades["bars_held"].mean()),
        "最大連敗": int(worst),
    }


def t_stat(trades: pd.DataFrame) -> float:
    """
    「1回あたりの期待値がゼロ」という帰無仮説に対するt値。
    目安として |t| < 2 なら、その利益は偶然の範囲を出ていない。
    """
    if len(trades) < 3:
        return 0.0
    r = trades["pnl_pct"]
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(len(r))) if sd > 1e-12 else 0.0


def deflated_note(n_trades: int, t: float) -> str:
    if n_trades < 30:
        return "取引回数が少なすぎて統計的な判断ができない"
    if abs(t) < 1.0:
        return "期待値はゼロと区別できない（偶然の範囲）"
    if abs(t) < 2.0:
        return "有意とは言えない。パラメータをいじって出た数字の可能性が高い"
    if abs(t) < 3.0:
        return "一応有意だが、探索回数を考えると割り引いて見るべき"
    return "統計的にはかなり強い。ただし期間外検証で再現するかが本番"


def summary(equity: pd.Series, trades: pd.DataFrame,
            exposure: pd.Series | None = None,
            benchmark: pd.Series | None = None) -> dict:
    out = {
        "最終資産": float(equity.iloc[-1]) if len(equity) else 0.0,
        "累積リターン": total_return(equity),
        "年率リターン(CAGR)": cagr(equity),
        "年率ボラティリティ": volatility(equity),
        "シャープレシオ": sharpe(equity),
        "ソルティノレシオ": sortino(equity),
        "最大ドローダウン": max_drawdown(equity),
        "水面下最長期間(日)": max_dd_duration_days(equity),
        "カルマーレシオ": calmar(equity),
    }
    out.update(trade_stats(trades))
    out["t値"] = t_stat(trades)
    out["統計的評価"] = deflated_note(len(trades), out["t値"])
    if exposure is not None and len(exposure):
        out["平均建玉比率"] = float(exposure.mean())
    if benchmark is not None and len(benchmark) > 1:
        out["ベンチマーク年率"] = cagr(benchmark)
        out["ベンチマーク最大DD"] = max_drawdown(benchmark)
        out["超過年率"] = out["年率リターン(CAGR)"] - out["ベンチマーク年率"]
    return out


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """月次リターンの表（年×月）。特定の年だけで稼いでいないかを見る"""
    if len(equity) < 2:
        return pd.DataFrame()
    m = equity.resample("ME").last().pct_change().dropna()
    df = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.values})
    return df.pivot(index="year", columns="month", values="ret")


def yearly_returns(equity: pd.Series) -> pd.Series:
    if len(equity) < 2:
        return pd.Series(dtype=float)
    y = equity.resample("YE").last()
    first = pd.Series([equity.iloc[0]], index=[equity.index[0]])
    y = pd.concat([first, y])
    return y.pct_change().dropna()

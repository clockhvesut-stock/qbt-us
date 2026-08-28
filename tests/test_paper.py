"""
ペーパートレードがバックテストと一致することを検証する。

これが一致していないと、検証で見た数字と実運用の記録を比べても意味がない。
同じ価格・同じルールを与えて、両者が同じ建玉・同じ損益を出すことを確かめる。

    python tests/test_paper.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbt import paper  # noqa: E402
from qbt.engine import Backtest, Costs, Portfolio, Rules, build_signal_frame  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'✓' if cond else '✗'} {name}" + (f"   {detail}" if detail else ""))


def make_data(n_sym=6, n_days=260, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n_days)
    out = {}
    for i in range(n_sym):
        r = rng.normal(0.0005, 0.018, n_days)
        base = float(rng.uniform(30, 250))
        close = base * np.exp(np.cumsum(r))
        op = close * (1 + rng.normal(0, 0.004, n_days))
        hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.006, n_days)))
        lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.006, n_days)))
        out[f"S{i}"] = pd.DataFrame(
            {"open": op, "high": hi, "low": lo, "close": close,
             "volume": rng.lognormal(14, 0.3, n_days)}, index=idx)
    return out


CFG = {
    "portfolio": {"initial_cash": 2000, "max_positions": 3, "lot_size": 1,
                  "allow_fractional": True},
    "costs": {"commission_bps": 0.0, "slippage_bps": 8.0},
}
RULES = {
    "entry": "cross_above(rsi(close,14), 40)",
    "exit": "rsi(close,14) > 65",
    "stop_loss_pct": 0.08,
    "max_hold_days": 15,
}


def run_paper(data, start_i=230):
    """ペーパートレードを1日ずつ進める"""
    tmp = tempfile.mkdtemp()
    book = paper.PaperBook(tmp, CFG, name="rule")
    cal = list(data["S0"].index)
    entry = build_signal_frame(RULES["entry"], data, pd.DatetimeIndex(cal))
    exit_s = build_signal_frame(RULES["exit"], data, pd.DatetimeIndex(cal))

    for i in range(start_i, len(cal)):
        d = str(cal[i].date())
        bars = {s: {"open": float(df["open"].iloc[i]), "high": float(df["high"].iloc[i]),
                    "low": float(df["low"].iloc[i]), "close": float(df["close"].iloc[i])}
                for s, df in data.items()}
        book.step(d, bars, RULES)

        flags = {p["symbol"]: ["手仕舞いシグナル"]
                 for p in book.positions if bool(exit_s.iloc[i].get(p["symbol"], False))}
        book.queue_exits(bars, RULES, flags)
        row = entry.iloc[i]
        cands = [{"symbol": s} for s in row.index if bool(row.get(s, False))]
        book.queue_entries(cands, market_ok=True)
        book.save(d)

    snap = book.snapshot({s: {"close": float(df["close"].iloc[-1])}
                          for s, df in data.items()})
    shutil.rmtree(tmp, ignore_errors=True)
    return snap, book


def run_backtest(data, start_i=230):
    cal = list(data["S0"].index)
    rules = Rules(entry=RULES["entry"], exit=RULES["exit"],
                  stop_loss_pct=RULES["stop_loss_pct"],
                  max_hold_days=RULES["max_hold_days"])
    pf = Portfolio(initial_cash=2000, max_positions=3, lot_size=1,
                   allow_fractional=True)
    bt = Backtest(data, rules, pf, Costs(commission_bps=0, slippage_bps=8))
    return bt.run(start=str(cal[start_i].date()))


# ====================================================================

def test_matches_backtest():
    print("\n[1] ペーパートレードとバックテストが一致すること")
    data = make_data()
    snap, book = run_paper(data)
    res = run_backtest(data)

    bt_final = float(res.equity.iloc[-1])
    pp_final = float(snap["equity"])
    diff = abs(pp_final - bt_final) / bt_final
    check("最終評価額がほぼ一致する", diff < 0.02,
          f"バックテスト ${bt_final:,.2f} / ペーパー ${pp_final:,.2f}（差 {diff*100:.2f}%）")

    bt_trades = len(res.trades_df)
    check("取引回数が近い", abs(len(book.trades) - bt_trades) <= 2,
          f"バックテスト {bt_trades}回 / ペーパー {len(book.trades)}回")

    if book.trades and bt_trades:
        bt_syms = set(res.trades_df["symbol"])
        pp_syms = {t["symbol"] for t in book.trades}
        overlap = len(bt_syms & pp_syms) / max(len(bt_syms | pp_syms), 1)
        check("売買した銘柄が概ね一致する", overlap >= 0.6,
              f"一致率 {overlap*100:.0f}%")


def test_constraints():
    print("\n[2] 資金と枠の制約が守られること")
    data = make_data(n_sym=8, seed=11)
    snap, book = run_paper(data, start_i=220)
    check("現金がマイナスにならない", book.cash >= -0.01, f"現金 ${book.cash:,.2f}")
    check("同時保有数が上限3を超えない", len(book.positions) <= 3,
          f"保有 {len(book.positions)}")
    check("建玉比率が100%を超えない",
          snap["market_value"] <= snap["equity"] * 1.001,
          f"建玉 ${snap['market_value']:,.2f} / 評価額 ${snap['equity']:,.2f}")


def test_stop_loss():
    print("\n[3] 逆指値が建値の-8%で執行されること")
    idx = pd.bdate_range("2025-01-01", periods=6)
    df = pd.DataFrame({
        "open":  [100, 100, 100, 99, 88, 88],
        "high":  [101, 101, 101, 99, 89, 89],
        "low":   [99, 99, 99, 85, 87, 87],   # 4日目に安値85 → -8%の92で損切り
        "close": [100, 100, 100, 88, 88, 88],
        "volume": [1e7] * 6}, index=idx)
    data = {"X": df}
    tmp = tempfile.mkdtemp()
    book = paper.PaperBook(tmp, CFG, name="rule")
    for i, d in enumerate(idx):
        bars = {"X": {k: float(df[k].iloc[i]) for k in ("open", "high", "low", "close")}}
        book.step(str(d.date()), bars, RULES)
        if i == 1:                       # 2日目の大引けで注文を出す
            book.queue_entries([{"symbol": "X"}])
        book.save(str(d.date()))
    shutil.rmtree(tmp, ignore_errors=True)

    check("損切りが記録されている", len(book.trades) == 1,
          f"{len(book.trades)}件")
    if book.trades:
        t = book.trades[0]
        ep, xp = t["entry_price"], t["exit_price"]
        # 建値は3日目の寄付100×(1+0.0008)、決済は92×(1-0.0008)
        check("決済価格が建値の-8%水準", abs(xp / ep - 0.92) < 0.003,
              f"建値 {ep:.3f} → 決済 {xp:.3f}（{(xp/ep-1)*100:.2f}%）")
        check("決済理由が損切り", t["exit_reason"] == "損切り", t["exit_reason"])


def test_idempotent():
    print("\n[4] 同じ日を二度処理しても建玉が重複しないこと")
    data = make_data(n_sym=3, seed=3)
    tmp = tempfile.mkdtemp()
    cal = list(data["S0"].index)
    book = paper.PaperBook(tmp, CFG, name="rule")
    i = 250
    d = str(cal[i].date())
    bars = {s: {k: float(df[k].iloc[i]) for k in ("open", "high", "low", "close")}
            for s, df in data.items()}
    book.queue_entries([{"symbol": "S0"}])
    book.step(d, bars, RULES)
    n1 = len(book.positions)
    book.save(d)

    book2 = paper.PaperBook(tmp, CFG, name="rule")
    book2.queue_entries([{"symbol": "S0"}])
    book2.step(d, bars, RULES)          # 同じ日をもう一度
    n2 = len(book2.positions)
    shutil.rmtree(tmp, ignore_errors=True)
    check("再実行しても保有数が増えない", n2 == n1, f"1回目 {n1} / 2回目 {n2}")


def test_persistence():
    print("\n[5] 状態がファイルに残り、次回に引き継がれること")
    data = make_data(n_sym=3, seed=5)
    tmp = tempfile.mkdtemp()
    cal = list(data["S0"].index)

    book = paper.PaperBook(tmp, CFG, name="rule")
    i = 240
    bars = {s: {k: float(df[k].iloc[i]) for k in ("open", "high", "low", "close")}
            for s, df in data.items()}
    book.queue_entries([{"symbol": "S1"}])
    book.step(str(cal[i].date()), bars, RULES)
    cash_before = book.cash
    book.save(str(cal[i].date()))

    reloaded = paper.PaperBook(tmp, CFG, name="rule")
    check("保有が復元される",
          [p["symbol"] for p in reloaded.positions] == [p["symbol"] for p in book.positions],
          f"{[p['symbol'] for p in reloaded.positions]}")
    check("現金が復元される", abs(reloaded.cash - cash_before) < 0.01,
          f"${reloaded.cash:,.2f}")
    for f in ("positions.json", "account.json", "trades.json", "pending.json"):
        check(f"{f} が作られている", os.path.exists(os.path.join(tmp, f)))
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("=" * 62)
    print("  ペーパートレード 検証テスト")
    print("=" * 62)
    for fn in (test_matches_backtest, test_constraints, test_stop_loss,
               test_idempotent, test_persistence):
        fn()
    print("\n" + "=" * 62)
    print(f"  合格 {len(PASS)} 件 / 不合格 {len(FAIL)} 件")
    for f in FAIL:
        print(f"    ✗ {f}")
    print("=" * 62)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

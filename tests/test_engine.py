"""
エンジンの正しさを検証するテスト。

バックテストのバグは「成績が良くなる方向」に出る。だから動いただけでは信用できない。
ここでは手計算できる状況を作って、エンジンの出す数字が一致するかを確かめる。

    python tests/test_engine.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbt import metrics as M  # noqa: E402
from qbt.engine import Backtest, Costs, ExpressionError, Portfolio, Rules, eval_expr  # noqa: E402
from qbt.indicators import rsi, sma  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'✓' if cond else '✗'} {name}" + (f"   {detail}" if detail else ""))


def mkdf(closes, opens=None, highs=None, lows=None, start="2020-01-01"):
    n = len(closes)
    idx = pd.bdate_range(start, periods=n)
    closes = np.array(closes, dtype=float)
    opens = np.array(opens, dtype=float) if opens is not None else closes.copy()
    highs = np.array(highs, dtype=float) if highs is not None else np.maximum(opens, closes)
    lows = np.array(lows, dtype=float) if lows is not None else np.minimum(opens, closes)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": np.full(n, 1e7)}, index=idx)


NO_COST = Costs(commission_bps=0, slippage_bps=0)


# ==================================================================== 1. 未来参照

def test_no_lookahead():
    print("\n[1] 未来参照ができないこと")
    df = mkdf([100, 110, 120, 130, 140])
    try:
        eval_expr("shift(close, -1) > close", df)
        check("shift に負値を渡すと拒否される", False, "拒否されなかった")
    except Exception:
        check("shift に負値を渡すと拒否される", True)

    try:
        eval_expr("close.shift(-1) > close", df)
        check("属性アクセス(.shift)が禁止されている", False, "通ってしまった")
    except ExpressionError:
        check("属性アクセス(.shift)が禁止されている", True)

    try:
        eval_expr("__import__('os').system('ls')", df)
        check("import 等の危険な構文が禁止されている", False, "通ってしまった")
    except Exception:
        check("import 等の危険な構文が禁止されている", True)

    # 指標そのものが未来を含んでいないか（末尾を書き換えても過去の値は変わらない）
    a = sma(df["close"], 3)
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("close")] = 9999.0
    b = sma(df2["close"], 3)
    check("指標の過去の値は将来の値に影響されない",
          bool((a.iloc[:-1].fillna(-1) == b.iloc[:-1].fillna(-1)).all()))


# ==================================================================== 2. 約定価格

def test_execution_price():
    print("\n[2] 約定は「翌営業日の寄付」で行われること")
    # 3日目にシグナル。買えるのは4日目の寄付=200 のはず（3日目の終値150ではない）
    df = mkdf(closes=[100, 100, 150, 150, 150],
              opens=[100, 100, 100, 200, 150])
    data = {"X": df}
    # 3日目(index=2)だけ True になる条件
    rules = Rules(entry="close > 120 and shift(close,1) < 120", exit=None)
    pf = Portfolio(initial_cash=1_000_000, max_positions=1, lot_size=1)
    res = Backtest(data, rules, pf, NO_COST).run()
    t = res.trades_df
    entry_px = None
    if not t.empty:
        entry_px = float(t.iloc[0]["entry_price"])
    else:
        # 手仕舞いしないので取引履歴には残らない。建値を資産曲線から逆算する
        entry_px = 200.0 if abs(res.equity.iloc[3] - 750_000) < 1 else None

    # 4日目寄付200で5000株買い → 現金0、大引け時価 150*5000=750,000
    check("翌営業日の寄付(200)で約定している",
          abs(res.equity.iloc[3] - 750_000) < 1.0,
          f"4日目終値評価額={res.equity.iloc[3]:,.0f} (期待 750,000)")
    check("シグナル当日(3日目)にはまだ建てていない",
          abs(res.equity.iloc[2] - 1_000_000) < 1.0,
          f"3日目={res.equity.iloc[2]:,.0f}")


# ==================================================================== 3. バイ&ホールド

def test_buy_and_hold():
    print("\n[3] 常時買いシグナル = バイ&ホールドと一致すること")
    rng = np.random.default_rng(3)
    closes = 1000 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, 500)))
    df = mkdf(closes, opens=closes)  # 寄付=終値にして誤差要因を消す
    data = {"X": df}
    rules = Rules(entry="close > 0", exit=None)
    pf = Portfolio(initial_cash=10_000_000, max_positions=1, lot_size=1)
    res = Backtest(data, rules, pf, NO_COST).run()

    # 2日目の寄付で全額買う → 以降は株価と同じ比率で動く
    buy_px = df["open"].iloc[1]
    shares = int(10_000_000 // buy_px)
    expected_final = shares * df["close"].iloc[-1] + (10_000_000 - shares * buy_px)
    check("最終資産がバイ&ホールドの手計算と一致する",
          abs(res.equity.iloc[-1] - expected_final) < 1.0,
          f"engine={res.equity.iloc[-1]:,.0f} / 手計算={expected_final:,.0f}")


# ==================================================================== 4. コスト

def test_costs_reduce_return():
    print("\n[4] コストが必ず成績を悪化させること")
    rng = np.random.default_rng(11)
    closes = 1000 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, 800)))
    df = mkdf(closes)
    data = {"X": df}
    rules = Rules(entry="rsi(close,14) < 45", exit="rsi(close,14) > 55")
    pf = Portfolio(initial_cash=5_000_000, max_positions=1, lot_size=1)

    free = Backtest(data, rules, pf, NO_COST).run()
    paid = Backtest(data, rules, pf, Costs(commission_bps=5, slippage_bps=10)).run()
    n = len(free.trades_df)
    check("コストありの方が最終資産が小さい",
          paid.equity.iloc[-1] < free.equity.iloc[-1],
          f"無料={free.equity.iloc[-1]:,.0f} / 有料={paid.equity.iloc[-1]:,.0f} ({n}回取引)")

    # 往復0.30%×取引回数ぶん、ざっくり目減りしているはず
    if n > 5:
        drag = 1 - paid.equity.iloc[-1] / free.equity.iloc[-1]
        expected = 1 - (1 - 0.0030) ** n
        check("目減り幅が往復コスト×取引回数とおおむね整合する",
              abs(drag - expected) < max(0.10, expected * 0.6),
              f"実測={drag*100:.1f}% / 理論={expected*100:.1f}%")


# ==================================================================== 5. 損切り

def test_stop_loss():
    print("\n[5] 逆指値が正しい価格で執行されること")
    #  1日目: 何もなし / 2日目寄付で買い(100) / 3日目に安値85まで下落 → -8%=92で損切り
    df = mkdf(closes=[100, 100, 88, 88, 88],
              opens=[100, 100, 99, 88, 88],
              highs=[101, 101, 99, 89, 89],
              lows=[99, 99, 85, 87, 87])
    data = {"X": df}
    rules = Rules(entry="close > 99 and shift(close,1) > 0 and barssince(close > 1e9) > 0",
                  exit=None, stop_loss_pct=0.08)
    # 1日目から常時シグナル。2日目寄付=100で建つ
    rules = Rules(entry="close >= 88", exit=None, stop_loss_pct=0.08)
    pf = Portfolio(initial_cash=1_000_000, max_positions=1, lot_size=1)
    res = Backtest(data, rules, pf, NO_COST).run()
    t = res.trades_df
    check("損切りが発生している", len(t) >= 1)
    if len(t):
        r = t.iloc[0]
        check("損切り価格 = 建値×0.92",
              abs(float(r["exit_price"]) - float(r["entry_price"]) * 0.92) < 1e-6,
              f"建値={r['entry_price']:.2f} 決済={r['exit_price']:.2f}")
        check("決済理由が「損切り」と記録される", r["exit_reason"] == "損切り")


# ==================================================================== 6. 資金と単元

def test_cash_and_lots():
    print("\n[6] 資金制約と単元株が守られること")
    rng = np.random.default_rng(5)
    data = {}
    for i in range(6):
        c = 1000 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, 400)))
        data[f"S{i}"] = mkdf(c)
    rules = Rules(entry="close > sma(close,20)", exit="close < sma(close,20)")
    pf = Portfolio(initial_cash=1_000_000, max_positions=6, lot_size=100)
    res = Backtest(data, rules, pf, Costs(commission_bps=5, slippage_bps=10)).run()

    check("資産がマイナスになっていない", bool((res.equity > 0).all()),
          f"最小={res.equity.min():,.0f}")
    check("同時保有数が上限を超えていない", int(res.positions_count.max()) <= 6,
          f"最大={int(res.positions_count.max())}")
    check("建玉比率が100%を超えていない（信用を使っていない）",
          float(res.exposure.max()) <= 1.0001,
          f"最大={res.exposure.max()*100:.1f}%")
    t = res.trades_df
    if not t.empty:
        check("全ての建玉が100株単位", bool((t["shares"] % 100 == 0).all()))


# ==================================================================== 7. 指標

def test_indicators():
    print("\n[7] 指標の値が定義どおりであること")
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    check("SMA(3) の最後の値 = (8+9+10)/3", abs(sma(s, 3).iloc[-1] - 9.0) < 1e-9)
    check("SMA(3) の最初の2つは NaN", bool(sma(s, 3).iloc[:2].isna().all()))

    up = pd.Series(np.arange(1, 40, dtype=float))
    check("単調上昇の RSI は 100", abs(rsi(up, 14).iloc[-1] - 100.0) < 1e-6,
          f"={rsi(up,14).iloc[-1]:.4f}")
    down = pd.Series(np.arange(40, 1, -1, dtype=float))
    check("単調下降の RSI は 0", abs(rsi(down, 14).iloc[-1] - 0.0) < 1e-6,
          f"={rsi(down,14).iloc[-1]:.4f}")


# ==================================================================== 8. 成績指標

def test_metrics():
    print("\n[8] 成績指標の計算が正しいこと")
    idx = pd.bdate_range("2020-01-01", periods=253)
    eq = pd.Series(np.linspace(100, 200, 253), index=idx)
    check("2倍になった1年の累積リターン = 100%", abs(M.total_return(eq) - 1.0) < 1e-9)
    check("単調増加なら最大DD = 0", abs(M.max_drawdown(eq)) < 1e-12)

    eq2 = pd.Series([100, 120, 60, 90, 150], index=pd.bdate_range("2020-01-01", periods=5))
    check("120→60 の最大DD = -50%", abs(M.max_drawdown(eq2) + 0.5) < 1e-9,
          f"={M.max_drawdown(eq2)*100:.1f}%")

    trades = pd.DataFrame({
        "pnl": [100, -50, 200, -80, 30],
        "pnl_pct": [0.10, -0.05, 0.20, -0.08, 0.03],
        "bars_held": [5, 3, 8, 2, 4],
        "exit_date": pd.bdate_range("2020-01-01", periods=5),
    })
    st = M.trade_stats(trades)
    check("勝率 3/5 = 60%", abs(st["勝率"] - 0.6) < 1e-9)
    check("プロフィットファクター = 330/130", abs(st["プロフィットファクター"] - 330 / 130) < 1e-9)
    check("最大連敗 = 1", st["最大連敗"] == 1)


# ==================================================================== 9. ランダムデータ

def test_random_data_gives_nothing():
    print("\n[9] ランダムな価格からは利益が出ないこと（最重要）")
    # ドリフトゼロの純粋なランダムウォーク。コストを引けば必ず負けるはず。
    # ここでプラスが出るならエンジンのどこかが未来を見ている。
    finals = []
    for seed in range(8):
        rng = np.random.default_rng(1000 + seed)
        data = {}
        for i in range(8):
            c = 1000 * np.exp(np.cumsum(rng.normal(0.0, 0.015, 1200)))
            data[f"R{i}"] = mkdf(c)
        rules = Rules(entry="close > sma(close,50) and rsi(close,14) < 40",
                      exit="rsi(close,14) > 65", rank="roc(close,60)",
                      stop_loss_pct=0.08, max_hold_days=40)
        pf = Portfolio(initial_cash=3_000_000, max_positions=4, lot_size=100)
        res = Backtest(data, rules, pf, Costs(commission_bps=5, slippage_bps=10)).run()
        finals.append(M.cagr(res.equity))
    mean_cagr = float(np.mean(finals))
    check("ランダムウォーク8本の平均年率が概ねゼロ以下",
          mean_cagr < 0.02,
          f"平均年率={mean_cagr*100:+.2f}%  （明確なプラスなら未来参照バグの疑い）")


# ==================================================================== 10. 端株

def test_fractional_shares():
    print("\n[10] 少額資金と端株の扱い")
    rng = np.random.default_rng(77)
    data = {}
    # 株価300ドル台の銘柄ばかりのユニバース。資金2,000ドルを8分割すると1銘柄250ドル。
    for i in range(6):
        c = 320 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, 600)))
        data[f"H{i}"] = mkdf(c)
    rules = Rules(entry="close > sma(close,20)", exit="cross_below(close, sma(close,20))")

    whole = Backtest(data, rules,
                     Portfolio(initial_cash=2000, max_positions=8, lot_size=1,
                               allow_fractional=False), NO_COST).run()
    frac = Backtest(data, rules,
                    Portfolio(initial_cash=2000, max_positions=8, lot_size=1,
                              allow_fractional=True), NO_COST).run()

    check("端株なしだと高価格帯の銘柄を買えず見送りが発生する",
          whole.skipped_unaffordable > 0,
          f"見送り {whole.skipped_unaffordable} 回")
    check("端株を有効にすると見送りがゼロになる",
          frac.skipped_unaffordable == 0 and len(frac.trades_df) > len(whole.trades_df),
          f"端株なし {len(whole.trades_df)}回 → 端株あり {len(frac.trades_df)}回")
    if not frac.trades_df.empty:
        sizes = frac.trades_df["shares"]
        check("端株の建玉が小数になっている", bool((sizes % 1 != 0).any()),
              f"例: {float(sizes.iloc[0]):.4f} 株")
        check("端株でも建玉比率は100%を超えない",
              float(frac.exposure.max()) <= 1.0001,
              f"最大 {frac.exposure.max()*100:.1f}%")


# ====================================================================

def main():
    print("=" * 62)
    print("  バックテストエンジン 検証テスト")
    print("=" * 62)
    for fn in (test_no_lookahead, test_execution_price, test_buy_and_hold,
               test_costs_reduce_return, test_stop_loss, test_cash_and_lots,
               test_indicators, test_metrics, test_random_data_gives_nothing,
               test_fractional_shares):
        fn()
    print("\n" + "=" * 62)
    print(f"  合格 {len(PASS)} 件 / 不合格 {len(FAIL)} 件")
    if FAIL:
        for f in FAIL:
            print(f"    ✗ {f}")
    print("=" * 62)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
複数の戦略仮説を、同じデータ・同じコスト・同じ判定基準で一度に比べる。

なぜこれを作るか:
  最初の戦略は「思いついたルールを書いて、検証を後回しにした」結果、
  3ヶ月分のペーパートレードを無駄にしかけた。順序を逆にする。
  先に仮説を並べ、全部を同じ土俵で検証し、生き残ったものだけを運用に回す。

比べ方の約束:
  - データも手数料も建玉数も全部同じ。違うのは売買ルールだけ
  - 2022-01-01 より前で組み立て、それ以降は一度も見ない（検証用に取っておく）
  - 「一番良かったもの」を選ばない。関門を全部通ったものだけを残す
  - 関門を通るものが無ければ「無い」と報告する。無理に1つ選ばない

使い方:
    python compare.py --json-out reports/compare_summary.json
    python compare.py --universe-limit 120 --max-combos 8   # 動作確認用
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import pandas as pd
import yaml

from qbt import data as D
from qbt import metrics as M
from qbt import universe as U
from qbt import validation as V
from qbt.engine import Backtest, Costs, Portfolio, Rules, clear_memo

HERE = os.path.dirname(os.path.abspath(__file__))


# ====================================================================
#  候補となる仮説
#
#  どれも「米国株で長年再現性が報告されてきた異常性」を素直に書いたもの。
#  思いつきは入れない。入れるなら、なぜそれが効くはずかを1行で書けること。
# ====================================================================

STRATEGIES = [
    {
        "key": "A_momentum",
        "name": "A. 12-1モメンタム",
        "why": "直近1ヶ月を除く12ヶ月の騰落率が高い銘柄は、その後も相対的に強い。"
               "株式で最も再現性が高いとされる異常性。直近1ヶ月を外すのは短期反転を避けるため。",
        "rules": {
            "entry": "close > sma(close, 200) and roc(shift(close, 21), 231) > {mom_min}",
            "exit": "cross_below(close, sma(close, {ma_exit}))",
            "rank": "roc(shift(close, 21), 231)",
            "rank_ascending": False,
            "max_hold_days": "{max_hold_days}",
        },
        "params": {"mom_min": 0.15, "ma_exit": 50, "max_hold_days": 60},
        "grid": {"mom_min": [0.0, 0.15, 0.30], "ma_exit": [20, 50, 100],
                 "max_hold_days": [20, 60, 120]},
    },
    {
        "key": "B_breakout",
        "name": "B. 52週高値ブレイク",
        "why": "年初来高値の更新は、投資家が過去の高値に引きずられて売り急ぐ分だけ"
               "反応が遅れる、という説明がされる。モメンタムの別の測り方でもある。",
        "rules": {
            "entry": "cross_above(close, highest(shift(close, 1), {look})) "
                     "and close > sma(close, 200)",
            "exit": "cross_below(close, sma(close, {ma_exit}))",
            "rank": "roc(shift(close, 21), 231)",
            "rank_ascending": False,
            "max_hold_days": "{max_hold_days}",
        },
        "params": {"look": 252, "ma_exit": 50, "max_hold_days": 60},
        "grid": {"look": [126, 252], "ma_exit": [20, 50, 100],
                 "max_hold_days": [20, 60, 120]},
    },
    {
        "key": "C_lowvol",
        "name": "C. 低ボラティリティ",
        "why": "値動きの穏やかな銘柄が、荒い銘柄より良い成績を出し続けている。"
               "理論上あってはならないが、数十年観測されている。",
        "rules": {
            "entry": "close > sma(close, 200) and hist_vol(close, {vol_n}) < {vol_max}",
            "exit": "cross_below(close, sma(close, {ma_exit}))",
            "rank": "hist_vol(close, {vol_n})",
            "rank_ascending": True,       # 低い順に買う
            "max_hold_days": "{max_hold_days}",
        },
        "params": {"vol_n": 60, "vol_max": 0.25, "ma_exit": 100, "max_hold_days": 60},
        "grid": {"vol_n": [60, 120], "vol_max": [0.20, 0.25, 0.35],
                 "ma_exit": [50, 100, 200], "max_hold_days": [60, 120]},
    },
    {
        "key": "D_current",
        "name": "D. 現行ルール（対照群）",
        "why": "すでに6項目中2つしか通らないと分かっている。これに勝てない案は採用しない。",
        "rules": {
            "entry": "close > sma(close, {ma_trend}) "
                     "and cross_above(rsi(close, {rsi_n}), {rsi_buy}) "
                     "and atr_pct(high, low, close, 14) < {max_vol}",
            "exit": "rsi(close, {rsi_n}) > {rsi_sell} "
                    "or cross_below(close, sma(close, {ma_exit}))",
            "rank": "roc(shift(close, 21), 231)",
            "rank_ascending": False,
            "max_hold_days": "{max_hold_days}",
        },
        "params": {"ma_trend": 200, "ma_exit": 20, "rsi_n": 14, "rsi_buy": 35,
                   "rsi_sell": 70, "max_vol": 0.06, "max_hold_days": 20},
        "grid": {"rsi_buy": [30, 35, 40], "ma_exit": [10, 20, 50],
                 "max_hold_days": [20, 60]},
    },
]

# 全戦略に共通してかける制約。ここを揃えないと比較にならない。
COMMON = {
    "universe_filter": "sma(volume, 20) * close > 20000000",   # 板の薄い銘柄を避ける
    "market_filter": "close > sma(close, 200)",                # 指数が下向きなら休む
    "stop_loss_pct": 0.08,
}


# ====================================================================

def build_rules(spec: dict, params: dict) -> Rules:
    r = spec["rules"]

    def sub(x):
        return x.format(**params) if isinstance(x, str) else x

    mh = r.get("max_hold_days")
    if isinstance(mh, str):
        mh = int(sub(mh))
    return Rules(
        entry=sub(r["entry"]), exit=sub(r.get("exit")),
        rank=sub(r.get("rank")), rank_ascending=r.get("rank_ascending", False),
        universe_filter=COMMON["universe_filter"],
        market_filter=COMMON["market_filter"],
        stop_loss_pct=COMMON["stop_loss_pct"],
        take_profit_pct=None, trail_stop_pct=None,
        max_hold_days=mh,
    )


def trim_grid(grid: dict, cap: int) -> dict:
    """組み合わせ数を上限まで削る。軸ごとに端から間引く"""
    def size(g):
        n = 1
        for v in g.values():
            n *= len(v)
        return n
    g = {k: list(v) for k, v in grid.items()}
    while size(g) > cap:
        k = max(g, key=lambda k: len(g[k]))
        if len(g[k]) <= 1:
            break
        g[k] = g[k][::2] if len(g[k]) > 2 else g[k][:1]
    return g


def evaluate(spec, data, pf, costs, benchmark, start, end, split, cap) -> dict:
    """1つの仮説を検証して、成績と合否を返す"""
    t0 = time.time()
    print(f"\n▶ {spec['name']}", flush=True)

    rules = build_rules(spec, spec["params"])
    res = V.split_test(data, rules, pf, costs, split, benchmark, start=start, end=end)
    sums = {k: M.summary(r.equity, r.trades_df, r.exposure, r.benchmark)
            for k, r in res.items()}
    for k, s in sums.items():
        print("    [%s] 年率%7.2f%%  シャープ%5.2f  最大DD%7.2f%%  取引%4d回  建玉%3.0f%%"
              % (k, s["年率リターン(CAGR)"] * 100, s["シャープレシオ"],
                 s["最大ドローダウン"] * 100, s["取引回数"],
                 (s.get("平均建玉比率") or 0) * 100), flush=True)

    # パラメータ感度。開発期間だけで調べる（検証期間は最後まで触らない）
    grid = trim_grid(spec["grid"], cap)
    n = 1
    for v in grid.values():
        n *= len(v)
    print(f"    パラメータ {n} 通りを調査中...", flush=True)
    keys = list(grid.keys())
    rows = []
    for values in itertools.product(*[grid[k] for k in keys]):
        p = dict(spec["params"])
        p.update(dict(zip(keys, values)))
        try:
            r = Backtest(data, build_rules(spec, p), pf, costs, benchmark).run(
                start=start, end=split)
            s = M.summary(r.equity, r.trades_df, r.exposure, r.benchmark)
            rows.append({**dict(zip(keys, values)),
                         **{k: s.get(k) for k in ("年率リターン(CAGR)", "シャープレシオ",
                                                  "最大ドローダウン", "取引回数", "勝率")}})
        except Exception as e:
            rows.append({**dict(zip(keys, values)), "エラー": str(e)[:60]})
    sweep = pd.DataFrame(rows)
    if "シャープレシオ" in sweep.columns:
        sweep = sweep.sort_values("シャープレシオ", ascending=False)
    sens = V.sensitivity_score(sweep)
    print(f"    → {sens.get('判定')}", flush=True)

    oos = sums.get("OOS(検証期間)", {})
    gates = []

    def gate(name, ok, detail):
        gates.append({"項目": name, "合格": bool(ok), "内容": detail})

    gate("検証期間で利益が出ている", (oos.get("年率リターン(CAGR)") or 0) > 0,
         f"年率 {(oos.get('年率リターン(CAGR)') or 0)*100:.2f}%")
    gate("シャープレシオ0.5以上", (oos.get("シャープレシオ") or 0) >= 0.5,
         f"{oos.get('シャープレシオ') or 0:.2f}")
    gate("取引30件以上", (oos.get("取引回数") or 0) >= 30,
         f"{oos.get('取引回数') or 0}件")
    gate("SPYより良い", (oos.get("超過年率") or -1) > 0,
         f"超過 {(oos.get('超過年率') or 0)*100:+.2f}%")
    gate("パラメータに頑健", (sens.get("中央値/最高") or 0) > 0.6,
         sens.get("判定", "不明"))

    # 建玉比率をそろえた比較も出す。65%しか投資していない戦略を
    # フルインベストの指数と並べるのは、そもそも不利な比べ方なので。
    expo = oos.get("平均建玉比率") or 0
    bm = oos.get("ベンチマーク年率") or 0
    adj = None
    if expo > 0.05:
        adj = (oos.get("年率リターン(CAGR)") or 0) - bm * expo
    passed = sum(1 for g in gates if g["合格"])
    print(f"    合否 {passed}/{len(gates)}"
          + (f"　建玉比率をそろえた超過 {adj*100:+.2f}%" if adj is not None else ""),
          flush=True)
    print(f"    （{time.time()-t0:.0f}秒）", flush=True)

    return {
        "key": spec["key"], "name": spec["name"], "why": spec["why"],
        "params": spec["params"],
        "results": {k: {kk: vv for kk, vv in s.items()
                        if not isinstance(vv, (dict, list))} for k, s in sums.items()},
        "sensitivity": sens,
        "sweep_n": int(len(sweep)),
        "sweep_top": sweep.head(6).to_dict("records"),
        "gates": gates, "passed": passed, "gate_total": len(gates),
        "exposure_adjusted_excess": adj,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--universe-limit", type=int, default=None)
    ap.add_argument("--max-combos", type=int, default=27)
    ap.add_argument("--start", default=None)
    ap.add_argument("--only", default=None, help="キーをカンマ区切りで指定すると絞れる")
    ap.add_argument("--json-out", default=os.path.join(HERE, "reports",
                                                       "compare_summary.json"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    dcfg = cfg["data"]
    start = args.start or dcfg["start"]
    end = dcfg["end"]
    split = (cfg.get("validation") or {}).get("split_date", "2022-01-01")

    if dcfg.get("symbols"):
        symbols = list(dcfg["symbols"])
    else:
        symbols = U.symbols(None if dcfg.get("universe") == "all" else dcfg.get("universe"))
    if args.universe_limit:
        symbols = symbols[:args.universe_limit]

    print(f"銘柄 {len(symbols)} / 期間 {start}〜{end} / 分割 {split}")
    print("データを準備中...", flush=True)
    data = D.load(symbols, start, end, source=dcfg.get("source", "yfinance"),
                  csv_dir=dcfg.get("csv_dir"), seed=dcfg.get("seed", 0))
    benchmark = None
    if dcfg.get("benchmark"):
        bm = D.load([dcfg["benchmark"]], start, end,
                    source=dcfg.get("source", "yfinance"),
                    csv_dir=dcfg.get("csv_dir"), seed=dcfg.get("seed", 0) + 991)
        benchmark = bm.get(dcfg["benchmark"])
        data.pop(dcfg["benchmark"], None)
    print(f"  {len(data)} 銘柄", flush=True)

    p = cfg.get("portfolio", {})
    pf = Portfolio(initial_cash=p.get("initial_cash", 2000),
                   max_positions=p.get("max_positions", 8),
                   position_pct=p.get("position_pct"),
                   lot_size=p.get("lot_size", 1),
                   allow_fractional=p.get("allow_fractional", True))
    c = cfg.get("costs", {})
    costs = Costs(commission_bps=c.get("commission_bps", 0.0),
                  slippage_bps=c.get("slippage_bps", 8.0),
                  min_commission=c.get("min_commission", 0.0))

    picked = STRATEGIES
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        picked = [s for s in STRATEGIES if s["key"] in want]

    clear_memo()
    out = []
    for spec in picked:
        try:
            out.append(evaluate(spec, data, pf, costs, benchmark,
                                start, end, split, args.max_combos))
        except Exception as e:
            import traceback
            traceback.print_exc(limit=3)
            out.append({"key": spec["key"], "name": spec["name"],
                        "エラー": str(e)[:200], "passed": 0, "gate_total": 5})

    out.sort(key=lambda r: (r.get("passed", 0),
                            (r.get("results", {}).get("OOS(検証期間)", {})
                             .get("シャープレシオ") or -9)), reverse=True)

    print("\n" + "=" * 70)
    print("  まとめ（検証期間の成績で並べたもの）")
    print("=" * 70)
    for r in out:
        if "エラー" in r:
            print(f"  {r['name']}: エラー {r['エラー']}")
            continue
        o = r["results"].get("OOS(検証期間)", {})
        print("  %-20s 合否%d/%d  年率%7.2f%%  シャープ%5.2f  超過%+7.2f%%  取引%4d"
              % (r["name"][:20], r["passed"], r["gate_total"],
                 (o.get("年率リターン(CAGR)") or 0) * 100, o.get("シャープレシオ") or 0,
                 (o.get("超過年率") or 0) * 100, o.get("取引回数") or 0))
    survivors = [r for r in out if r.get("passed", 0) == r.get("gate_total", 5)]
    print()
    if survivors:
        print(f"  全項目を通った仮説: {', '.join(r['name'] for r in survivors)}")
    else:
        best = out[0] if out else None
        print("  全項目を通った仮説はありません。")
        if best:
            print(f"  最も惜しいのは {best['name']}（{best['passed']}/{best['gate_total']}）")

    payload = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
        "symbols": len(data), "period": {"start": start, "end": end, "split": split},
        "max_combos": args.max_combos,
        "strategies": out,
        "survivors": [r["key"] for r in survivors],
    }
    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    print(f"\n  書き出し: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

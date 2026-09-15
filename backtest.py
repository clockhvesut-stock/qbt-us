#!/usr/bin/env python3
"""
バックテスト実行スクリプト。

使い方:
    python run.py                          # config.yaml を実行
    python run.py strategies/momentum.yaml # 別の戦略ファイルを実行
    python run.py --no-cache               # データを取り直す

コードを触る必要はない。YAMLファイルの数字と条件式を書き換えるだけ。
"""
from __future__ import annotations

import argparse
import os
import sys
import webbrowser

import pandas as pd
import yaml

from qbt import data as D
from qbt import metrics as M
from qbt import report as R
from qbt import universe as U
from qbt import validation as V
from qbt.engine import Backtest, Costs, Portfolio, Rules

HERE = os.path.dirname(os.path.abspath(__file__))


def resolve_symbols(dcfg: dict) -> list[str]:
    """
    銘柄リストを決める。以下の優先順で解決する。

      1. data.symbols に直接書かれていればそれを使う
      2. data.universe が指定されていれば universe.json から読む
         ("sp500" / "midcap" / "etf" / "all")
    """
    if dcfg.get("symbols"):
        return list(dcfg["symbols"])
    group = dcfg.get("universe")
    if not group:
        raise ValueError("config の data に symbols か universe のどちらかが必要です")
    limit = dcfg.get("universe_limit")
    syms = U.symbols(None if group == "all" else group, limit=limit)
    if not syms:
        raise ValueError(f"ユニバース '{group}' が空です。build_universe.py を実行してください")
    return syms


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="ルールベース戦略のバックテスト")
    ap.add_argument("config", nargs="?", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--no-cache", action="store_true", help="キャッシュを使わず再取得する")
    ap.add_argument("--out", default=None, help="レポートの出力先HTML")
    ap.add_argument("--open", action="store_true", help="完了後にブラウザで開く")
    ap.add_argument("--universe-limit", type=int, default=None,
                    help="銘柄数の上限。まず小さく試すときに使う")
    ap.add_argument("--start", default=None, help="開始日をconfigより優先する")
    ap.add_argument("--skip-walk-forward", action="store_true",
                    help="ウォークフォワード検証を飛ばす（一番時間がかかる）")
    ap.add_argument("--max-combos", type=int, default=None,
                    help="パラメータ探索の組み合わせ数の上限")
    ap.add_argument("--json-out", default=None,
                    help="結果の要約をJSONで書き出す先")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg.get("name", "無題の戦略")
    dcfg = cfg["data"]
    if args.start:
        dcfg["start"] = args.start
    if args.universe_limit:
        dcfg["universe_limit"] = args.universe_limit
    symbols = resolve_symbols(dcfg)
    if args.universe_limit:
        symbols = symbols[:args.universe_limit]

    print(f"\n▶ 戦略: {name}")
    print(f"  期間: {dcfg['start']} 〜 {dcfg['end']}　銘柄数: {len(symbols)}")

    # ---------------- データ取得 ----------------
    print("  データを準備中...")
    data = D.load(symbols, dcfg["start"], dcfg["end"],
                  source=dcfg.get("source", "yfinance"),
                  use_cache=not args.no_cache,
                  csv_dir=dcfg.get("csv_dir"), seed=dcfg.get("seed", 0))
    print(f"  {len(data)} 銘柄を読み込みました")

    benchmark = None
    if dcfg.get("benchmark"):
        bm = D.load([dcfg["benchmark"]], dcfg["start"], dcfg["end"],
                    source=dcfg.get("source", "yfinance"),
                    use_cache=not args.no_cache,
                    csv_dir=dcfg.get("csv_dir"), seed=dcfg.get("seed", 0) + 991)
        benchmark = bm.get(dcfg["benchmark"])
        data.pop(dcfg["benchmark"], None)

    # ---------------- 設定の組み立て ----------------
    rcfg = cfg["rules"]
    rules = Rules(
        entry=rcfg["entry"], exit=rcfg.get("exit"),
        rank=rcfg.get("rank"), rank_ascending=rcfg.get("rank_ascending", False),
        universe_filter=rcfg.get("universe_filter"),
        market_filter=rcfg.get("market_filter"),
        stop_loss_pct=rcfg.get("stop_loss_pct"),
        take_profit_pct=rcfg.get("take_profit_pct"),
        trail_stop_pct=rcfg.get("trail_stop_pct"),
        max_hold_days=rcfg.get("max_hold_days"),
    )
    pcfg = cfg.get("portfolio", {})
    pf = Portfolio(
        initial_cash=pcfg.get("initial_cash", 3_000_000),
        max_positions=pcfg.get("max_positions", 10),
        position_pct=pcfg.get("position_pct"),
        lot_size=pcfg.get("lot_size", 100),
        allow_fractional=pcfg.get("allow_fractional", False),
    )
    ccfg = cfg.get("costs", {})
    costs = Costs(
        commission_bps=ccfg.get("commission_bps", 5.0),
        slippage_bps=ccfg.get("slippage_bps", 10.0),
        min_commission=ccfg.get("min_commission", 0.0),
    )

    # パラメータのプレースホルダを埋める（{ma} など）
    params = cfg.get("params", {})
    if params:
        rules = V._fmt(rules, params)

    # ---------------- 実行 ----------------
    vcfg = cfg.get("validation", {})
    split = vcfg.get("split_date")
    results = {}

    if split:
        print(f"  期間分割検証: {split} で開発期間と検証期間に分けます")
        results = V.split_test(data, rules, pf, costs, split, benchmark,
                               start=dcfg["start"], end=dcfg["end"])
    else:
        bt = Backtest(data, rules, pf, costs, benchmark)
        results = {"全期間": bt.run(start=dcfg["start"], end=dcfg["end"], label="全期間")}

    for k, r in results.items():
        s = M.summary(r.equity, r.trades_df, r.exposure, r.benchmark)
        print(f"    [{k}] 年率 {s['年率リターン(CAGR)']*100:6.2f}%  "
              f"最大DD {s['最大ドローダウン']*100:6.2f}%  "
              f"シャープ {s['シャープレシオ']:5.2f}  "
              f"取引 {s['取引回数']:4d}回  勝率 {s['勝率']*100:5.1f}%")

    extras = {}

    # モンテカルロ
    last = list(results.values())[-1]
    all_trades = pd.concat([r.trades_df for r in results.values()], ignore_index=True)
    if vcfg.get("monte_carlo", True):
        # 取引1件の損益率をそのまま積み上げると、全資金を1銘柄に入れた場合の
        # ドローダウンになってしまう。実際は8分割なので、1枠ぶんに割り戻す。
        mc_trades = all_trades.copy()
        if len(mc_trades):
            w = pf.position_pct or (1.0 / max(pf.max_positions, 1))
            mc_trades["pnl_pct"] = mc_trades["pnl_pct"] * w
        extras["monte_carlo"] = V.monte_carlo(mc_trades)

    # パラメータ感度
    grid = cfg.get("param_grid")
    if grid and args.max_combos:
        # 組み合わせ数を上限まで削る。軸ごとに端から間引き、最初の値は必ず残す。
        import itertools as _it

        def _size(g):
            n = 1
            for v in g.values():
                n *= len(v)
            return n
        grid = {k: list(v) for k, v in grid.items()}
        while _size(grid) > args.max_combos:
            k = max(grid, key=lambda k: len(grid[k]))
            if len(grid[k]) <= 1:
                break
            grid[k] = grid[k][::2] if len(grid[k]) > 2 else grid[k][:1]
        print(f"  組み合わせを {_size(grid)} 通りに絞りました")
    if grid:
        print(f"  パラメータ感度を調査中（{len(grid)}軸）...")
        base_rules = Rules(
            entry=rcfg["entry"], exit=rcfg.get("exit"), rank=rcfg.get("rank"),
            rank_ascending=rcfg.get("rank_ascending", False),
            universe_filter=rcfg.get("universe_filter"),
            market_filter=rcfg.get("market_filter"),
            stop_loss_pct=rcfg.get("stop_loss_pct"),
            take_profit_pct=rcfg.get("take_profit_pct"),
            trail_stop_pct=rcfg.get("trail_stop_pct"),
            max_hold_days=rcfg.get("max_hold_days"),
        )
        sweep = V.param_sweep(data, base_rules, pf, costs, grid, benchmark,
                              start=dcfg["start"], end=split or dcfg["end"])
        extras["sweep"] = sweep
        extras["sensitivity"] = V.sensitivity_score(sweep)
        print(f"    → {extras['sensitivity'].get('判定')}")

        if vcfg.get("walk_forward") and not args.skip_walk_forward:
            wfc = vcfg["walk_forward"]
            print("  ウォークフォワード検証を実行中（時間がかかります）...")
            extras["walk_forward"] = V.walk_forward(
                data, base_rules, pf, costs, grid,
                start=dcfg["start"], end=dcfg["end"],
                train_years=wfc.get("train_years", 3),
                test_years=wfc.get("test_years", 1),
                benchmark=benchmark)
            deg = extras["walk_forward"].get("degradation")
            if deg and deg.get("劣化率") is not None:
                print(f"    → 学習 {deg['学習平均']:.2f} / 検証 {deg['検証平均']:.2f} "
                      f"（劣化率 {deg['劣化率']*100:.0f}%）")

    # ---------------- 合否の判定 ----------------
    # 「良さそう」で終わらせないために、先に決めた5つの関門を機械的に当てる。
    # ここを通らない限り実弾には進まない。
    summaries = {k: M.summary(r.equity, r.trades_df, r.exposure, r.benchmark)
                 for k, r in results.items()}
    oos = summaries.get("OOS(検証期間)") or list(summaries.values())[-1]
    gates = []

    def gate(name, ok, detail):
        gates.append({"項目": name, "合格": bool(ok), "内容": detail})

    gate("検証期間で利益が出ている",
         oos.get("年率リターン(CAGR)", 0) > 0,
         f"年率 {oos.get('年率リターン(CAGR)', 0)*100:.2f}%")
    gate("検証期間のシャープレシオが0.5以上",
         (oos.get("シャープレシオ") or 0) >= 0.5,
         f"シャープ {oos.get('シャープレシオ', 0):.2f}")
    gate("取引数が30件以上ある",
         (oos.get("取引回数") or 0) >= 30,
         f"{oos.get('取引回数', 0)}件")
    # ここが本丸。指数を買って放置するより良くないなら、手間をかける意味がない。
    if oos.get("超過年率") is not None:
        gate("SPYを買って放置するより良い",
             oos["超過年率"] > 0,
             f"超過年率 {oos['超過年率']*100:+.2f}%"
             f"（戦略 {oos.get('年率リターン(CAGR)',0)*100:.2f}% / "
             f"SPY {oos.get('ベンチマーク年率',0)*100:.2f}%）")
    sens = extras.get("sensitivity") or {}
    gate("パラメータを動かしても崩れない",
         (sens.get("中央値/最高") or 0) > 0.6,
         sens.get("判定", "未実施"))
    deg = (extras.get("walk_forward") or {}).get("degradation")
    if deg and deg.get("劣化率") is not None:
        gate("ウォークフォワードの劣化が5割未満",
             deg["劣化率"] < 0.5,
             f"学習 {deg['学習平均']:.2f} → 検証 {deg['検証平均']:.2f}"
             f"（劣化 {deg['劣化率']*100:.0f}%）")
    else:
        gate("ウォークフォワードの劣化が5割未満", False, "未実施")

    passed = sum(1 for g in gates if g["合格"])
    print("\n  ---- 合否 ----")
    for g in gates:
        print(f"    {'合格' if g['合格'] else '不合格'}  {g['項目']}　{g['内容']}")
    print(f"    → {passed}/{len(gates)} 項目")

    if args.json_out:
        import json as _json
        sweep_df = extras.get("sweep")
        payload = {
            "name": name,
            "generated_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
            "symbols": len(data),
            "period": {"start": dcfg["start"], "end": dcfg["end"],
                       "split": split},
            "results": {k: {kk: (None if isinstance(vv, float) and pd.isna(vv) else vv)
                            for kk, vv in s.items() if not isinstance(vv, (dict, list))}
                        for k, s in summaries.items()},
            "monte_carlo": extras.get("monte_carlo"),
            "sensitivity": sens,
            "sweep_top": (sweep_df.head(10).to_dict("records")
                          if sweep_df is not None and len(sweep_df) else []),
            "sweep_n": int(len(sweep_df)) if sweep_df is not None else 0,
            "walk_forward": {
                "degradation": deg,
                "folds": ((extras.get("walk_forward") or {}).get("folds").to_dict("records")
                          if (extras.get("walk_forward") or {}).get("folds") is not None
                          and len((extras["walk_forward"]["folds"])) else []),
            },
            "gates": gates,
            "passed": passed,
            "gate_total": len(gates),
        }
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            _json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
        print(f"  結果の要約: {args.json_out}")

    # ---------------- レポート ----------------
    out = args.out or os.path.join(HERE, "reports", f"{name}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(args.config, encoding="utf-8") as f:
        cfg_text = f.read()
    R.build(results, out, title=f"{name} — バックテスト結果",
            config_text=cfg_text, extras=extras)
    print(f"\n✓ レポートを書き出しました: {out}\n")

    # 取引履歴もCSVで残す
    csv_out = out.replace(".html", "_trades.csv")
    if not all_trades.empty:
        all_trades.to_csv(csv_out, index=False, encoding="utf-8-sig")
        print(f"  取引履歴: {csv_out}")

    if args.open:
        webbrowser.open("file://" + os.path.abspath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg.get("name", "無題の戦略")
    dcfg = cfg["data"]
    symbols = resolve_symbols(dcfg)

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
        extras["monte_carlo"] = V.monte_carlo(all_trades)

    # パラメータ感度
    grid = cfg.get("param_grid")
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

        if vcfg.get("walk_forward"):
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

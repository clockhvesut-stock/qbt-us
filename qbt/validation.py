"""
過学習を検出するための検証ツール。

バックテストで良い数字を出すのは簡単。パラメータをいじれば必ず綺麗な曲線が作れる。
問題は「その数字が期間外でも再現するか」だけ。ここのモジュールはそれを潰しにいく。

  1. 期間分割 (In-Sample / Out-of-Sample)
     前半でルールを作り、後半では一切触らずに検証する。最低限これはやる。

  2. ウォークフォワード
     「直近N年で最適化 → 次のM年で実運用」を時間をずらしながら繰り返す。
     実運用の手順をそのまま再現するので、期間分割より現実に近い。

  3. パラメータ感度
     最適値の周辺でも成績が保たれるか。ピンポイントでしか勝てないルールは偶然。

  4. モンテカルロ
     取引の順序を入れ替えて、ドローダウンの分布を見る。
     「運が悪い並び方」だとどこまで落ちるかを知っておく。
"""
from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd

from .engine import Backtest, Costs, Portfolio, Rules, Result
from .metrics import max_drawdown, summary


def _fmt(rules: Rules, params: dict) -> Rules:
    """ルール式の {param} を実際の値に置き換える"""
    def sub(x):
        return x.format(**params) if isinstance(x, str) else x
    return replace(
        rules,
        entry=sub(rules.entry), exit=sub(rules.exit), rank=sub(rules.rank),
        universe_filter=sub(rules.universe_filter), market_filter=sub(rules.market_filter),
        stop_loss_pct=params.get("stop_loss_pct", rules.stop_loss_pct),
        take_profit_pct=params.get("take_profit_pct", rules.take_profit_pct),
        max_hold_days=params.get("max_hold_days", rules.max_hold_days),
    )


# ------------------------------------------------------------------ 期間分割

def split_test(data, rules: Rules, pf: Portfolio, costs: Costs,
               split_date: str, benchmark=None,
               start: str | None = None, end: str | None = None) -> dict[str, Result]:
    """split_date より前を開発期間、以後を検証期間として2回走らせる"""
    bt = Backtest(data, rules, pf, costs, benchmark)
    return {
        "IS(開発期間)": bt.run(start=start, end=split_date, label="IS(開発期間)"),
        "OOS(検証期間)": bt.run(start=split_date, end=end, label="OOS(検証期間)"),
    }


# ------------------------------------------------------------------ パラメータ探索

def param_sweep(data, rules: Rules, pf: Portfolio, costs: Costs,
                grid: dict[str, list], benchmark=None,
                start: str | None = None, end: str | None = None,
                metric: str = "シャープレシオ") -> pd.DataFrame:
    """
    パラメータの全組み合わせでバックテストし、成績一覧を返す。

    注意: ここで一番良かった組み合わせを「発見」と呼んではいけない。
    500通り試せば、コイン投げでも1つは素晴らしい成績が出る。
    この表は「良い値を選ぶため」ではなく「成績が安定する範囲を見るため」に使う。
    """
    keys = list(grid.keys())
    rows = []
    combos = list(itertools.product(*[grid[k] for k in keys]))
    for n, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        try:
            bt = Backtest(data, _fmt(rules, params), pf, costs, benchmark)
            res = bt.run(start=start, end=end)
            s = summary(res.equity, res.trades_df, res.exposure, res.benchmark)
        except Exception as e:  # 極端なパラメータで取引ゼロ等
            s = {"エラー": str(e)[:60]}
        rows.append({**params, **{k: s.get(k) for k in
                    ("年率リターン(CAGR)", "シャープレシオ", "最大ドローダウン",
                     "取引回数", "勝率", "プロフィットファクター", "t値")}})
        if n % 20 == 0:
            print(f"    パラメータ探索 {n}/{len(combos)} 件完了")
    df = pd.DataFrame(rows)
    return df.sort_values(metric, ascending=False) if metric in df.columns else df


def sensitivity_score(sweep: pd.DataFrame, metric: str = "シャープレシオ") -> dict:
    """
    パラメータ感度の評価。
    最高値だけが飛び抜けていて周りが悪いなら、それは山ではなく針。実運用では踏み抜く。
    """
    v = sweep[metric].dropna()
    if len(v) < 5:
        return {"判定": "サンプル不足"}
    best, med = float(v.max()), float(v.median())
    top_q = float(v.quantile(0.75))
    ratio = med / best if best > 1e-9 else 0.0
    if best <= 0:
        verdict = "全滅。この戦略の骨格自体が機能していない"
    elif ratio > 0.6:
        verdict = "良好。広いパラメータ範囲で機能している（頑健）"
    elif ratio > 0.3:
        verdict = "やや不安。最適値付近に依存している"
    else:
        verdict = "危険。特定のパラメータでしか勝てない＝カーブフィッティングの疑い"
    return {"最高": best, "中央値": med, "上位25%": top_q,
            "中央値/最高": ratio, "判定": verdict}


# ------------------------------------------------------------------ ウォークフォワード

def walk_forward(data, rules: Rules, pf: Portfolio, costs: Costs,
                 grid: dict[str, list], start: str, end: str,
                 train_years: float = 3.0, test_years: float = 1.0,
                 benchmark=None, metric: str = "シャープレシオ") -> dict:
    """
    「学習期間で最適化 → 直後の期間で運用」を時間をずらして繰り返す。
    返り値の equity は、各テスト期間の成績だけをつないだ資産曲線。
    これが右肩上がりでないなら、その戦略は実運用に耐えない。
    """
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    folds, oos_curves = [], []
    train_start = t0
    while True:
        train_end = train_start + pd.DateOffset(days=int(train_years * 365.25))
        test_end = train_end + pd.DateOffset(days=int(test_years * 365.25))
        if train_end >= t1:
            break
        test_end = min(test_end, t1)
        # 検証期間が短すぎるフォールドは捨てる。
        # 数日しかない期間のシャープレシオは極端な値になり、平均を壊す。
        if (test_end - train_end).days < test_years * 365.25 * 0.5:
            break

        best_score, best_params = -np.inf, None
        for values in combos:
            params = dict(zip(keys, values))
            try:
                bt = Backtest(data, _fmt(rules, params), pf, costs, benchmark)
                r = bt.run(start=str(train_start.date()), end=str(train_end.date()))
                s = summary(r.equity, r.trades_df)
                # 取引が少なすぎるものは採用しない（偶然の1発を掴まない）
                if s["取引回数"] < 5:
                    continue
                score = s[metric]
            except Exception:
                continue
            if score > best_score:
                best_score, best_params = score, params

        if best_params is None:
            train_start = train_start + pd.DateOffset(days=int(test_years * 365.25))
            continue

        bt = Backtest(data, _fmt(rules, best_params), pf, costs, benchmark)
        r_oos = bt.run(start=str(train_end.date()), end=str(test_end.date()))
        s_oos = summary(r_oos.equity, r_oos.trades_df)
        folds.append({
            "学習期間": f"{train_start.date()}〜{train_end.date()}",
            "検証期間": f"{train_end.date()}〜{test_end.date()}",
            "採用パラメータ": best_params,
            f"学習{metric}": round(best_score, 3),
            f"検証{metric}": round(s_oos[metric], 3),
            "検証リターン": s_oos["累積リターン"],
            "検証最大DD": s_oos["最大ドローダウン"],
            "取引回数": s_oos["取引回数"],
        })
        oos_curves.append(r_oos.equity)
        train_start = train_start + pd.DateOffset(days=int(test_years * 365.25))

    # 各フォールドの検証期間だけを連結して1本の資産曲線にする
    stitched = None
    if oos_curves:
        parts, base = [], pf.initial_cash
        for eq in oos_curves:
            if len(eq) < 2:
                continue
            scaled = eq / eq.iloc[0] * base
            parts.append(scaled)
            base = float(scaled.iloc[-1])
        if parts:
            stitched = pd.concat(parts)
            stitched = stitched[~stitched.index.duplicated(keep="first")].sort_index()

    fold_df = pd.DataFrame(folds)
    degradation = None
    if len(fold_df):
        tr = fold_df[f"学習{metric}"].mean()
        te = fold_df[f"検証{metric}"].mean()
        degradation = {"学習平均": tr, "検証平均": te,
                       "劣化率": 1 - (te / tr) if tr > 1e-9 else None}
    return {"folds": fold_df, "equity": stitched, "degradation": degradation}


# ------------------------------------------------------------------ モンテカルロ

def monte_carlo(trades: pd.DataFrame, n_sims: int = 2000, seed: int = 42) -> dict:
    """
    取引の順番をシャッフルして資産曲線を作り直し、ドローダウンの分布を見る。
    実際に起きた最大DDは「たまたま運が良かった並び」かもしれない。
    運用前に「95%の確率でここまでは食らう」という数字を知っておく。
    """
    if len(trades) < 10:
        return {"判定": "取引数が不足（10件以上必要）"}
    r = trades.sort_values("exit_date")["pnl_pct"].values
    rng = np.random.default_rng(seed)
    mdds, finals = [], []
    for _ in range(n_sims):
        shuffled = rng.permutation(r)
        eq = np.cumprod(1 + shuffled)
        mdds.append(float((eq / np.maximum.accumulate(eq) - 1).min()))
        finals.append(float(eq[-1] - 1))
    mdds, finals = np.array(mdds), np.array(finals)
    return {
        "実績最大DD": float((np.cumprod(1 + r) / np.maximum.accumulate(np.cumprod(1 + r)) - 1).min()),
        "想定最大DD(中央値)": float(np.median(mdds)),
        "想定最大DD(95%タイル)": float(np.percentile(mdds, 5)),
        "想定最大DD(最悪)": float(mdds.min()),
        "累積リターンが負になる確率": float((finals < 0).mean()),
    }

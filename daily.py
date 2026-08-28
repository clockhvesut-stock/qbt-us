#!/usr/bin/env python3
"""
毎日のデータ収集パイプライン。GitHub Actions 上で動く。

やること:
  1. 相場環境（マクロ指標・市場の内部状況）を測る
  2. ユニバース全銘柄の価格を取得し、検証済みルールでシグナルを出す（トラックA）
  3. 候補銘柄まわりのニュースとSEC提出書類を集める（トラックB用の材料）
  4. 保有中のポジションを評価し、手仕舞い条件を判定する
  5. すべてを1つのJSONに書き出す

判断はここではしない。事実を集めるだけ。
ニュースの解釈と最終的な売買判断はAI側（Claude）が担当する。
分業をはっきりさせておくことで、後から「なぜそう判断したか」を追跡できる。

使い方:
    python daily.py                  # 通常実行
    python daily.py --dry-run        # ファイルを書かずに内容だけ表示
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qbt import feeds, universe  # noqa: E402
from qbt.engine import build_signal_frame, eval_expr  # noqa: E402
from qbt.indicators import atr_pct, roc, rsi, sma, zscore  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "reports")
STATE = os.path.join(HERE, "state")


def log(msg: str):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def safe(fn, label: str, default=None):
    """1つの取得が失敗しても全体を止めない。何が失敗したかは記録に残す"""
    try:
        return fn()
    except Exception as e:
        log(f"  [失敗] {label}: {e}")
        traceback.print_exc(limit=2)
        return default


# ==================================================================== 特徴量

def compute_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    全銘柄の現在の状態を1つの表にまとめる。
    ここに並ぶ数字が、ルール判断とAI判断の共通の土台になる。
    """
    rows = []
    for sym, df in data.items():
        if len(df) < 220:
            continue
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        try:
            last = float(c.iloc[-1])
            rows.append({
                "symbol": sym,
                "price": round(last, 2),
                "chg_1d": round(float(c.iloc[-1] / c.iloc[-2] - 1), 4),
                "chg_5d": round(float(c.iloc[-1] / c.iloc[-6] - 1), 4),
                "chg_21d": round(float(c.iloc[-1] / c.iloc[-22] - 1), 4),
                # 直近1ヶ月を除いた12ヶ月モメンタム（短期反転の影響を除く定番の作り方）
                "mom_12_1": round(float(c.iloc[-21] / c.iloc[-252] - 1), 4),
                "rsi14": round(float(rsi(c, 14).iloc[-1]), 1),
                "rsi3": round(float(rsi(c, 3).iloc[-1]), 1),
                "zscore10": round(float(zscore(c, 10).iloc[-1]), 2),
                "vs_ma50": round(float(last / sma(c, 50).iloc[-1] - 1), 4),
                "vs_ma200": round(float(last / sma(c, 200).iloc[-1] - 1), 4),
                "atr_pct": round(float(atr_pct(h, l, c, 14).iloc[-1]), 4),
                "vol_ratio": round(float(v.iloc[-1] / v.tail(20).mean()), 2),
                "dollar_vol": int(float((c * v).tail(20).median())),
                "from_52w_high": round(float(last / c.tail(252).max() - 1), 4),
                "from_52w_low": round(float(last / c.tail(252).min() - 1), 4),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def compute_regime(macro: dict, breadth: dict) -> dict:
    """
    相場の「局面」を判定する。
    どの戦略が効くかは局面で変わるので、これを毎日記録しておくと後で効いてくる。
    """
    spy = macro.get("^GSPC", {})
    vix = macro.get("^VIX", {})
    tnx = macro.get("^TNX", {})
    trend_up = (spy.get("vs_ma200") or 0) > 0
    vix_last = vix.get("last") or 0
    broad = (breadth.get("above_ma200_pct") or 0) > 0.5

    if trend_up and broad and vix_last < 20:
        name, note = "順行", "トレンドフォロー・モメンタムが効きやすい局面"
    elif trend_up and not broad:
        name, note = "選別", "指数は上だが中身が伴っていない。銘柄選択がより重要"
    elif not trend_up and vix_last > 28:
        name, note = "混乱", "新規建ては見送り推奨。逆張りも落ちるナイフになりやすい"
    elif not trend_up:
        name, note = "調整", "下降トレンド。買い戦略は成績が落ちる前提で"
    else:
        name, note = "中立", "明確な方向感がない"

    return {
        "regime": name,
        "note": note,
        "spy_above_ma200": trend_up,
        "breadth_healthy": broad,
        "vix": vix_last,
        "vix_percentile_1y": vix.get("pctile_1y"),
        "rate_10y": tnx.get("last"),
        "rate_10y_chg_20d": tnx.get("chg_20d"),
    }


# ==================================================================== ポジション

def load_positions() -> list[dict]:
    path = os.path.join(STATE, "positions.json")
    if not os.path.exists(path):
        return []
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return []


def evaluate_positions(positions: list[dict], data: dict[str, pd.DataFrame],
                       cfg: dict) -> list[dict]:
    """保有中の銘柄について、手仕舞い条件に当たっていないかを見る"""
    rules = cfg.get("rules", {})
    stop = rules.get("stop_loss_pct")
    take = rules.get("take_profit_pct")
    maxhold = rules.get("max_hold_days")
    out = []
    for p in positions:
        sym = p.get("symbol")
        if sym not in data:
            out.append({**p, "status": "価格取得できず"})
            continue
        c = data[sym]["close"]
        px = float(c.iloc[-1])
        ep = float(p.get("entry_price", px))
        pnl = px / ep - 1
        held = int(p.get("bars_held", 0)) + 1

        flags = []
        if stop and pnl <= -stop:
            flags.append(f"損切りライン(-{stop*100:.0f}%)に到達")
        if take and pnl >= take:
            flags.append(f"利確ライン(+{take*100:.0f}%)に到達")
        if maxhold and held >= maxhold:
            flags.append(f"保有{held}営業日で期間満了")
        exit_expr = rules.get("exit")
        if exit_expr:
            try:
                s = eval_expr(exit_expr, data[sym])
                if bool(s.iloc[-1]):
                    flags.append("手仕舞いシグナル点灯")
            except Exception:
                pass

        out.append({
            "symbol": sym,
            "shares": p.get("shares"),
            "entry_price": round(ep, 2),
            "entry_date": p.get("entry_date"),
            "price": round(px, 2),
            "pnl_pct": round(pnl, 4),
            "bars_held": held,
            "stop_price": round(ep * (1 - stop), 2) if stop else None,
            "exit_flags": flags,
            "action": "手仕舞い" if flags else "継続保有",
        })
    return out


# ==================================================================== 本体

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-symbols", type=int, default=None,
                    help="デバッグ用。処理する銘柄数を制限する")
    ap.add_argument("--news-symbols", type=int, default=25,
                    help="ニュースを集める銘柄数（上位候補のみ）")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    today = datetime.now(timezone.utc)
    os.makedirs(REPORTS, exist_ok=True)
    os.makedirs(STATE, exist_ok=True)

    out: dict = {
        "generated_at": today.isoformat(timespec="seconds"),
        "date": today.strftime("%Y-%m-%d"),
        "strategy": cfg.get("name", ""),
        "errors": [],
    }

    # ---------- 1. ユニバース ----------
    log("ユニバースを読み込み中...")
    uni = safe(lambda: universe.load(), "ユニバース読み込み")
    if uni is None or uni.empty:
        log("  ユニバース未作成。S&P500とETFで代用します")
        uni = pd.concat([universe.sp500(), universe.etf_universe()], ignore_index=True)
    syms = uni["symbol"].tolist()
    if args.max_symbols:
        syms = syms[:args.max_symbols]
    ciks = {r["symbol"]: str(r.get("cik") or "") for _, r in uni.iterrows()}
    log(f"  {len(syms)} 銘柄")

    # ---------- 2. 価格 ----------
    log("価格データを取得中...")
    import yfinance as yf

    def _prices():
        raw = yf.download(syms, period="2y", auto_adjust=True, progress=False,
                          group_by="column", threads=True)
        d = {}
        for s in syms:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    df = pd.DataFrame({
                        "open": raw[("Open", s)], "high": raw[("High", s)],
                        "low": raw[("Low", s)], "close": raw[("Close", s)],
                        "volume": raw[("Volume", s)]})
                else:
                    df = raw.rename(columns=str.lower)[
                        ["open", "high", "low", "close", "volume"]]
            except KeyError:
                continue
            df = df.dropna(subset=["close"])
            if len(df) < 220:
                continue
            df.index = pd.to_datetime(df.index).tz_localize(None)
            d[s] = df.sort_index()
        return d

    data = safe(_prices, "価格取得", {}) or {}
    log(f"  {len(data)} 銘柄の価格を取得")
    if not data:
        out["errors"].append("価格データを1銘柄も取得できませんでした")
        _write(out, args.dry_run)
        return 1
    out["data_date"] = str(max(df.index[-1] for df in data.values()).date())

    # ---------- 3. 相場環境 ----------
    log("相場環境を測定中...")
    macro = safe(feeds.macro_snapshot, "マクロ指標", {}) or {}
    closes = pd.DataFrame({s: d["close"] for s, d in data.items()})
    breadth = safe(lambda: feeds.market_breadth(closes), "市場の内部状況", {}) or {}
    out["macro"] = macro
    out["breadth"] = breadth
    out["regime"] = compute_regime(macro, breadth)
    log(f"  局面: {out['regime']['regime']}（{out['regime']['note']}）")

    # ---------- 4. 特徴量とシグナル（トラックA: ルールベース） ----------
    log("特徴量を計算中...")
    feat = compute_features(data)
    out["universe_size"] = int(len(feat))

    rules = cfg.get("rules", {})
    params = cfg.get("params", {})

    def _fmt(x):
        return x.format(**params) if isinstance(x, str) and params else x

    cal = pd.DatetimeIndex(sorted(set().union(*[df.index for df in data.values()])))
    signals = []

    def _signals():
        entry = build_signal_frame(_fmt(rules["entry"]), data, cal)
        uni_ok = (build_signal_frame(_fmt(rules["universe_filter"]), data, cal)
                  if rules.get("universe_filter") else None)
        rank = (build_signal_frame(_fmt(rules["rank"]), data, cal, boolean=False)
                if rules.get("rank") else None)
        row = entry.iloc[-1]
        cands = [s for s in row.index if bool(row.get(s, False))]
        if uni_ok is not None:
            cands = [s for s in cands if bool(uni_ok.iloc[-1].get(s, False))]
        if rank is not None:
            rv = rank.iloc[-1]
            cands = sorted(cands, key=lambda s: float(rv.get(s, -1e18))
                           if pd.notna(rv.get(s)) else -1e18, reverse=True)
        res = []
        for i, s in enumerate(cands[:40], 1):
            f = feat[feat["symbol"] == s]
            res.append({
                "rank": i, "symbol": s,
                "rank_score": (round(float(rank.iloc[-1].get(s)), 4)
                               if rank is not None and pd.notna(rank.iloc[-1].get(s)) else None),
                **({} if f.empty else {k: v for k, v in f.iloc[0].to_dict().items()
                                       if k != "symbol"}),
            })
        return res

    signals = safe(_signals, "シグナル計算", []) or []
    out["track_a_signals"] = signals
    log(f"  ルールベースの候補: {len(signals)} 銘柄")

    # 相場フィルタ
    if rules.get("market_filter") and "^GSPC" in macro:
        out["market_filter_pass"] = bool(macro["^GSPC"].get("vs_ma200", 0) > 0)

    # ---------- 5. ニュースとSEC提出書類（トラックB用の材料） ----------
    watch = [s["symbol"] for s in signals[:args.news_symbols]]
    held = [p.get("symbol") for p in load_positions()]
    watch = list(dict.fromkeys(watch + held))[:args.news_symbols + 10]

    log(f"ニュースを収集中（{len(watch)}銘柄）...")
    news = safe(lambda: feeds.alpaca_news(watch), "Alpacaニュース", []) or []
    if not news:
        news = safe(lambda: feeds.yahoo_news(watch), "Yahooニュース", []) or []
    out["news"] = news[:80]
    log(f"  {len(news)} 件")

    log("SEC提出書類を確認中...")
    filings = safe(lambda: feeds.sec_recent_filings(
        {s: ciks.get(s, "") for s in watch if ciks.get(s)}), "SEC EDGAR", []) or []
    out["filings"] = filings[:40]
    log(f"  {len(filings)} 件")

    out["earnings"] = safe(lambda: feeds.earnings_calendar(watch), "決算予定", []) or []

    # ---------- 6. 保有ポジション ----------
    positions = load_positions()
    out["positions"] = evaluate_positions(positions, data, cfg)
    out["cash"] = _read_json(os.path.join(STATE, "account.json"), {}).get(
        "cash", cfg.get("portfolio", {}).get("initial_cash", 2000))

    # ---------- 7. セクター動向 ----------
    def _sectors():
        sec_map = uni.set_index("symbol")["sector"].to_dict()
        f = feat.copy()
        f["sector"] = f["symbol"].map(sec_map)
        g = f[f["sector"].notna() & (f["sector"] != "")].groupby("sector").agg(
            銘柄数=("symbol", "size"), 平均1日=("chg_1d", "mean"),
            平均5日=("chg_5d", "mean"), 平均21日=("chg_21d", "mean"),
            MA200超え比率=("vs_ma200", lambda x: float((x > 0).mean())))
        return json.loads(g.round(4).reset_index().to_json(orient="records"))

    out["sectors"] = safe(_sectors, "セクター集計", []) or []

    _write(out, args.dry_run)
    log("完了")
    return 0


def _read_json(path: str, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def _write(out: dict, dry: bool):
    if dry:
        print(json.dumps(out, ensure_ascii=False, indent=1)[:4000])
        return
    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, f"{out['date']}_raw.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(REPORTS, "latest_raw.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"書き出し: {path}（{os.path.getsize(path)/1024:.0f}KB）")


if __name__ == "__main__":
    sys.exit(main())

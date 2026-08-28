"""
売買対象にする銘柄の母集団（ユニバース）を組み立てる。

ユニバースの選び方は戦略そのものと同じくらい成績を左右する。
だからここでは1つに決め打ちせず、複数の候補を作って検証で比較できるようにしてある。

  sp500      : S&P500の大型株。流動性が最高。ただし最も多くの参加者に分析されている
  midcap     : S&P500に入っていない中型株。アナリストのカバーが薄く、情報の非効率が残りやすい
  etf        : 主要ETFとセクターETF。個別企業リスクを避け、資金の流れそのものを取りにいく
  liquid_all : 流動性フィルタを通した全上場銘柄

注意: ここで作るリストは「今日時点で上場している銘柄」でしかない。
過去の検証に使うと、倒産・上場廃止した銘柄が最初から除外されるため、
成績が実態より必ず良く出る（生存者バイアス）。
本格的に信じられる数字が欲しくなったら、Point-in-Timeのユニバースを持つ
有料データ（Sharadar等）に切り替えること。
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# 主要ETF。セクターETFはSPDRの11セクターを網羅している。
ETFS = {
    "SPY": "S&P500", "QQQ": "ナスダック100", "IWM": "ラッセル2000", "DIA": "ダウ30",
    "XLK": "テクノロジー", "XLF": "金融", "XLV": "ヘルスケア", "XLE": "エネルギー",
    "XLI": "資本財", "XLY": "一般消費財", "XLP": "生活必需品", "XLU": "公益",
    "XLB": "素材", "XLRE": "不動産", "XLC": "通信サービス",
    "TLT": "米長期国債20年+", "IEF": "米中期国債7-10年", "HYG": "ハイイールド債",
    "GLD": "金", "SLV": "銀", "USO": "原油", "UNG": "天然ガス",
    "EEM": "新興国株", "EFA": "先進国株(除く米)", "FXI": "中国株", "EWJ": "日本株",
    "UUP": "ドル指数", "VIXY": "VIX短期先物",
}


def sp500() -> pd.DataFrame:
    """S&P500の構成銘柄。GICSセクターとSECのCIK番号つき"""
    path = os.path.join(DATA_DIR, "sp500.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    df = pd.DataFrame([{
        "symbol": r["Symbol"].replace(".", "-"),   # BRK.B → BRK-B (yfinance形式)
        "name": r["Security"],
        "sector": r["GICS Sector"],
        "industry": r["GICS Sub-Industry"],
        "cik": str(r["CIK"]).zfill(10),            # SEC EDGAR用の10桁ゼロ埋め
        "group": "sp500",
    } for r in rows])
    return df


def all_listed() -> pd.DataFrame:
    """NASDAQとNYSEの全上場銘柄コード"""
    out = []
    for exch in ("nasdaq", "nyse"):
        path = os.path.join(DATA_DIR, f"{exch}_tickers.json")
        if not os.path.exists(path):
            continue
        for t in json.load(open(path, encoding="utf-8")):
            t = str(t).strip().upper()
            # 優先株・ワラント・ユニット等を除外（末尾に記号がつく）
            if not t or not t.replace("-", "").isalpha() or len(t) > 5:
                continue
            out.append({"symbol": t, "name": "", "sector": "", "industry": "",
                        "cik": "", "group": exch})
    df = pd.DataFrame(out).drop_duplicates(subset=["symbol"])
    return df.reset_index(drop=True)


def etf_universe() -> pd.DataFrame:
    return pd.DataFrame([{"symbol": k, "name": v, "sector": "ETF",
                          "industry": "ETF", "cik": "", "group": "etf"}
                         for k, v in ETFS.items()])


def midcap_candidates() -> pd.DataFrame:
    """全上場銘柄からS&P500を除いたもの。ここに流動性フィルタをかけて中型株を得る"""
    sp = set(sp500()["symbol"])
    df = all_listed()
    df = df[~df["symbol"].isin(sp)].copy()
    df["group"] = "midcap"
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- 流動性フィルタ

def apply_liquidity_filter(
    symbols: list[str],
    min_dollar_volume: float = 20_000_000,
    min_price: float = 5.0,
    max_price: float = 10_000.0,
    lookback_days: int = 90,
    max_symbols: int | None = None,
    batch_size: int = 200,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    実際の出来高を見て、売買できる銘柄だけに絞る。
    ※ この関数はネット接続が必要（Mac本体で実行すること）

    min_dollar_volume : 1日あたりの平均売買代金の下限（ドル）
                        30万円程度の資金でも、薄い銘柄は板が飛ぶので2000万ドルは欲しい
    min_price         : 低位株は最小刻み(1セント)の影響が相対的に大きく、
                        スリッページが跳ね上がるので除外する
    """
    import yfinance as yf

    rows = []
    total = len(symbols)
    for i in range(0, total, batch_size):
        batch = symbols[i:i + batch_size]
        if verbose:
            print(f"    流動性を確認中 {i + len(batch)}/{total} 銘柄...")
        try:
            raw = yf.download(batch, period=f"{lookback_days}d", auto_adjust=True,
                              progress=False, group_by="column", threads=True)
        except Exception as e:
            print(f"    [警告] 取得失敗: {e}")
            continue
        if raw is None or len(raw) == 0:
            continue
        for sym in batch:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    close = raw[("Close", sym)]
                    vol = raw[("Volume", sym)]
                else:
                    close, vol = raw["Close"], raw["Volume"]
            except KeyError:
                continue
            close, vol = close.dropna(), vol.dropna()
            if len(close) < lookback_days * 0.6:
                continue
            px = float(close.iloc[-1])
            dv = float((close * vol).tail(60).median())
            if not np.isfinite(px) or not np.isfinite(dv):
                continue
            if px < min_price or px > max_price or dv < min_dollar_volume:
                continue
            rows.append({"symbol": sym, "price": round(px, 2),
                         "dollar_volume": round(dv), "bars": len(close)})

    df = pd.DataFrame(rows).sort_values("dollar_volume", ascending=False)
    if max_symbols:
        df = df.head(max_symbols)
    return df.reset_index(drop=True)


def build(
    include: tuple[str, ...] = ("sp500", "midcap", "etf"),
    min_dollar_volume: float = 20_000_000,
    min_price: float = 5.0,
    max_midcap: int = 500,
    out_path: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    ユニバースを組み立ててJSONに保存する。ネット接続が必要。
    一度作れば当面は使い回せる（四半期に1回作り直す程度で十分）。
    """
    frames = []
    if "sp500" in include:
        frames.append(sp500())
    if "etf" in include:
        frames.append(etf_universe())

    meta = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if verbose:
        print(f"  S&P500 + ETF: {len(meta)} 銘柄")

    # 大型株とETFは流動性が自明なので、フィルタは中型株にだけかける
    if "midcap" in include:
        cands = midcap_candidates()
        if verbose:
            print(f"  中型株候補 {len(cands)} 銘柄に流動性フィルタをかけます...")
        liq = apply_liquidity_filter(
            cands["symbol"].tolist(), min_dollar_volume=min_dollar_volume,
            min_price=min_price, max_symbols=max_midcap, verbose=verbose)
        mid = cands[cands["symbol"].isin(liq["symbol"])].copy()
        mid = mid.merge(liq[["symbol", "price", "dollar_volume"]], on="symbol", how="left")
        meta = pd.concat([meta, mid], ignore_index=True)
        if verbose:
            print(f"  中型株 {len(mid)} 銘柄が通過")

    meta = meta.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    meta["built_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = out_path or os.path.join(DATA_DIR, "universe.json")
    meta.to_json(out_path, orient="records", force_ascii=False, indent=1)
    if verbose:
        print(f"  合計 {len(meta)} 銘柄 → {out_path}")
    return meta


def load(group: str | None = None, path: str | None = None) -> pd.DataFrame:
    """保存済みのユニバースを読む。group で 'sp500' / 'midcap' / 'etf' に絞れる"""
    path = path or os.path.join(DATA_DIR, "universe.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} がありません。先に `python build_universe.py` を実行してください。")
    df = pd.read_json(path)
    if group:
        df = df[df["group"] == group]
    return df.reset_index(drop=True)


def symbols(group: str | None = None, limit: int | None = None) -> list[str]:
    df = load(group)
    s = df["symbol"].tolist()
    return s[:limit] if limit else s

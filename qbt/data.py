"""
株価データの取得とキャッシュ。

対応データソース:
  yfinance   : 日本株(7203.T形式) / 米国株(AAPL形式) 両対応。無料。検証用の第一候補。
  csv        : 自前のCSVを読む（J-Quants等から落としたデータをここに置く）
  jquants    : J-Quants API v2（要APIキー。日本株の公式データ）
  synthetic  : ネット接続なしで動作確認するための合成データ

一度取得したデータは data_cache/ に parquet で保存し、次回以降は再ダウンロードしない。
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CACHE_DIR = os.environ.get("QBT_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache"))

# 1銘柄あたり必要な列
OHLCV = ["open", "high", "low", "close", "volume"]


def _cache_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{h}.pkl.gz")


def load(
    symbols: list[str],
    start: str,
    end: str,
    source: str = "yfinance",
    use_cache: bool = True,
    csv_dir: str | None = None,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """
    銘柄コードのリストから {シンボル: OHLCVのDataFrame} を返す。

    DataFrame の index は日付（tz naive）、列は open/high/low/close/volume（小文字）。
    株式分割・配当は調整済みの価格を使う（yfinance の auto_adjust=True 相当）。
    """
    key = f"{source}|{','.join(sorted(symbols))}|{start}|{end}|{seed}"
    cp = _cache_path(key)

    if use_cache and os.path.exists(cp):
        try:
            return _unflatten(pd.read_pickle(cp))
        except Exception:
            os.remove(cp)  # 壊れたキャッシュは捨てて取り直す

    if source == "yfinance":
        data = _load_yfinance(symbols, start, end)
    elif source == "csv":
        data = _load_csv(symbols, start, end, csv_dir)
    elif source == "jquants":
        data = _load_jquants(symbols, start, end)
    elif source == "synthetic":
        data = _load_synthetic(symbols, start, end, seed)
    else:
        raise ValueError(f"未知のデータソース: {source}")

    if not data:
        raise RuntimeError("データを1銘柄も取得できませんでした。銘柄コードと期間、ネット接続を確認してください。")

    if use_cache:
        _flatten(data).to_pickle(cp)
    return data


def _flatten(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for sym, df in data.items():
        d = df.copy()
        d["symbol"] = sym
        frames.append(d.reset_index().rename(columns={d.index.name or "index": "date"}))
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def _unflatten(flat: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for sym, g in flat.groupby("symbol"):
        d = g.drop(columns=["symbol"]).set_index("date").sort_index()
        out[str(sym)] = d[OHLCV]
    return out


# ------------------------------------------------------------------ yfinance

def _load_yfinance(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(
        symbols, start=start, end=end,
        auto_adjust=True, progress=False, group_by="column", threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    if raw is None or len(raw) == 0:
        return out

    for sym in symbols:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = pd.DataFrame({
                    "open": raw[("Open", sym)], "high": raw[("High", sym)],
                    "low": raw[("Low", sym)], "close": raw[("Close", sym)],
                    "volume": raw[("Volume", sym)],
                })
            else:
                df = raw.rename(columns=str.lower)[OHLCV]
        except KeyError:
            print(f"  [警告] {sym}: データを取得できませんでした（スキップ）")
            continue

        df = df.dropna(subset=["close"])
        df.index = pd.to_datetime(df.index).tz_localize(None)
        if len(df) < 30:
            print(f"  [警告] {sym}: データが {len(df)} 本しかありません（スキップ）")
            continue
        out[sym] = df.sort_index()
    return out


# ------------------------------------------------------------------ CSV

def _load_csv(symbols: list[str], start: str, end: str, csv_dir: str | None) -> dict[str, pd.DataFrame]:
    if not csv_dir:
        raise ValueError("source: csv を使う場合は config の data.csv_dir を指定してください")
    out = {}
    for sym in symbols:
        path = os.path.join(csv_dir, f"{sym}.csv")
        if not os.path.exists(path):
            print(f"  [警告] {path} が見つかりません（スキップ）")
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        datecol = next((c for c in ("date", "日付", "datetime", "time") if c in df.columns), df.columns[0])
        df["date"] = pd.to_datetime(df[datecol])
        df = df.set_index("date").sort_index()
        missing = [c for c in OHLCV if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: 列 {missing} がありません。open/high/low/close/volume が必要です")
        out[sym] = df.loc[start:end, OHLCV].dropna(subset=["close"])
    return out


# ------------------------------------------------------------------ J-Quants

def _load_jquants(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """
    J-Quants API v2 から日本株の日足を取得する。
    環境変数 JQUANTS_API_KEY にAPIキーを設定しておくこと。
    銘柄コードは "7203" のような4桁または5桁のコードで指定する。
    """
    import requests

    api_key = os.environ.get("JQUANTS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "環境変数 JQUANTS_API_KEY が設定されていません。\n"
            "J-Quants のダッシュボードで発行したAPIキーを設定してください。"
        )
    base = "https://api.jquants.com/v2"
    headers = {"Authorization": f"Bearer {api_key}"}
    out = {}
    for sym in symbols:
        code = sym.replace(".T", "")
        rows, pagination_key = [], None
        while True:
            params = {"code": code, "from": start, "to": end}
            if pagination_key:
                params["pagination_key"] = pagination_key
            r = requests.get(f"{base}/prices/daily_quotes", headers=headers, params=params, timeout=60)
            r.raise_for_status()
            j = r.json()
            rows.extend(j.get("daily_quotes", []))
            pagination_key = j.get("pagination_key")
            if not pagination_key:
                break
        if not rows:
            print(f"  [警告] {sym}: J-Quantsからデータを取得できませんでした（スキップ）")
            continue
        df = pd.DataFrame(rows)
        # 分割調整済みの列を優先して使う
        colmap = {
            "open": "AdjustmentOpen", "high": "AdjustmentHigh", "low": "AdjustmentLow",
            "close": "AdjustmentClose", "volume": "AdjustmentVolume",
        }
        d = pd.DataFrame({k: pd.to_numeric(df[v], errors="coerce") for k, v in colmap.items() if v in df.columns})
        d.index = pd.to_datetime(df["Date"])
        out[sym] = d.dropna(subset=["close"]).sort_index()
    return out


# ------------------------------------------------------------------ 合成データ

def _load_synthetic(symbols: list[str], start: str, end: str, seed: int = 0) -> dict[str, pd.DataFrame]:
    """
    ネット接続なしでエンジンの動作確認をするための擬似株価。
    トレンド+平均回帰+ノイズを混ぜた幾何ブラウン運動で生成する。
    """
    dates = pd.bdate_range(start, end)
    out = {}
    for i, sym in enumerate(symbols):
        rng = np.random.default_rng(seed + i * 7919)
        n = len(dates)
        drift = rng.uniform(0.00015, 0.0006)
        vol = rng.uniform(0.010, 0.025)
        # 低周波のトレンド成分を足して、局面がある相場らしくする
        trend = np.sin(np.linspace(0, rng.uniform(3, 9) * np.pi, n)) * vol * 0.8
        rets = drift + trend * 0.15 + rng.normal(0, vol, n)
        # 米国株の実際の株価帯（20〜400ドル）に合わせる。
        # ここを非現実的な水準にすると、少額運用の「1株も買えない」問題が
        # 検証で再現されなくなってしまう。
        base = float(rng.uniform(20, 400))
        close = base * np.exp(np.cumsum(rets))
        intraday = np.abs(rng.normal(0, vol * 0.7, n))
        op = close * (1 + rng.normal(0, vol * 0.4, n))
        hi = np.maximum(close, op) * (1 + intraday)
        lo = np.minimum(close, op) * (1 - intraday)
        vols = rng.lognormal(13, 0.5, n)
        out[sym] = pd.DataFrame(
            {"open": op, "high": hi, "low": lo, "close": close, "volume": vols}, index=dates
        )
    return out


def align_calendar(data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """全銘柄の日付の和集合を取り、共通の取引カレンダーを作る"""
    idx = None
    for df in data.values():
        idx = df.index if idx is None else idx.union(df.index)
    return pd.DatetimeIndex(sorted(idx))

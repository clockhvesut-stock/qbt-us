"""
データの正しさを守るための検査。

新しい情報源を増やすより、いま持っているデータが正しいかを確かめるほうが先。
バックテストのバグと同じで、データの欠陥は「良く見える方向」に効くことが多い。

  check_freshness   … 日中の未完成データ・古いデータ・祝日を検知する
  correlation_check … 候補どうしが実質同じ賭けになっていないかを見る
  sector_relative   … 銘柄固有の弱さか、業種全体の弱さかを切り分ける
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

import numpy as np
import pandas as pd

# 米国市場の取引時間（UTC）。夏時間13:30-20:00、冬時間14:30-21:00。
# 両方を含む広めの範囲で「市場が開いている可能性がある」と判定する。
US_OPEN_UTC = time(13, 25)
US_CLOSE_UTC = time(21, 5)

# 米国市場の休場日（固定日 + 主要な移動祝日）。年に一度見直す程度で足りる。
US_HOLIDAYS_2026 = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
}
US_HOLIDAYS_2027 = {
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}
US_HOLIDAYS = US_HOLIDAYS_2026 | US_HOLIDAYS_2027


def last_trading_day(now: datetime) -> pd.Timestamp:
    """直近の取引日を返す（土日と祝日を飛ばす）"""
    d = pd.Timestamp(now.date())
    # 市場が閉まる前なら、その日はまだ「終わった取引日」ではない
    if now.timetz().replace(tzinfo=None) < US_CLOSE_UTC:
        d -= pd.Timedelta(days=1)
    while d.weekday() >= 5 or d.strftime("%Y-%m-%d") in US_HOLIDAYS:
        d -= pd.Timedelta(days=1)
    return d


def check_freshness(data_date: str, generated_at: str,
                    volumes: pd.Series | None = None) -> dict:
    """
    取得したデータが「大引け後の確定値」かどうかを判定する。

    ここを見ないと、取引開始1時間後の未完成な日足を使って
    RSIやATRを計算し、それを根拠に判断してしまう。実際に一度やらかした。

    volumes: 当日出来高 ÷ 20日平均出来高 の系列。
             これが極端に小さいときは、1日分の取引が終わっていない動かぬ証拠になる。
    """
    try:
        gen = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
    except Exception:
        gen = datetime.now(timezone.utc)
    if gen.tzinfo is None:
        gen = gen.replace(tzinfo=timezone.utc)

    d_data = pd.Timestamp(str(data_date)[:10])
    expected = last_trading_day(gen)
    t = gen.timetz().replace(tzinfo=None)
    is_weekday = gen.weekday() < 5
    market_open_now = is_weekday and US_OPEN_UTC <= t <= US_CLOSE_UTC \
        and gen.strftime("%Y-%m-%d") not in US_HOLIDAYS

    # 出来高からの裏取り。中央値が0.5を切るなら1日分に満たない
    vol_med = None
    if volumes is not None and len(volumes.dropna()):
        vol_med = float(volumes.dropna().median())

    problems, level = [], "ok"

    if market_open_now and d_data.date() == gen.date():
        problems.append(
            "取引時間中に取得しているため、当日の足はまだ完成していません。"
            "RSI・ATR・出来高はすべて途中経過の値です")
        level = "bad"

    if vol_med is not None and vol_med < 0.5:
        problems.append(
            f"当日出来高が20日平均の{vol_med*100:.0f}%しかありません。"
            "1日分の取引が終わっていない可能性が高いです")
        level = "bad"

    stale_days = (expected - d_data).days
    if stale_days >= 1 and level != "bad":
        problems.append(
            f"データが{stale_days}日古いままです（直近の取引日は{expected.date()}）。"
            "祝日か、データ取得の失敗が考えられます")
        level = "warn" if stale_days <= 3 else "bad"

    if gen.strftime("%Y-%m-%d") in US_HOLIDAYS:
        problems.append("米国市場は本日休場です。前営業日のデータを見ています")
        level = max(level, "warn", key=["ok", "warn", "bad"].index)

    return {
        "level": level,
        "usable": level != "bad",
        "data_date": str(d_data.date()),
        "expected_trading_day": str(expected.date()),
        "generated_at_utc": gen.isoformat(timespec="seconds"),
        "market_open_at_fetch": market_open_now,
        "volume_ratio_median": round(vol_med, 3) if vol_med is not None else None,
        "problems": problems,
        "summary": ("データは大引け後の確定値です" if level == "ok"
                    else "／".join(problems)),
    }


# ---------------------------------------------------------------- 相関・集中

def correlation_check(data: dict[str, pd.DataFrame], symbols: list[str],
                      lookback: int = 60, high: float = 0.75) -> dict:
    """
    候補と保有の間の相関を見る。

    分散したつもりで同じ賭けを重ねるのが、少額運用で最もよくある失敗。
    実際に、シグナルが出た2銘柄が両方ショッピングセンターREITだったことがある。
    枠を2つ使って、リスクは1銘柄分しか分散していない状態だった。
    """
    syms = [s for s in dict.fromkeys(symbols) if s in data]
    if len(syms) < 2:
        return {"pairs": [], "note": "比較対象が足りません", "max_corr": None}

    rets = pd.DataFrame({
        s: data[s]["close"].pct_change().tail(lookback) for s in syms
    }).dropna(how="all")
    if len(rets) < lookback * 0.6:
        return {"pairs": [], "note": "履歴が不足しています", "max_corr": None}

    c = rets.corr()
    pairs = []
    for i, a in enumerate(syms):
        for b in syms[i + 1:]:
            try:
                v = float(c.loc[a, b])
            except Exception:
                continue
            if np.isfinite(v) and v >= high:
                pairs.append({"a": a, "b": b, "corr": round(v, 3)})
    pairs.sort(key=lambda x: -x["corr"])

    vals = c.where(~np.eye(len(c), dtype=bool)).stack()
    mx = float(vals.max()) if len(vals) else None
    note = ("相関の高い組み合わせはありません" if not pairs else
            f"{len(pairs)}組が相関{high}以上。実質同じ賭けになっている可能性があります")
    return {"pairs": pairs[:10], "max_corr": round(mx, 3) if mx else None,
            "lookback_days": lookback, "note": note}


def sector_relative(feat: pd.DataFrame, sector_map: dict,
                    sectors: list[dict]) -> pd.DataFrame:
    """
    各銘柄の騰落から、その業種の平均を引く。

    「その銘柄が弱い」のか「業種全体が弱い」のかを切り分ける。
    逆張りで拾うなら、業種全体の下げに巻き込まれただけの銘柄より、
    業種は堅調なのに単独で売られた銘柄のほうが筋がいい。
    """
    if feat is None or feat.empty:
        return feat
    f = feat.copy()
    f["sector"] = f["symbol"].map(sector_map)
    means = {s.get("sector"): s for s in (sectors or [])}
    for col, key in (("chg_5d", "平均5日"), ("chg_21d", "平均21日")):
        if col not in f.columns:
            continue
        base = f["sector"].map(lambda x: (means.get(x) or {}).get(key))
        f[f"rel_{col}"] = (f[col] - base).round(4)
    return f


def concentration(positions: list[dict], candidates: list[dict],
                  sector_map: dict) -> dict:
    """業種の偏りを見る。同じ業種に枠が集中していないか"""
    from collections import Counter
    held = [p.get("symbol") for p in (positions or [])]
    cand = [c.get("symbol") for c in (candidates or [])][:8]
    held_sec = Counter(sector_map.get(s, "不明") for s in held if s)
    both_sec = Counter(sector_map.get(s, "不明") for s in (held + cand) if s)
    warn = [f"{k} に {v} 銘柄が集中" for k, v in both_sec.items() if v >= 3]
    return {
        "held_by_sector": dict(held_sec),
        "held_plus_candidates_by_sector": dict(both_sec),
        "warnings": warn,
    }

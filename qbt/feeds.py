"""
価格以外の情報を集める。

ここで集めるのは「事実」だけで、解釈はしない。
ニュースが強気か弱気かの判断はAI側（Claude）が担当する。
分業をはっきりさせておくと、後から「なぜそう判断したのか」を追跡できる。

データ源はすべて無料:
  yfinance     : 価格、ニュース見出し、マクロ指標
  SEC EDGAR    : 提出書類（8-K, 10-Q, 10-K等）。提出時刻つき。APIキー不要
  Alpaca       : ニュース本文（ペーパー口座の無料APIキーがあれば）
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

SEC_UA = os.environ.get("SEC_USER_AGENT", "qbt-research contact@example.com")

# マクロの体温計。世界情勢の変化はまずここに出る。
MACRO = {
    "^GSPC": "S&P500",
    "^VIX": "VIX(恐怖指数)",
    "^TNX": "米10年金利",
    "^IRX": "米3ヶ月金利",
    "DX-Y.NYB": "ドル指数",
    "CL=F": "WTI原油",
    "GC=F": "金",
    "HG=F": "銅",
    "BTC-USD": "ビットコイン",
    "^N225": "日経平均",
    "^STOXX50E": "欧州株",
}


# ---------------------------------------------------------------- マクロ

def macro_snapshot() -> dict:
    """相場環境の現況。前日比・20日比・200日線との位置関係まで出す"""
    import yfinance as yf

    out = {}
    try:
        raw = yf.download(list(MACRO.keys()), period="1y", auto_adjust=True,
                          progress=False, group_by="column", threads=True)
    except Exception as e:
        return {"error": f"マクロ指標の取得に失敗: {e}"}

    for sym, label in MACRO.items():
        try:
            s = raw[("Close", sym)].dropna() if isinstance(raw.columns, pd.MultiIndex) \
                else raw["Close"].dropna()
        except KeyError:
            continue
        if len(s) < 30:
            continue
        last = float(s.iloc[-1])
        ma200 = float(s.tail(200).mean()) if len(s) >= 200 else None
        out[sym] = {
            "label": label,
            "last": round(last, 4),
            "chg_1d": round(float(s.iloc[-1] / s.iloc[-2] - 1), 5) if len(s) > 1 else None,
            "chg_5d": round(float(s.iloc[-1] / s.iloc[-6] - 1), 5) if len(s) > 5 else None,
            "chg_20d": round(float(s.iloc[-1] / s.iloc[-21] - 1), 5) if len(s) > 21 else None,
            "vs_ma200": round(last / ma200 - 1, 5) if ma200 else None,
            "pctile_1y": round(float((s < last).mean()), 3),
        }
    return out


def market_breadth(closes: pd.DataFrame) -> dict:
    """
    市場の内部状況。指数が上がっていても中身がスカスカなことがある。
    closes: 日付×銘柄 の終値行列
    """
    if closes is None or closes.empty:
        return {}
    ma50 = closes.rolling(50, min_periods=50).mean()
    ma200 = closes.rolling(200, min_periods=200).mean()
    last = closes.iloc[-1]
    up = (closes.iloc[-1] > closes.iloc[-2]) if len(closes) > 1 else pd.Series(dtype=bool)
    hi52 = closes.tail(252).max()
    lo52 = closes.tail(252).min()
    return {
        "above_ma50_pct": round(float((last > ma50.iloc[-1]).mean()), 3),
        "above_ma200_pct": round(float((last > ma200.iloc[-1]).mean()), 3),
        "advancers_pct": round(float(up.mean()), 3) if len(up) else None,
        "near_52w_high_pct": round(float((last / hi52 > 0.95).mean()), 3),
        "near_52w_low_pct": round(float((last / lo52 < 1.05).mean()), 3),
        "symbols_counted": int(closes.shape[1]),
    }


# ---------------------------------------------------------------- ニュース

def yahoo_news(symbols: list[str], limit_per_symbol: int = 4,
               max_age_hours: int = 48) -> list[dict]:
    """yfinance経由でニュース見出しを取る。APIキー不要"""
    import yfinance as yf

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    out = []
    for sym in symbols:
        try:
            items = yf.Ticker(sym).news or []
        except Exception:
            continue
        for it in items[:limit_per_symbol]:
            c = it.get("content", it)
            title = c.get("title") or ""
            if not title:
                continue
            pub = c.get("pubDate") or c.get("providerPublishTime")
            try:
                ts = (datetime.fromisoformat(str(pub).replace("Z", "+00:00"))
                      if isinstance(pub, str)
                      else datetime.fromtimestamp(float(pub), tz=timezone.utc))
            except Exception:
                ts = datetime.now(timezone.utc)
            if ts < cutoff:
                continue
            prov = c.get("provider") or {}
            out.append({
                "symbol": sym,
                "title": title.strip(),
                "summary": (c.get("summary") or "")[:600],
                "publisher": prov.get("displayName") if isinstance(prov, dict) else str(prov),
                "published_at": ts.isoformat(),
                "url": ((c.get("canonicalUrl") or {}).get("url")
                        if isinstance(c.get("canonicalUrl"), dict) else c.get("link", "")),
            })
        time.sleep(0.12)   # 相手のサーバーに負荷をかけない
    return out


def alpaca_news(symbols: list[str], hours: int = 48, limit: int = 50) -> list[dict]:
    """
    Alpacaのニュース API（Benzinga配信）。本文つきで質が高い。
    ペーパー口座の無料APIキーがあれば使える。
    環境変数 ALPACA_API_KEY / ALPACA_SECRET_KEY を設定しておくこと。
    """
    key = os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        return []
    import requests

    start = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    out, page_token = [], None
    try:
        while len(out) < limit:
            params = {"symbols": ",".join(symbols[:50]), "start": start,
                      "limit": min(50, limit - len(out)), "include_content": "true"}
            if page_token:
                params["page_token"] = page_token
            r = requests.get("https://data.alpaca.markets/v1beta1/news",
                             headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec},
                             params=params, timeout=30)
            r.raise_for_status()
            j = r.json()
            for n in j.get("news", []):
                out.append({
                    "symbol": ",".join(n.get("symbols", [])),
                    "title": n.get("headline", ""),
                    "summary": (n.get("summary") or "")[:600],
                    "publisher": n.get("source", "Benzinga"),
                    "published_at": n.get("created_at", ""),
                    "url": n.get("url", ""),
                })
            page_token = j.get("next_page_token")
            if not page_token:
                break
    except Exception as e:
        print(f"  [警告] Alpacaニュースの取得に失敗: {e}")
    return out


# ---------------------------------------------------------------- SEC EDGAR

def sec_recent_filings(ciks: dict[str, str], hours: int = 72,
                       forms: tuple[str, ...] = ("8-K", "10-Q", "10-K", "SC 13D", "SC 13G", "4")
                       ) -> list[dict]:
    """
    SEC EDGARから直近の提出書類を取る。APIキー不要、完全無料。

    重要: acceptanceDateTime（受理時刻）が取れるので、
    「その情報が何時何分に公開されたか」が正確に分かる。
    バックテストで未来を見ないための土台になる。

    ciks: {シンボル: 10桁CIK} の辞書
    """
    import requests

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    headers = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}
    out = []
    for sym, cik in ciks.items():
        if not cik:
            continue
        try:
            r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                             headers=headers, timeout=20)
            if r.status_code != 200:
                continue
            recent = r.json().get("filings", {}).get("recent", {})
        except Exception:
            continue
        n = len(recent.get("accessionNumber", []))
        for i in range(min(n, 25)):
            form = recent["form"][i]
            if forms and form not in forms:
                continue
            acc_dt = recent.get("acceptanceDateTime", [None] * n)[i]
            try:
                ts = datetime.fromisoformat(str(acc_dt).replace("Z", "+00:00"))
            except Exception:
                continue
            if ts < cutoff:
                continue
            acc = recent["accessionNumber"][i].replace("-", "")
            out.append({
                "symbol": sym,
                "form": form,
                "filed_date": recent["filingDate"][i],
                "accepted_at": ts.isoformat(),
                "items": recent.get("items", [""] * n)[i],
                "url": (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/"
                        f"{recent['primaryDocument'][i]}"),
            })
        time.sleep(0.11)   # SECのレート制限は10リクエスト/秒
    return sorted(out, key=lambda x: x["accepted_at"], reverse=True)


def sec_company_facts(cik: str, tags: tuple[str, ...] = ("Revenues", "NetIncomeLoss")) -> dict:
    """
    SEC EDGARのXBRLから財務数値を取る。
    各数値に「いつ提出されたか」がついているので、Point-in-Timeの検証ができる。
    """
    import requests

    out = {}
    for tag in tags:
        try:
            r = requests.get(
                f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json",
                headers={"User-Agent": SEC_UA}, timeout=20)
            if r.status_code != 200:
                continue
            units = r.json().get("units", {}).get("USD", [])
            out[tag] = [{"end": u.get("end"), "val": u.get("val"),
                         "filed": u.get("filed"), "form": u.get("form")}
                        for u in units[-12:]]
        except Exception:
            continue
        time.sleep(0.11)
    return out


def earnings_calendar(symbols: list[str]) -> list[dict]:
    """直近の決算発表予定。決算跨ぎのリスク管理に使う"""
    import yfinance as yf

    out = []
    for sym in symbols:
        try:
            cal = yf.Ticker(sym).calendar
        except Exception:
            continue
        if not cal:
            continue
        d = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if not d:
            continue
        dates = d if isinstance(d, list) else [d]
        for dt in dates[:1]:
            try:
                days = (pd.Timestamp(dt).normalize() - pd.Timestamp.now().normalize()).days
            except Exception:
                continue
            if -1 <= days <= 21:
                out.append({"symbol": sym, "earnings_date": str(dt), "days_until": days})
        time.sleep(0.1)
    return out

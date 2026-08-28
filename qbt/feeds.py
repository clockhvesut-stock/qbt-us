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
import re
import time
from datetime import datetime, timedelta, timezone

import numpy as np
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


# ---------------------------------------------------------------- 市場スキャン

# セクターと資産クラスの代表ETF。個別銘柄と無関係に「相場の材料」を拾うために使う。
MACRO_TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "XLK", "XLF", "XLE",
                 "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]


def market_scan(feat: pd.DataFrame, min_dollar_vol: float = 30_000_000,
                n: int = 15) -> dict:
    """
    その日の相場で「何かが起きた」銘柄を、買いルールとは無関係に洗い出す。

    買いシグナルが出た銘柄だけを見ていると、価格が動いたあとにしか材料に気づけない。
    値動きと出来高の異常から先に拾っておくことで、順序を逆にする。

      gainers / losers  … 値動きが大きかった銘柄
      unusual_volume    … 出来高が普段の2倍以上。「誰かが何かを知っている」ことが多い
      breakouts         … 52週高値圏に到達した銘柄
      breakdowns        … 52週安値圏に沈んだ銘柄
    """
    if feat is None or feat.empty:
        return {}
    f = feat[feat["dollar_vol"] >= min_dollar_vol].copy()
    if f.empty:
        f = feat.copy()

    def take(df, cols=("symbol", "price", "chg_1d", "vol_ratio",
                       "rsi14", "vs_ma200", "from_52w_high")):
        return [{c: r[c] for c in cols if c in r} for _, r in df.iterrows()]

    return {
        "gainers": take(f.nlargest(n, "chg_1d")),
        "losers": take(f.nsmallest(n, "chg_1d")),
        "unusual_volume": take(
            f[f["vol_ratio"] >= 2.0].nlargest(n, "vol_ratio")),
        "breakouts": take(
            f[f["from_52w_high"] >= -0.01].nlargest(n, "chg_5d")),
        "breakdowns": take(
            f[f["from_52w_low"] <= 0.03].nsmallest(n, "chg_5d")),
    }


def build_watchlist(feat: pd.DataFrame, signals: list[dict], positions: list[str],
                    scan: dict, per_bucket: int = 8, cap: int = 70) -> dict[str, str]:
    """
    ニュースと開示を調べる対象を決める。{シンボル: 監視理由} を返す。

    買い候補だけでなく、保有中・大きく動いた・出来高異常・高値安値更新の銘柄も含める。
    「なぜこの銘柄を見ているのか」を理由として持たせることで、
    後からAIが判断するときに文脈が分かる。
    """
    out: dict[str, str] = {}

    def add(syms, reason, limit=None):
        for s in (syms[:limit] if limit else syms):
            sym = s if isinstance(s, str) else s.get("symbol")
            if sym and sym not in out and len(out) < cap:
                out[sym] = reason

    add(positions, "保有中")
    add([s.get("symbol") for s in signals], "買い候補", per_bucket * 2)
    add(scan.get("unusual_volume", []), "出来高異常", per_bucket)
    add(scan.get("gainers", []), "大幅高", per_bucket)
    add(scan.get("losers", []), "大幅安", per_bucket)
    add(scan.get("breakouts", []), "52週高値圏", per_bucket // 2)
    add(scan.get("breakdowns", []), "52週安値圏", per_bucket // 2)
    return out


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


def macro_news(hours: int = 36, per_ticker: int = 3) -> list[dict]:
    """
    相場全体・資産クラス単位のニュース。個別銘柄とは別枠で拾う。

    指数・国債・金・原油・ドルのティッカーに紐づくニュースは、
    金融政策、地政学、景気指標といった「世界情勢」の話題が中心になる。
    個別株のニュースを追うだけでは絶対に拾えない層。
    """
    tickers = ["^GSPC", "SPY", "QQQ", "TLT", "GLD", "USO", "UUP", "^VIX",
               "XLE", "XLF", "XLK"]
    items = macro_labels = None
    items = yahoo_news(tickers, limit_per_symbol=per_ticker, max_age_hours=hours)
    label = {"^GSPC": "米国株全体", "SPY": "米国株全体", "QQQ": "ハイテク",
             "TLT": "米国債・金利", "GLD": "金・実物資産", "USO": "原油・エネルギー",
             "UUP": "ドル・為替", "^VIX": "リスク認識",
             "XLE": "エネルギー", "XLF": "金融", "XLK": "テクノロジー"}
    for it in items:
        it["topic"] = label.get(it.get("symbol"), "マクロ")
        it["scope"] = "macro"
    return _dedupe_news(items)


# 世界情勢を拾うためのRSS。APIキー不要で、いずれも公的機関か大手メディアの公開フィード。
# config.yaml の feeds.rss で差し替え・追加ができる。
DEFAULT_RSS = [
    ("https://www.federalreserve.gov/feeds/press_all.xml", "FRB発表"),
    ("https://home.treasury.gov/system/files/126/press-releases.xml", "米財務省"),
    ("https://feeds.a.dj.com/rss/RSSWorldNews.xml", "世界情勢"),
    ("https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "市場"),
    ("https://www.ecb.europa.eu/rss/press.html", "ECB発表"),
]


def rss_headlines(feeds: list | None = None, hours: int = 36,
                  per_feed: int = 8) -> list[dict]:
    """
    RSSから世界情勢の見出しを取る。標準ライブラリだけで動くので追加インストール不要。

    中央銀行の発表、地政学、政策変更といった「個別銘柄に紐づかない材料」を拾う。
    取れなかったフィードは黙って飛ばす（1つ落ちても全体は止めない）。
    """
    import xml.etree.ElementTree as ET

    import requests

    feeds = feeds or DEFAULT_RSS
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    out = []
    for entry in feeds:
        url, label = (entry if isinstance(entry, (list, tuple))
                      else (entry.get("url"), entry.get("label", "ニュース")))
        try:
            r = requests.get(url, timeout=20, headers={
                "User-Agent": "Mozilla/5.0 (compatible; qbt-research/1.0)"})
            if r.status_code != 200:
                continue
            root = ET.fromstring(r.content)
        except Exception as e:
            print(f"  [警告] RSS取得に失敗 {label}: {str(e)[:80]}")
            continue

        # RSS 2.0 と Atom の両方に対応する
        entries = root.findall(".//item") or root.findall(
            ".//{http://www.w3.org/2005/Atom}entry")
        for e in entries[:per_feed]:
            def txt(*names):
                for n in names:
                    el = e.find(n)
                    if el is not None and (el.text or el.get("href")):
                        return (el.text or el.get("href") or "").strip()
                return ""

            title = txt("title", "{http://www.w3.org/2005/Atom}title")
            if not title:
                continue
            pub = txt("pubDate", "{http://purl.org/dc/elements/1.1/}date",
                      "{http://www.w3.org/2005/Atom}updated")
            ts = _parse_date(pub)
            if ts and ts < cutoff:
                continue
            out.append({
                "symbol": "", "topic": label, "scope": "world",
                "title": title,
                "summary": (txt("description",
                                "{http://www.w3.org/2005/Atom}summary") or "")[:400],
                "publisher": label,
                "published_at": ts.isoformat() if ts else "",
                "url": txt("link", "{http://www.w3.org/2005/Atom}link"),
            })
        time.sleep(0.2)
    return _dedupe_news(out)


def _parse_date(s: str):
    if not s:
        return None
    from email.utils import parsedate_to_datetime
    for fn in (parsedate_to_datetime,
               lambda x: datetime.fromisoformat(x.replace("Z", "+00:00"))):
        try:
            d = fn(s)
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _dedupe_news(items: list[dict]) -> list[dict]:
    """同じ見出しが複数のティッカー・フィードから来ることがあるので1つにまとめる"""
    seen, out = set(), []
    for it in items:
        key = (it.get("title") or "").strip().lower()[:90]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return sorted(out, key=lambda x: x.get("published_at") or "", reverse=True)


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


def corporate_events(symbols: list[str], max_hold_days: int = 20) -> dict:
    """
    決算発表日と配当落ち日を取る。優位性の話ではなく、事故を防ぐための情報。

    決算跨ぎ: 保有期間が3〜20日の戦略にとって、その間に決算があるかは死活問題。
              決算は方向が読めないまま±5〜10%飛ぶ。ルールで拾った小さな優位性を
              一晩で吹き飛ばす。

    配当落ち: 権利落ち日に株価は配当額ぶん機械的に下がる。
              RSIで「売られすぎ」を拾うルールにとって、これは偽シグナルの発生源。
              「安くなった」のではなく「権利が抜けた」だけなので、反発しない。
    """
    import yfinance as yf

    today = pd.Timestamp.now().normalize()
    earnings, dividends, in_window = [], [], []

    for sym in symbols:
        try:
            cal = yf.Ticker(sym).calendar
        except Exception:
            continue
        if not cal or not isinstance(cal, dict):
            time.sleep(0.08)
            continue

        ed = cal.get("Earnings Date")
        if ed:
            dates = ed if isinstance(ed, list) else [ed]
            try:
                d0 = pd.Timestamp(dates[0]).normalize()
                days = int((d0 - today).days)
            except Exception:
                days = None
            if days is not None and -2 <= days <= 60:
                rec = {"symbol": sym, "date": str(d0.date()), "days_until": days}
                earnings.append(rec)
                if 0 <= days <= max_hold_days:
                    in_window.append({**rec, "risk": "決算跨ぎ",
                                      "note": f"保有期間({max_hold_days}日)の中に決算がある"})

        xd = cal.get("Ex-Dividend Date")
        if xd:
            try:
                d0 = pd.Timestamp(xd).normalize()
                days = int((d0 - today).days)
            except Exception:
                days = None
            if days is not None and -5 <= days <= 30:
                rec = {"symbol": sym, "date": str(d0.date()), "days_until": days,
                       "amount": cal.get("Dividend Value")}
                dividends.append(rec)
                if -3 <= days <= 1:
                    in_window.append({**rec, "risk": "配当落ち",
                                      "note": "権利落ちによる機械的な下げ。押し目ではない"})
        time.sleep(0.08)

    return {
        "earnings": sorted(earnings, key=lambda x: x["days_until"]),
        "ex_dividends": sorted(dividends, key=lambda x: x["days_until"]),
        "alerts": in_window,
    }


# ---------------------------------------------------------------- 8-Kの分類

# 8-Kの項目コード。何が起きたかがコードだけで分かる。
# high=True は株価に直結しやすいもの。
ITEM_8K = {
    "1.01": ("重要な契約の締結", True),
    "1.02": ("重要な契約の終了", True),
    "1.03": ("破産・管財手続き", True),
    "2.01": ("資産の取得または処分", True),
    "2.02": ("業績の公表", True),
    "2.03": ("債務の発生", False),
    "2.04": ("債務の期限の利益喪失", True),
    "2.05": ("リストラ費用の計上", True),
    "2.06": ("資産の減損", True),
    "3.01": ("上場廃止通知・上場規則違反", True),
    "3.02": ("未登録株式の発行", False),
    "3.03": ("株主の権利の変更", False),
    "4.01": ("監査法人の異動", True),
    "4.02": ("過年度決算の信頼性を否定", True),
    "5.01": ("支配権の変更", True),
    "5.02": ("役員の異動", True),
    "5.03": ("定款・決算期の変更", False),
    "5.07": ("株主総会の結果", False),
    "7.01": ("Reg FD開示", False),
    "8.01": ("その他の重要事象", False),
    "9.01": ("財務諸表・添付資料", False),
}


def classify_filing(f: dict) -> dict:
    """
    提出書類に「何が起きたか」のラベルを付ける。
    すでに取得している items フィールドを使うだけなので、追加のリクエストは不要。
    """
    form = (f.get("form") or "").strip()
    items_raw = (f.get("items") or "")
    codes = re.findall(r"\d\.\d{2}", items_raw)
    labels, high = [], False
    for c in codes:
        name, hi = ITEM_8K.get(c, (None, False))
        if name:
            labels.append(f"{c} {name}")
            high = high or hi

    if not labels:
        form_label = {
            "10-K": "年次報告書", "10-Q": "四半期報告書",
            "SC 13D": "大量保有報告（経営関与の意図あり）",
            "SC 13G": "大量保有報告（純投資）",
            "4": "内部者の売買報告", "DEF 14A": "委任状勧誘",
            "S-1": "新規登録届出", "424B": "目論見書",
        }
        for k, v in form_label.items():
            if form.startswith(k):
                labels = [v]
                high = k in ("SC 13D",)
                break

    return {**f, "labels": labels or [form or "その他"], "significant": high}


def sec_fulltext_search(query: str, forms: tuple[str, ...] = ("8-K",),
                        days: int = 7, limit: int = 20) -> list[dict]:
    """
    EDGAR全文検索。開示の「本文」からキーワードを引く。無料、APIキー不要。

    書類が出たという事実だけでなく、中身に何が書かれているかを見にいく。
    「guidance」「impairment」「restructuring」といった語を追うと、
    株価が動く前の材料に当たることがある。
    """
    import requests

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    params = {
        "q": f'"{query}"',
        "dateRange": "custom",
        "startdt": start.isoformat(),
        "enddt": end.isoformat(),
    }
    if forms:
        params["forms"] = ",".join(forms)
    try:
        r = requests.get("https://efts.sec.gov/LATEST/search-index",
                         params=params, timeout=25,
                         headers={"User-Agent": SEC_UA,
                                  "Accept": "application/json"})
        if r.status_code != 200:
            return []
        hits = (r.json().get("hits") or {}).get("hits") or []
    except Exception as e:
        print(f"  [警告] EDGAR全文検索に失敗 ({query}): {str(e)[:80]}")
        return []

    out = []
    for h in hits[:limit]:
        src = h.get("_source") or {}
        tickers = src.get("tickers") or []
        out.append({
            "query": query,
            "symbol": (tickers[0] if tickers else ""),
            "company": (src.get("display_names") or [""])[0],
            "form": src.get("root_form") or src.get("file_type") or "",
            "filed_at": src.get("file_date") or "",
            "url": f"https://www.sec.gov/Archives/edgar/data/"
                   f"{(src.get('ciks') or [''])[0].lstrip('0')}/"
                   f"{h.get('_id','').split(':')[0].replace('-','')}/",
        })
    time.sleep(0.15)
    return out


# ---------------------------------------------------------------- Reddit

REDDIT_SUBS = ["wallstreetbets", "stocks", "investing", "StockMarket", "options"]

# 銘柄コードと紛らわしい一般英単語。これを除かないとノイズだらけになる。
TICKER_STOPWORDS = {
    "A", "I", "IT", "BE", "SO", "ON", "OR", "AT", "IS", "AS", "BY", "GO", "AN",
    "ALL", "ARE", "CEO", "CFO", "DD", "EPS", "ETF", "FED", "FOR", "GDP", "IPO",
    "NEW", "NOW", "ONE", "OUT", "PE", "PM", "PT", "SEC", "SPY", "THE", "TWO",
    "USA", "USD", "WSB", "YOLO", "AI", "EV", "IV", "OP", "TA", "US", "CPI",
}


def reddit_mentions(symbols: list[str], subs: list[str] | None = None,
                    posts_per_sub: int = 100) -> dict:
    """
    主要サブレディットでの銘柄言及数を数える。非商用なら無料（毎分100リクエスト）。

    注意: 言及数そのものにはもう優位性はほとんどない。同じデータを無数の業者が見ている。
    意味があるのは「急増」のほう。普段5件の銘柄が急に80件になったら何かが起きている。
    出来高異常と同じ発想で、別の観測経路を1本増やすという位置づけ。

    環境変数 REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET があればOAuthを使う。
    無ければ公開JSONを試すが、こちらは弾かれることがある。
    """
    import requests
    from collections import Counter

    subs = subs or REDDIT_SUBS
    ua = "qbt-research/1.0 (personal research; contact via github)"
    sess = requests.Session()
    sess.headers.update({"User-Agent": ua})
    base = "https://www.reddit.com"

    cid = os.environ.get("REDDIT_CLIENT_ID")
    csec = os.environ.get("REDDIT_CLIENT_SECRET")
    if cid and csec:
        try:
            tr = requests.post("https://www.reddit.com/api/v1/access_token",
                               auth=(cid, csec), data={"grant_type": "client_credentials"},
                               headers={"User-Agent": ua}, timeout=20)
            tok = tr.json().get("access_token")
            if tok:
                sess.headers.update({"Authorization": f"bearer {tok}"})
                base = "https://oauth.reddit.com"
        except Exception as e:
            print(f"  [警告] Reddit認証に失敗: {str(e)[:60]}")

    valid = {s.upper() for s in symbols if s.upper() not in TICKER_STOPWORDS
             and 2 <= len(s) <= 5 and s.replace("-", "").isalpha()}
    counts, samples = Counter(), {}

    for sub in subs:
        for sort in ("hot", "new"):
            try:
                r = sess.get(f"{base}/r/{sub}/{sort}.json",
                             params={"limit": posts_per_sub}, timeout=20)
                if r.status_code != 200:
                    continue
                children = (r.json().get("data") or {}).get("children") or []
            except Exception:
                continue
            for c in children:
                d = c.get("data") or {}
                text = f"{d.get('title','')} {d.get('selftext','')[:900]}"
                # $TSLA 形式と、単独の大文字トークンの両方を拾う
                found = set(re.findall(r"\$([A-Z]{1,5})\b", text))
                found |= {w for w in re.findall(r"\b([A-Z]{2,5})\b", text)}
                for t in found & valid:
                    counts[t] += 1
                    if t not in samples and d.get("title"):
                        samples[t] = {"title": d["title"][:140],
                                      "sub": sub, "score": d.get("score", 0)}
            time.sleep(0.7)   # 毎分100リクエストの制限に対して十分な余裕

    return {
        "counts": dict(counts.most_common(40)),
        "samples": samples,
        "subs": subs,
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def reddit_spikes(today: dict, history_path: str, top: int = 12) -> list[dict]:
    """
    言及数の履歴と比べて、急増している銘柄を拾う。
    水準ではなく変化率を見るのが要点。
    """
    hist = {}
    if os.path.exists(history_path):
        try:
            hist = json.load(open(history_path, encoding="utf-8"))
        except Exception:
            hist = {}

    counts = today.get("counts") or {}
    day = today.get("collected_at", "")[:10]
    out = []
    for sym, n in counts.items():
        past = [v for d, v in (hist.get(sym) or {}).items() if d != day]
        if len(past) >= 3:
            base = float(np.median(past))
            ratio = n / base if base > 0 else float(n)
            if n >= 5 and ratio >= 2.5:
                out.append({"symbol": sym, "count": n,
                            "baseline": round(base, 1), "ratio": round(ratio, 1),
                            "sample": (today.get("samples") or {}).get(sym)})
        elif n >= 15:
            out.append({"symbol": sym, "count": n, "baseline": None,
                        "ratio": None, "sample": (today.get("samples") or {}).get(sym),
                        "note": "履歴が足りないため急増かどうかは未判定"})

    # 履歴を更新（30日分だけ保持）
    for sym, n in counts.items():
        hist.setdefault(sym, {})[day] = n
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    hist = {s: {d: v for d, v in rec.items() if d >= cutoff}
            for s, rec in hist.items()}
    hist = {s: r for s, r in hist.items() if r}
    try:
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        json.dump(hist, open(history_path, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception:
        pass

    return sorted(out, key=lambda x: -(x.get("ratio") or 0))[:top]

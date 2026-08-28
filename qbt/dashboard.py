"""
ダッシュボードHTMLを組み立てる。

daily.py が集めた事実（raw.json）と、Claudeが書いた解釈（outbox/*.json）を
1枚のページにまとめる。外部サーバーへの通信は一切しない自己完結HTMLなので、
どこに置いても、どの端末で開いても同じものが表示される。

    from qbt import dashboard
    dashboard.build(raw, verdict, out_path="dashboard.html")
"""
from __future__ import annotations

import html
import json
import math
from datetime import datetime

# ---------------------------------------------------------------- 表示ヘルパ

def pct(x, d=1, sign=True):
    try:
        v = float(x) * 100
    except (TypeError, ValueError):
        return "—"
    return f"{v:+.{d}f}%" if sign else f"{v:.{d}f}%"


def num(x, d=2):
    try:
        return f"{float(x):,.{d}f}"
    except (TypeError, ValueError):
        return "—"


def money(x, d=2):
    try:
        return f"${float(x):,.{d}f}"
    except (TypeError, ValueError):
        return "—"


def esc(x):
    return html.escape(str(x if x is not None else ""))


def tone_of(x, good_high=True):
    """数値の符号から意味色を決める"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "flat"
    if abs(v) < 1e-9:
        return "flat"
    up = v > 0
    return "up" if up == good_high else "down"


# ---------------------------------------------------------------- SVG部品

def sparkline(values, width=118, height=30, tone="up"):
    """小さな折れ線。数字の隣に置いて推移の形だけ伝える"""
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    step = width / (len(vals) - 1)
    pts = " ".join(f"{i*step:.1f},{height - (v-lo)/rng*(height-4) - 2:.1f}"
                   for i, v in enumerate(vals))
    last_x = width
    last_y = height - (vals[-1] - lo) / rng * (height - 4) - 2
    return (f'<svg class="spark {tone}" viewBox="0 0 {width} {height}" '
            f'preserveAspectRatio="none" aria-hidden="true">'
            f'<polyline points="{pts}" fill="none" stroke="currentColor" '
            f'stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>'
            f'<circle cx="{last_x-1:.1f}" cy="{last_y:.1f}" r="2.4" fill="currentColor"/>'
            f'</svg>')


def equity_chart(points, initial, width=920, height=220):
    """
    ペーパートレードの資産推移。
    points: [[日付文字列, 資産額], ...]
    """
    if not points or len(points) < 2:
        return ('<p class="empty">まだ記録がありません。'
                'ペーパートレードを始めると、ここに資産の推移が出ます。</p>')

    vals = [float(p[1]) for p in points]
    labels = [str(p[0]) for p in points]
    ml, mr, mt, mb = 58, 14, 12, 26
    pw, ph = width - ml - mr, height - mt - mb
    lo, hi = min(vals + [initial]), max(vals + [initial])
    pad = (hi - lo) * 0.12 or max(abs(hi) * 0.05, 1)
    lo, hi = lo - pad, hi + pad
    sx = lambda i: ml + i / max(len(vals) - 1, 1) * pw
    sy = lambda v: mt + (hi - v) / (hi - lo) * ph

    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="資産推移">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = sy(v)
        out.append(f'<line class="grid" x1="{ml}" x2="{width-mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{ml-8}" y="{y+4:.1f}" text-anchor="end">'
                   f'${v:,.0f}</text>')
    # 元本ライン
    y0 = sy(initial)
    out.append(f'<line class="baseline" x1="{ml}" x2="{width-mr}" y1="{y0:.1f}" y2="{y0:.1f}"/>')
    out.append(f'<text class="tick base" x="{width-mr}" y="{y0-6:.1f}" text-anchor="end">元本</text>')

    pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
    tone = "up" if vals[-1] >= initial else "down"
    out.append(f'<polygon class="area {tone}" points="{ml},{sy(lo):.1f} {pts} '
               f'{sx(len(vals)-1):.1f},{sy(lo):.1f}"/>')
    out.append(f'<polyline class="line {tone}" points="{pts}" fill="none" stroke-width="2"/>')
    out.append(f'<circle class="dot {tone}" cx="{sx(len(vals)-1):.1f}" '
               f'cy="{sy(vals[-1]):.1f}" r="3.6"/>')

    stepn = max(1, len(labels) // 6)
    for i in range(0, len(labels), stepn):
        out.append(f'<text class="tick" x="{sx(i):.1f}" y="{height-6}" '
                   f'text-anchor="middle">{esc(labels[i][5:])}</text>')
    out.append("</svg>")
    return "".join(out)


def sector_bars(sectors, width=920):
    """セクター別の5日騰落。資金がどこへ向かっているかを一目で見る"""
    if not sectors:
        return '<p class="empty">セクターデータがありません。</p>'
    rows = sorted(sectors, key=lambda s: s.get("平均5日") or 0, reverse=True)
    vmax = max(abs(float(s.get("平均5日") or 0)) for s in rows) or 0.01
    out = ['<div class="sectorlist">']
    for s in rows:
        v = float(s.get("平均5日") or 0)
        w = abs(v) / vmax * 50.0          # 中央から左右50%ずつ。%指定なので幅が変わっても崩れない
        pos = v >= 0
        left = 50.0 if pos else 50.0 - w
        out.append(
            f'<div class="secrow">'
            f'<span class="secname">{esc(s.get("sector",""))}</span>'
            f'<span class="secbar"><i class="{"up" if pos else "down"}" '
            f'style="left:{left:.1f}%;width:{max(w,1.0):.1f}%"></i>'
            f'<b class="mid"></b></span>'
            f'<span class="secval {tone_of(v)}">{pct(v)}</span>'
            f'<span class="secma">MA200超 {pct(s.get("MA200超え比率"),0,sign=False)}</span>'
            f'</div>')
    out.append("</div>")
    return "".join(out)


# ---------------------------------------------------------------- セクション
#
#  構成の方針:
#    見るべき順に、上から4つだけ。
#      1. 今日やること
#      2. 保有ポジション
#      3. 相場のひとこと
#      4. 成績
#    それ以外（候補一覧・セクター・SEC書類）は畳んでおく。
#    毎朝1分で読み終わるページにする。

def _today_section(verdict, raw):
    """今日やること。ページの主役"""
    buys = (verdict or {}).get("buys") or []
    sells = (verdict or {}).get("sells") or []
    summary = (verdict or {}).get("summary") or ""

    if not buys and not sells:
        return f"""
<section id="today">
  <div class="nothing">
    <p>買いの条件を満たす銘柄はなく、保有銘柄にも手仕舞いの合図も出ていません。
       見送りも判断のうちです。</p>
  </div>
  {f'<p class="say">{esc(summary)}</p>' if summary else ''}
</section>"""

    cards = []
    for b in sells:
        cards.append(f"""
<article class="card sell">
  <div class="ch"><span class="tag sell">売り</span>
    <span class="tk">{esc(b.get('symbol'))}</span>
    <span class="pr">{money(b.get('price'))}</span>
    <span class="pl {tone_of(b.get('pnl_pct'))}">{pct(b.get('pnl_pct'))}</span></div>
  <p class="cw">{esc(b.get('reason',''))}</p>
  <div class="cf">{num(b.get('shares'),4)} 株を手仕舞い</div>
</article>""")
    for b in buys:
        cards.append(f"""
<article class="card buy">
  <div class="ch"><span class="tag buy">買い</span>
    <span class="tk">{esc(b.get('symbol'))}</span>
    <span class="pr">{money(b.get('price'))}</span></div>
  <p class="cw">{esc(b.get('reason',''))}</p>
  <div class="cf">{num(b.get('shares'),4)} 株 / {money(b.get('notional'))}　
    損切り {money(b.get('stop_price'))}</div>
</article>"""
        )
    return f"""
<section id="today">
  {f'<p class="say">{esc(summary)}</p>' if summary else ''}
  <div class="cards">{''.join(cards)}</div>
</section>"""


def _risk_section(raw):
    """決算跨ぎ・配当落ち・相関の集中。判断の前に目に入る位置に置く"""
    rows = []

    for a in ((raw.get("corporate_events") or {}).get("alerts") or [])[:8]:
        rows.append(f'<div class="rk"><span class="rt">{esc(a.get("risk"))}</span>'
                    f'<span class="rs">{esc(a.get("symbol"))}</span>'
                    f'<span class="rx">{esc(a.get("date"))}　{esc(a.get("note",""))}</span></div>')

    for p in ((raw.get("correlation") or {}).get("pairs") or [])[:4]:
        rows.append(f'<div class="rk"><span class="rt">相関</span>'
                    f'<span class="rs">{esc(p.get("a"))}・{esc(p.get("b"))}</span>'
                    f'<span class="rx">相関 {num(p.get("corr"),2)}。'
                    f'2枠使っても分散は1銘柄分にしかなりません</span></div>')

    for w in ((raw.get("concentration") or {}).get("warnings") or [])[:3]:
        rows.append(f'<div class="rk"><span class="rt">集中</span>'
                    f'<span class="rs">—</span><span class="rx">{esc(w)}</span></div>')

    if not rows:
        return ""
    return f'<section id="risk"><h2>気をつけること</h2><div class="rks">{"".join(rows)}</div></section>'


def _holdings_section(positions, account):
    if not positions:
        return f"""
<section id="holdings">
  <h2>保有</h2>
  <p class="none">なし　現金 {money(account.get('cash'))}</p>
</section>"""

    rows = []
    for p in positions:
        flags = p.get("exit_flags") or []
        rows.append(f"""
<div class="hold{' warn' if flags else ''}">
  <div class="hl">
    <span class="tk">{esc(p.get('symbol'))}</span>
    <span class="hd">{esc(p.get('bars_held'))}日目</span>
  </div>
  <div class="hr">
    <span class="hp {tone_of(p.get('pnl_pct'))}">{pct(p.get('pnl_pct'))}</span>
    <span class="hs">{money(p.get('price'))}　損切り {money(p.get('stop_price'))}</span>
  </div>
  {('<div class="hf">' + ' / '.join(esc(f) for f in flags) + '</div>') if flags else ''}
</div>""")
    return f"""
<section id="holdings">
  <h2>保有 <span class="cnt">{len(positions)}</span></h2>
  <div class="holds">{''.join(rows)}</div>
  <p class="none">現金 {money(account.get('cash'))}</p>
</section>"""


def _market_line(raw, verdict=None):
    """相場はひとことと数字4つだけ。詳細は畳む"""
    macro = raw.get("macro") or {}
    regime = raw.get("regime") or {}
    breadth = raw.get("breadth") or {}

    tiles = []
    for sym in ("^GSPC", "^VIX", "^TNX", "DX-Y.NYB"):
        m = macro.get(sym)
        if not m:
            continue
        t = tone_of(m.get("chg_1d"), good_high=(sym != "^VIX"))
        v = float(m.get("last") or 0)
        tiles.append(f"""
<div class="tile">
  <div class="tl">{esc(m.get('label'))}</div>
  <div class="tv">{num(v, 2 if abs(v) < 1000 else 0)}</div>
  <div class="td {t}">{pct(m.get('chg_1d'))}</div>
</div>""")

    b200 = breadth.get("above_ma200_pct")

    # 世界情勢のうち、AIが「効く」と判断したものだけを表に出す
    wn = (verdict or {}).get("world") or []
    world = ""
    if wn:
        rows = "".join(
            f'<div class="wr"><span class="wt">{esc(w.get("topic",""))}</span>'
            f'<span class="wx">{esc(w.get("note") or w.get("title",""))}</span></div>'
            for w in wn[:5])
        world = f'<div class="world"><div class="wh">世界の動き</div>{rows}</div>'

    return f"""
<section id="market">
  <h2>相場</h2>
  <p class="say"><span class="chip {esc(regime.get('regime',''))}">{esc(regime.get('regime','—'))}</span>
    {esc(regime.get('note',''))}</p>
  <div class="tiles">{''.join(tiles)}</div>
  <p class="none">200日線を上回る銘柄 {pct(b200,0,sign=False)}
    （半分を割ると、指数が上でも中身は弱い）</p>
  {world}
</section>"""


def _score_section(perf, account):
    initial = float((perf or {}).get("initial") or account.get("initial_cash") or 2000)
    points = (perf or {}).get("equity") or []
    last = float(points[-1][1]) if points else initial
    ret = last / initial - 1 if initial else 0
    st = (perf or {}).get("stats") or {}
    return f"""
<section id="score">
  <h2>成績</h2>
  <div class="big">
    <div class="bn {tone_of(ret)}">{pct(ret)}</div>
    <div class="bs">{money(last)}　元本 {money(initial)}</div>
  </div>
  <div class="tiles">
    <div class="tile"><div class="tl">最大下落</div>
      <div class="tv down">{pct(st.get('max_dd'))}</div></div>
    <div class="tile"><div class="tl">取引</div>
      <div class="tv">{st.get('trades',0)}</div></div>
    <div class="tile"><div class="tl">勝率</div>
      <div class="tv">{pct(st.get('win_rate'),0,sign=False)}</div></div>
    <div class="tile"><div class="tl">連敗</div>
      <div class="tv">{st.get('losing_streak',0)}</div></div>
  </div>
  {equity_chart(points, initial)}
</section>"""


def _details_section(raw, verdict):
    """詳しく見たいときだけ開く。普段は畳んでおく"""
    blocks = []

    wnews = raw.get("world_news") or []
    if wnews:
        items = []
        for n in wnews[:25]:
            items.append(f"""
<div class="nw">
  <div class="nh"><span class="wt">{esc(n.get('topic',''))}</span>
    <span class="nd">{esc(str(n.get('published_at',''))[:10])}</span></div>
  <div class="nt">{esc(n.get('title'))}</div>
</div>""")
        blocks.append(("世界情勢・マクロ", f'<div class="nws">{"".join(items)}</div>'))

    news = raw.get("news") or []
    scores = ((verdict or {}).get("news_scores") or {})
    if news:
        items = []
        for n in news[:20]:
            sc = scores.get(n.get("url")) or scores.get(n.get("title")) or {}
            lvl, note = sc.get("impact"), sc.get("note")
            items.append(f"""
<div class="nw">
  <div class="nh"><span class="tk">{esc(n.get('symbol'))}</span>
    {f'<span class="lv {esc(lvl)}">{esc(lvl)}</span>' if lvl else ''}
    {f'<span class="wt">{esc(n.get("topic"))}</span>' if n.get("topic") else ''}
    <span class="nd">{esc(str(n.get('published_at',''))[:10])}</span></div>
  <div class="nt">{esc(n.get('title'))}</div>
  {f'<div class="nv">{esc(note)}</div>' if note else ''}
</div>""")
        blocks.append(("ニュース", f'<div class="nws">{"".join(items)}</div>'))

    sigs = raw.get("track_a_signals") or []
    if sigs:
        rows = "".join(
            f'<tr><td>{esc(s.get("rank"))}</td><td class="tk">{esc(s.get("symbol"))}</td>'
            f'<td class="r">{money(s.get("price"))}</td>'
            f'<td class="r">{num(s.get("rsi14"),1)}</td>'
            f'<td class="r {tone_of(s.get("chg_21d"))}">{pct(s.get("chg_21d"))}</td></tr>'
            for s in sigs[:15])
        blocks.append(("ルールが出した候補",
                       f'<div class="tw"><table><thead><tr><th></th><th>銘柄</th>'
                       f'<th class="r">株価</th><th class="r">RSI</th>'
                       f'<th class="r">21日</th></tr></thead><tbody>{rows}</tbody></table></div>'))

    spikes = raw.get("reddit_spikes") or []
    if spikes:
        rows = "".join(
            f'<tr><td class="tk">{esc(x.get("symbol"))}</td>'
            f'<td class="r">{esc(x.get("count"))}件</td>'
            f'<td class="r">{("x"+num(x.get("ratio"),1)) if x.get("ratio") else "—"}</td>'
            f'<td class="sm">{esc((x.get("sample") or {}).get("title",""))}</td></tr>'
            for x in spikes[:12])
        blocks.append(("Redditの言及急増",
                       f'<div class="tw"><table><thead><tr><th>銘柄</th>'
                       f'<th class="r">言及</th><th class="r">平常比</th>'
                       f'<th>投稿例</th></tr></thead><tbody>{rows}</tbody></table></div>'))

    ce = raw.get("corporate_events") or {}
    if ce.get("earnings"):
        rows = "".join(
            f'<tr><td class="tk">{esc(e.get("symbol"))}</td><td>{esc(e.get("date"))}</td>'
            f'<td class="r">{esc(e.get("days_until"))}日後</td></tr>'
            for e in ce["earnings"][:20])
        blocks.append(("決算発表の予定",
                       f'<div class="tw"><table><thead><tr><th>銘柄</th><th>発表日</th>'
                       f'<th class="r">残り</th></tr></thead><tbody>{rows}</tbody></table></div>'))

    scan = raw.get("scan") or {}
    if scan:
        parts = []
        for key, label in (("unusual_volume", "出来高異常"), ("gainers", "大幅高"),
                           ("losers", "大幅安"), ("breakouts", "52週高値圏"),
                           ("breakdowns", "52週安値圏")):
            rows = scan.get(key) or []
            if not rows:
                continue
            cells = "".join(
                f'<span class="sc"><b>{esc(r.get("symbol"))}</b>'
                f'<i class="{tone_of(r.get("chg_1d"))}">{pct(r.get("chg_1d"))}</i>'
                + (f'<u>x{num(r.get("vol_ratio"),1)}</u>' if key == "unusual_volume" else "")
                + '</span>' for r in rows[:10])
            parts.append(f'<div class="scg"><div class="scl">{label}</div>'
                         f'<div class="scr">{cells}</div></div>')
        if parts:
            blocks.append(("市場スキャン", "".join(parts)))

    secs = raw.get("sectors") or []
    if secs:
        blocks.append(("セクター動向", sector_bars(secs)))

    fil = raw.get("filings") or []
    if fil:
        rows = "".join(
            f'<tr><td class="tk">{esc(f.get("symbol"))}</td>'
            f'<td><span class="form">{esc(f.get("form"))}</span></td>'
            f'<td>{esc(str(f.get("accepted_at",""))[:16].replace("T"," "))}</td>'
            f'<td>{esc(" / ".join(f.get("labels") or []) or f.get("items",""))}{"　⚠" if f.get("significant") else ""}</td></tr>' for f in fil[:15])
        blocks.append(("SEC提出書類",
                       f'<div class="tw"><table><thead><tr><th>銘柄</th><th>種類</th>'
                       f'<th>受理時刻</th><th>項目</th></tr></thead>'
                       f'<tbody>{rows}</tbody></table></div>'))

    if not blocks:
        return ""
    inner = "".join(f'<details><summary>{esc(t)}</summary><div class="db">{b}</div></details>'
                    for t, b in blocks)
    return f'<section id="more"><h2>詳細</h2>{inner}</section>'


# ---------------------------------------------------------------- 本体

def build(raw: dict, verdict: dict | None = None, out_path: str | None = None,
          sample: bool = False) -> str:
    verdict = verdict or {}
    account = raw.get("account") or {"cash": raw.get("cash"), "initial_cash": 2000}
    positions = raw.get("positions") or []
    perf = raw.get("performance") or {}

    n_buy = len(verdict.get("buys") or [])
    n_sell = len(verdict.get("sells") or [])
    if n_buy or n_sell:
        parts = []
        if n_buy:
            parts.append(f"買い {n_buy}")
        if n_sell:
            parts.append(f"売り {n_sell}")
        headline = " / ".join(parts)
    else:
        headline = "今日は何もしない"

    banner = ""
    if sample:
        banner = '<div class="note">サンプルデータで表示しています</div>'

    # データが未完成・古いときは、他の何よりも先に伝える。
    # 途中経過の日足で計算した数字を、確定値として読ませないため。
    dq = raw.get("data_quality") or {}
    if dq.get("problems"):
        cls = "note bad" if dq.get("level") == "bad" else "note"
        items = "".join(f"<div>・{esc(p)}</div>" for p in dq["problems"])
        head = ("このデータは判断に使えません" if dq.get("level") == "bad"
                else "データに注意点があります")
        banner += (f'<div class="{cls}"><b>{head}</b>{items}</div>')

    if raw.get("errors"):
        banner += ('<div class="note bad">'
                   + "　".join(esc(e) for e in raw["errors"]) + "</div>")

    body = (_today_section(verdict, raw)
            + _risk_section(raw)
            + _holdings_section(positions, account)
            + _market_line(raw, verdict)
            + _score_section(perf, account)
            + _details_section(raw, verdict))

    doc = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>米国株デイリー</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=M+PLUS+2:wght@400;500;700;800&family=M+PLUS+1+Code:wght@400;500;600&display=swap">
<style>{_CSS}</style></head>
<body>
<div class="page">
{banner}
<header>
  <p class="date">{esc(raw.get('data_date') or raw.get('date',''))} の終値時点</p>
  <h1>{esc(headline)}</h1>
</header>
{body}
<footer>更新 {esc(str(raw.get('generated_at',''))[:16].replace('T',' '))} UTC<br>
情報提供を目的としたもので、投資助言ではありません。</footer>
</div>
</body></html>"""
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(doc)
    return doc


_CSS = """
:root{color-scheme:light;
--bg:#f6f7f9;--card:#fff;--soft:#eef1f4;
--ink:#151a20;--ink2:#5a6472;--ink3:#8c96a3;
--line:#e3e7ec;--line2:#cdd4dc;
--accent:#1d5c7a;--accent-bg:#1d5c7a12;
--up:#12795a;--up-bg:#12795a12;--down:#a83a35;--down-bg:#a83a3512;
--warn:#a3690f;--warn-bg:#a3690f12}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--bg:#0f1317;--card:#181d23;--soft:#212831;
--ink:#eff2f5;--ink2:#a9b3bf;--ink3:#79838f;
--line:#262d36;--line2:#39424d;
--accent:#5aa8c8;--accent-bg:#5aa8c81c;
--up:#3ba887;--up-bg:#3ba8871c;--down:#d8736e;--down-bg:#d8736e1c;
--warn:#d5a54e;--warn-bg:#d5a54e1c}}
:root[data-theme="dark"]{color-scheme:dark;
--bg:#0f1317;--card:#181d23;--soft:#212831;
--ink:#eff2f5;--ink2:#a9b3bf;--ink3:#79838f;
--line:#262d36;--line2:#39424d;
--accent:#5aa8c8;--accent-bg:#5aa8c81c;
--up:#3ba887;--up-bg:#3ba8871c;--down:#d8736e;--down-bg:#d8736e1c;
--warn:#d5a54e;--warn-bg:#d5a54e1c}

*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:"M PLUS 2","Hiragino Sans",system-ui,sans-serif;
font-size:16px;line-height:1.85;-webkit-font-smoothing:antialiased}
.page{max-width:660px;margin:0 auto;padding:34px 20px 90px}
.tk,.tv,.pr,.pl,.hp,.bn,td.r,.tl,.nd{font-family:"M PLUS 1 Code",ui-monospace,monospace;
font-variant-numeric:tabular-nums}
.up{color:var(--up)}.down{color:var(--down)}.flat{color:var(--ink3)}

.note{background:var(--warn-bg);color:var(--warn);border-radius:9px;
padding:9px 14px;margin-bottom:20px;font-size:13px;font-weight:600}
.note.bad{background:var(--down-bg);color:var(--down)}

header{margin-bottom:30px}
.date{font-family:"M PLUS 1 Code",monospace;font-size:12px;color:var(--ink3);margin:0 0 6px}
h1{font-size:clamp(26px,6vw,36px);font-weight:800;letter-spacing:-.025em;
margin:0;line-height:1.3}

section{margin:0 0 38px}
h2{font-size:13px;font-weight:700;color:var(--ink3);letter-spacing:.06em;
margin:0 0 12px;display:flex;align-items:center;gap:8px}
h2 .cnt{font-family:"M PLUS 1 Code",monospace;background:var(--soft);color:var(--ink2);
padding:1px 8px;border-radius:20px;font-size:12px}
.none{color:var(--ink3);font-size:13.5px;margin:10px 0 0}
.say{color:var(--ink2);font-size:15px;line-height:1.85;margin:0 0 16px;
padding-left:14px;border-left:3px solid var(--accent)}

/* 今日 */
.nothing{background:var(--card);border:1px solid var(--line);border-radius:16px;
padding:22px 24px}
.nothing .nh{font-size:22px;font-weight:800;margin-bottom:6px;letter-spacing:-.02em}
.nothing p{margin:0;color:var(--ink2);font-size:14.5px;line-height:1.85}
.cards{display:flex;flex-direction:column;gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;
padding:20px 22px;border-left:4px solid var(--line2)}
.card.buy{border-left-color:var(--up)}
.card.sell{border-left-color:var(--down)}
.ch{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:10px}
.tag{font-size:11px;font-weight:700;padding:3px 9px;border-radius:6px;
font-family:"M PLUS 1 Code",monospace}
.tag.buy{background:var(--up-bg);color:var(--up)}
.tag.sell{background:var(--down-bg);color:var(--down)}
.tk{font-size:20px;font-weight:700;letter-spacing:-.02em}
.pr{font-size:15px;color:var(--ink2)}
.pl{margin-left:auto;font-size:15px;font-weight:600}
.cw{margin:0 0 12px;font-size:14.5px;color:var(--ink2);line-height:1.85}
.cf{font-family:"M PLUS 1 Code",monospace;font-size:12.5px;color:var(--ink3);
padding-top:11px;border-top:1px solid var(--line)}

/* 保有 */
.holds{display:flex;flex-direction:column;gap:8px}
.hold{background:var(--card);border:1px solid var(--line);border-radius:13px;
padding:14px 18px;display:grid;grid-template-columns:1fr auto;gap:4px 12px;align-items:baseline}
.hold.warn{border-color:var(--down);background:var(--down-bg)}
.hl{display:flex;align-items:baseline;gap:10px}
.hl .tk{font-size:17px}
.hd{font-size:12px;color:var(--ink3)}
.hr{text-align:right}
.hp{font-size:17px;font-weight:700;display:block}
.hs{font-size:11.5px;color:var(--ink3);font-family:"M PLUS 1 Code",monospace}
.hf{grid-column:1/-1;font-size:12.5px;color:var(--down);font-weight:600;
padding-top:8px;border-top:1px solid var(--line)}

/* 相場・成績 */
.chip{font-family:"M PLUS 1 Code",monospace;font-size:11px;font-weight:700;
padding:3px 9px;border-radius:6px;background:var(--accent-bg);color:var(--accent);
margin-right:4px;white-space:nowrap}
.chip.順行{background:var(--up-bg);color:var(--up)}
.chip.混乱,.chip.調整{background:var(--down-bg);color:var(--down)}
.chip.選別{background:var(--warn-bg);color:var(--warn)}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:12px 13px}
.tl{font-size:10.5px;color:var(--ink3);white-space:nowrap;overflow:hidden;
text-overflow:ellipsis;margin-bottom:2px}
.tv{font-size:18px;font-weight:600;letter-spacing:-.02em;line-height:1.3}
.td{font-size:12px;font-family:"M PLUS 1 Code",monospace;margin-top:1px}
.big{margin-bottom:14px}
.bn{font-size:44px;font-weight:800;letter-spacing:-.04em;line-height:1.05}
.bs{font-family:"M PLUS 1 Code",monospace;font-size:13px;color:var(--ink3);margin-top:4px}
.chart{width:100%;height:auto;display:block;background:var(--card);
border:1px solid var(--line);border-radius:13px;margin-top:14px;padding:6px 0}
.grid{stroke:var(--line);stroke-width:1}
.baseline{stroke:var(--line2);stroke-width:1;stroke-dasharray:4 4}
.tick{fill:var(--ink3);font-size:10.5px;font-family:"M PLUS 1 Code",monospace}
.line.up{stroke:var(--up)}.line.down{stroke:var(--down)}
.area.up{fill:var(--up);opacity:.09}.area.down{fill:var(--down);opacity:.09}
.dot.up{fill:var(--up)}.dot.down{fill:var(--down)}

/* 詳細（畳んである） */
details{background:var(--card);border:1px solid var(--line);border-radius:13px;
margin-bottom:8px;overflow:hidden}
summary{cursor:pointer;list-style:none;padding:14px 18px;font-weight:600;font-size:14px;
display:flex;gap:10px;align-items:center;color:var(--ink2)}
summary::-webkit-details-marker{display:none}
summary::before{content:"›";font-size:17px;color:var(--accent);
transition:transform .18s;display:inline-block}
details[open] summary::before{transform:rotate(90deg)}
summary:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.db{padding:0 18px 18px}
.nws{display:flex;flex-direction:column;gap:14px}
.nw{padding-bottom:14px;border-bottom:1px solid var(--line)}
.nw:last-child{border-bottom:none;padding-bottom:0}
.nw .nh{display:flex;gap:8px;align-items:baseline;margin-bottom:3px}
.nw .tk{font-size:12.5px}
.lv{font-size:10px;font-weight:700;padding:2px 7px;border-radius:5px;
font-family:"M PLUS 1 Code",monospace}
.lv.強気{background:var(--up-bg);color:var(--up)}
.lv.弱気{background:var(--down-bg);color:var(--down)}
.lv.中立,.lv.軽微{background:var(--soft);color:var(--ink3)}
.nd{font-size:10.5px;color:var(--ink3);margin-left:auto}
.nt{font-size:13.5px;line-height:1.65}
.nv{font-size:12.5px;color:var(--ink2);margin-top:5px;padding-left:11px;
border-left:2px solid var(--line2);line-height:1.7}
.tw{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:320px}
th{text-align:left;padding:7px 10px 7px 0;font-size:10.5px;color:var(--ink3);
font-family:"M PLUS 1 Code",monospace;font-weight:600;border-bottom:1px solid var(--line);
white-space:nowrap}
th.r,td.r{text-align:right}
td{padding:9px 10px 9px 0;border-bottom:1px solid var(--line);white-space:nowrap}
tr:last-child td{border-bottom:none}
.form{font-family:"M PLUS 1 Code",monospace;font-size:10.5px;background:var(--accent-bg);
color:var(--accent);padding:2px 7px;border-radius:5px;font-weight:600}
.sectorlist{display:flex;flex-direction:column}
.secrow{display:grid;grid-template-columns:1fr 110px 58px;gap:10px;align-items:center;
padding:7px 0;border-bottom:1px solid var(--line);font-size:13px}
.secrow:last-child{border-bottom:none}
.secname{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.secbar{position:relative;height:12px;background:var(--soft);border-radius:3px}
.secbar > i{position:absolute;top:2px;height:8px;border-radius:2px}
.secbar > i.up{background:var(--up)}.secbar > i.down{background:var(--down)}
.secbar .mid{position:absolute;left:50%;top:0;width:1px;height:12px;background:var(--line2)}
.secval{font-family:"M PLUS 1 Code",monospace;font-size:12px;text-align:right}
.secma{display:none}

.note b{display:block;margin-bottom:4px}
.note div{font-weight:400;line-height:1.7}
.rks{display:flex;flex-direction:column;gap:7px}
.rk{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--warn);
border-radius:11px;padding:11px 15px;display:grid;grid-template-columns:auto auto 1fr;
gap:10px;align-items:baseline;font-size:13.5px}
.rt{font-size:10.5px;font-weight:700;padding:2px 8px;border-radius:5px;background:var(--warn-bg);
color:var(--warn);font-family:"M PLUS 1 Code",monospace;white-space:nowrap}
.rs{font-family:"M PLUS 1 Code",monospace;font-weight:700;white-space:nowrap}
.rx{color:var(--ink2);line-height:1.7}
td.sm{white-space:normal;font-size:12px;color:var(--ink3);max-width:260px}
.world{margin-top:14px;background:var(--card);border:1px solid var(--line);
border-radius:12px;padding:14px 16px}
.wh{font-size:11px;color:var(--ink3);font-weight:700;letter-spacing:.06em;margin-bottom:8px}
.wr{display:flex;gap:10px;align-items:baseline;padding:5px 0;font-size:13.5px;line-height:1.7}
.wr+.wr{border-top:1px solid var(--line)}
.wt{font-size:10px;font-weight:700;padding:2px 7px;border-radius:5px;background:var(--accent-bg);
color:var(--accent);font-family:"M PLUS 1 Code",monospace;white-space:nowrap;flex:none}
.wx{color:var(--ink2)}
.scg{margin-bottom:12px}
.scg:last-child{margin-bottom:0}
.scl{font-size:11px;color:var(--ink3);font-weight:700;margin-bottom:5px}
.scr{display:flex;flex-wrap:wrap;gap:6px}
.sc{display:inline-flex;gap:5px;align-items:baseline;background:var(--soft);
border-radius:7px;padding:3px 9px;font-family:"M PLUS 1 Code",monospace;font-size:11.5px}
.sc b{font-weight:700}.sc i{font-style:normal}.sc u{text-decoration:none;color:var(--ink3)}
footer{margin-top:44px;padding-top:18px;border-top:1px solid var(--line);
font-family:"M PLUS 1 Code",monospace;font-size:11px;color:var(--ink3);line-height:1.9}

@media (max-width:560px){
.page{padding:24px 15px 70px}
.tiles{grid-template-columns:repeat(2,1fr)}
.secrow{grid-template-columns:1fr 58px}
.secbar{display:none}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

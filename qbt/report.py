"""
バックテスト結果をHTMLレポートに書き出す。

外部ライブラリもCDNも使わず、SVGを自前で組み立てて1ファイルに閉じる。
ブラウザで開けばそのまま読めるし、そのまま人に渡せる。
"""
from __future__ import annotations

import html
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd

from . import metrics as M

# dataviz の検証済みパレット
PAL = {
    "s1": ("#2a78d6", "#3987e5"),   # 戦略
    "s2": ("#eb6834", "#d95926"),   # ベンチマーク
    "pos": ("#2a78d6", "#3987e5"),
    "neg": ("#d03b3b", "#d03b3b"),
    "good": "#0ca30c",
    "crit": "#d03b3b",
}

PCT = lambda x: f"{x*100:,.1f}%" if x is not None and np.isfinite(x) else "—"
NUM = lambda x, d=2: f"{x:,.{d}f}" if x is not None and np.isfinite(x) else "—"
MONEY = lambda x: f"{x:,.0f}" if x is not None and np.isfinite(x) else "—"


# ------------------------------------------------------------------ SVG部品

def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """軸ラベルをキリの良い数値に丸める（1/2/2.5/5 刻み）"""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return list(np.linspace(lo, hi, n))
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(abs(raw))) if raw > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if raw / mag <= m:
            step = m * mag
            break
    else:
        step = 10 * mag
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + step * 1e-9:
        out.append(round(v, 10))
        v += step
    return out or list(np.linspace(lo, hi, n))


def _line_chart(series: dict[str, pd.Series], height: int = 300, ylabel: str = "",
                y_pct: bool = False, fill_below: bool = False) -> str:
    """複数系列の折れ線。x軸は日付、ホバーで十字線とツールチップが出る。"""
    W, H = 1000, height
    ml, mr, mt, mb = 62, 18, 14, 30
    pw, ph = W - ml - mr, H - mt - mb

    all_idx = None
    for s in series.values():
        all_idx = s.index if all_idx is None else all_idx.union(s.index)
    all_idx = pd.DatetimeIndex(sorted(all_idx))
    if len(all_idx) < 2:
        return "<p>データが不足しています</p>"

    vals = np.concatenate([s.dropna().values for s in series.values() if len(s.dropna())])
    if not len(vals):
        return "<p>データが不足しています</p>"
    ymin, ymax = float(np.nanmin(vals)), float(np.nanmax(vals))
    pad = (ymax - ymin) * 0.06 or abs(ymax) * 0.06 or 1.0
    ymin, ymax = ymin - pad, ymax + pad
    if fill_below:
        ymax = 0.0 if float(np.nanmax(vals)) <= 0 else ymax

    t0, t1 = all_idx[0].value, all_idx[-1].value
    sx = lambda t: ml + (t - t0) / max(t1 - t0, 1) * pw
    sy = lambda v: mt + (ymax - v) / max(ymax - ymin, 1e-12) * ph

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" preserveAspectRatio="none">']

    # y軸グリッド（キリの良い値に丸める）
    ticks = _nice_ticks(ymin, ymax, 5)
    for tv in ticks:
        y = sy(tv)
        parts.append(f'<line x1="{ml}" x2="{W-mr}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        lab = PCT(tv) if y_pct else f"{tv:,.0f}"
        parts.append(f'<text x="{ml-8}" y="{y+4:.1f}" class="tick" text-anchor="end">{lab}</text>')
    if fill_below and ymin < 0 < ymax:
        parts.append(f'<line x1="{ml}" x2="{W-mr}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" class="baseline"/>')

    # x軸（年ラベル）
    years = sorted({d.year for d in all_idx})
    step = max(1, len(years) // 9)
    for yr in years[::step]:
        d = all_idx[all_idx.year == yr][0]
        x = sx(d.value)
        parts.append(f'<text x="{x:.1f}" y="{H-8}" class="tick" text-anchor="middle">{yr}</text>')

    colors = ["s1", "s2"]
    _label_ys: list[float] = []
    for i, (name, s) in enumerate(series.items()):
        s = s.dropna()
        if len(s) < 2:
            continue
        key = colors[i % len(colors)]
        pts = " ".join(f"{sx(ts.value):.1f},{sy(v):.1f}" for ts, v in s.items())
        if fill_below:
            first_x, last_x = sx(s.index[0].value), sx(s.index[-1].value)
            parts.append(f'<polygon points="{first_x:.1f},{sy(0):.1f} {pts} {last_x:.1f},{sy(0):.1f}" '
                         f'fill="var(--{key})" opacity="0.16"/>')
        parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--{key})" '
                     f'stroke-width="2" stroke-linejoin="round" class="ln"/>')
        # 系列名を線の終端に直接置く（凡例だけに頼らない）。
        # 終端が近いと重なるので、既に置いたラベルとの距離を見てずらす。
        ly = sy(s.iloc[-1]) - 8
        for prev in _label_ys:
            if abs(ly - prev) < 16:
                ly = prev + 18
        _label_ys.append(ly)
        ly = min(max(ly, mt + 12), mt + ph - 4)
        parts.append(f'<text x="{sx(s.index[-1].value)-4:.1f}" y="{ly:.1f}" '
                     f'class="endlab" fill="var(--{key})" text-anchor="end">{html.escape(name)}</text>')

    parts.append(f'<line class="crosshair" x1="0" x2="0" y1="{mt}" y2="{mt+ph}" style="display:none"/>')
    parts.append("</svg>")

    # ホバー用データ
    hover = {
        "x0": ml, "x1": W - mr, "t0": int(t0), "t1": int(t1),
        "series": {k: [[int(ts.value), float(v)] for ts, v in s.dropna().items()]
                   for k, s in series.items()},
        "pct": bool(y_pct),
    }
    cid = f"c{abs(hash(json.dumps(list(series.keys())) + str(height) + str(ymin)))%100000}"
    return (f'<div class="chart-wrap" data-chart=\'{html.escape(json.dumps(hover), quote=True)}\' id="{cid}">'
            + "".join(parts) + '<div class="tip"></div></div>')


def _monthly_heatmap(equity: pd.Series) -> str:
    """月次リターンのヒートマップ。青=プラス、赤=マイナス、灰色が中立。"""
    pv = M.monthly_returns(equity)
    if pv.empty:
        return "<p>データが不足しています</p>"
    vmax = float(np.nanmax(np.abs(pv.values))) or 0.01
    cell, lw = 62, 46
    W = lw + cell * 12 + 70
    H = 26 + cell * len(pv.index) * 0.55 + 10
    ch = cell * 0.55
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart heat" preserveAspectRatio="xMidYMid meet">']
    for m in range(1, 13):
        out.append(f'<text x="{lw + cell*(m-0.5):.0f}" y="16" class="tick" text-anchor="middle">{m}月</text>')
    out.append(f'<text x="{lw + cell*12 + 34:.0f}" y="16" class="tick" text-anchor="middle">年間</text>')
    for r, yr in enumerate(pv.index):
        y = 26 + r * ch
        out.append(f'<text x="{lw-10}" y="{y+ch*0.65:.0f}" class="tick" text-anchor="end">{yr}</text>')
        yearsum = 1.0
        for m in range(1, 13):
            v = pv.loc[yr, m] if m in pv.columns else np.nan
            x = lw + cell * (m - 1)
            if v is None or not np.isfinite(v):
                out.append(f'<rect x="{x}" y="{y:.0f}" width="{cell-2}" height="{ch-2:.0f}" class="cell-na"/>')
                continue
            yearsum *= (1 + v)
            a = min(abs(v) / vmax, 1.0) * 0.85 + 0.08
            col = "var(--pos)" if v >= 0 else "var(--neg)"
            out.append(f'<rect x="{x}" y="{y:.0f}" width="{cell-2}" height="{ch-2:.0f}" '
                       f'fill="{col}" opacity="{a:.2f}" rx="3"/>')
            # 薄い塗りの上では白文字が読めないので、濃さに応じて文字色を切り替える
            tcls = "cellv" if a >= 0.55 else "cellv dark"
            out.append(f'<text x="{x+(cell-2)/2:.0f}" y="{y+ch*0.62:.0f}" class="{tcls}" '
                       f'text-anchor="middle">{v*100:.1f}</text>')
        yv = yearsum - 1
        x = lw + cell * 12 + 4
        col = "var(--pos)" if yv >= 0 else "var(--neg)"
        out.append(f'<rect x="{x}" y="{y:.0f}" width="{62}" height="{ch-2:.0f}" fill="{col}" '
                   f'opacity="0.9" rx="3"/>')
        out.append(f'<text x="{x+31}" y="{y+ch*0.62:.0f}" class="cellv strong" '
                   f'text-anchor="middle">{yv*100:.1f}</text>')
    out.append("</svg>")
    return "".join(out)


def _tile(label: str, value: str, sub: str = "", tone: str = "") -> str:
    cls = f"tile {tone}".strip()
    return (f'<div class="{cls}"><div class="tl">{html.escape(label)}</div>'
            f'<div class="tv">{value}</div>'
            + (f'<div class="ts">{html.escape(sub)}</div>' if sub else "") + "</div>")


def _table(df: pd.DataFrame, max_rows: int | None = None, fmts: dict | None = None) -> str:
    if df is None or df.empty:
        return "<p class='muted'>該当データなし</p>"
    d = df.head(max_rows) if max_rows else df
    fmts = fmts or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns)
    rows = []
    for _, r in d.iterrows():
        tds = []
        for c in d.columns:
            v = r[c]
            f = fmts.get(c)
            if f:
                txt = f(v)
            elif isinstance(v, (float, np.floating)):
                txt = f"{v:,.3f}"
            elif isinstance(v, pd.Timestamp):
                txt = v.strftime("%Y-%m-%d")
            else:
                txt = str(v)
            neg = isinstance(v, (float, int, np.floating)) and not isinstance(v, bool) and v < 0
            tds.append(f'<td class="{"neg" if neg else ""}">{html.escape(txt)}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="tw"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


# ------------------------------------------------------------------ 本体

def build(results: dict, out_path: str, title: str = "バックテスト結果",
          config_text: str = "", extras: dict | None = None) -> str:
    """
    results: {"IS(開発期間)": Result, "OOS(検証期間)": Result, ...}
    extras:  {"walk_forward": {...}, "monte_carlo": {...}, "sweep": DataFrame, ...}
    """
    extras = extras or {}
    sections = []

    for name, res in results.items():
        eq, tr = res.equity, res.trades_df
        s = M.summary(eq, tr, res.exposure, res.benchmark)
        tone = "ok" if s["年率リターン(CAGR)"] > 0 else "bad"

        tiles = "".join([
            _tile("年率リターン", PCT(s["年率リターン(CAGR)"]),
                  f"累積 {PCT(s['累積リターン'])}", tone),
            _tile("最大ドローダウン", PCT(s["最大ドローダウン"]),
                  f"水面下 最長 {s['水面下最長期間(日)']}日", "bad"),
            _tile("シャープレシオ", NUM(s["シャープレシオ"]),
                  f"カルマー {NUM(s['カルマーレシオ'])}"),
            _tile("勝率", PCT(s["勝率"]),
                  f"{s['取引回数']}回 / PF {NUM(s['プロフィットファクター'])}"),
            _tile("1回あたり期待値", PCT(s["期待値(1回あたり%)"]),
                  f"平均保有 {NUM(s['平均保有日数'],1)}日"),
            _tile("t値", NUM(s["t値"]), s["統計的評価"],
                  "ok" if abs(s["t値"]) >= 2 else "warn"),
        ])

        curves = {"戦略": eq}
        if res.benchmark is not None:
            curves["ベンチマーク"] = res.benchmark

        dd = M.drawdown_series(eq)
        yearly = M.yearly_returns(eq).to_frame("リターン")
        yearly.index = [d.year for d in yearly.index]
        yearly = yearly.reset_index().rename(columns={"index": "年"})

        trades_view = pd.DataFrame()
        if not tr.empty:
            trades_view = tr.sort_values("exit_date", ascending=False).head(40)[
                ["symbol", "entry_date", "exit_date", "entry_price", "exit_price",
                 "shares", "pnl", "pnl_pct", "bars_held", "exit_reason"]
            ].rename(columns={
                "symbol": "銘柄", "entry_date": "建て日", "exit_date": "手仕舞い日",
                "entry_price": "建値", "exit_price": "決済値", "shares": "株数",
                "pnl": "損益", "pnl_pct": "損益率", "bars_held": "保有日数",
                "exit_reason": "決済理由"})

        reason_tbl = pd.DataFrame()
        if not tr.empty:
            g = tr.groupby("exit_reason").agg(
                件数=("pnl", "size"), 勝率=("pnl", lambda x: (x > 0).mean()),
                合計損益=("pnl", "sum"), 平均損益率=("pnl_pct", "mean")).reset_index()
            reason_tbl = g.rename(columns={"exit_reason": "決済理由"})

        sections.append(f"""
<section>
  <h2>{html.escape(name)}</h2>
  <p class="period muted">{eq.index[0].date()} 〜 {eq.index[-1].date()}
     初期資金 {MONEY(eq.iloc[0])} → 最終 <b>{MONEY(s['最終資産'])}</b></p>
  <div class="tiles">{tiles}</div>

  <h3>資産曲線</h3>
  <div class="legend"><span class="k" style="background:var(--s1)"></span>戦略
    {"<span class='k' style='background:var(--s2)'></span>ベンチマーク" if res.benchmark is not None else ""}</div>
  {_line_chart(curves, 320)}

  <h3>ドローダウン</h3>
  {_line_chart({"ドローダウン": dd}, 190, y_pct=True, fill_below=True)}

  <h3>月次リターン（%）</h3>
  {_monthly_heatmap(eq)}

  <div class="two">
    <div><h3>年次リターン</h3>{_table(yearly, fmts={"リターン": PCT})}</div>
    <div><h3>決済理由の内訳</h3>{_table(reason_tbl, fmts={"勝率": PCT, "平均損益率": PCT, "合計損益": MONEY})}</div>
  </div>

  <h3>直近の取引（最大40件）</h3>
  {_table(trades_view, fmts={"損益": MONEY, "損益率": PCT, "建値": lambda v: NUM(v,1), "決済値": lambda v: NUM(v,1)})}
</section>""")

    # ---- 追加検証 ----
    if extras.get("walk_forward") is not None:
        wf = extras["walk_forward"]
        deg = wf.get("degradation") or {}
        wf_chart = ""
        if wf.get("equity") is not None and len(wf["equity"]) > 2:
            wf_chart = _line_chart({"ウォークフォワード(検証期間のみ)": wf["equity"]}, 260)
        deg_txt = ""
        if deg.get("劣化率") is not None:
            d = deg["劣化率"]
            verdict = ("学習と検証の差が小さい。頑健性は高い" if d < 0.3 else
                       "検証期間で成績が半減以下。実運用では期待値を大きく割り引くこと" if d < 0.7 else
                       "検証期間でほぼ崩壊。この戦略は過学習している")
            deg_txt = (f"<p>学習期間の平均 <b>{NUM(deg['学習平均'])}</b> → "
                       f"検証期間の平均 <b>{NUM(deg['検証平均'])}</b>　"
                       f"劣化率 <b>{PCT(d)}</b><br><span class='verdict'>{html.escape(verdict)}</span></p>")
        sections.append(f"""
<section>
  <h2>ウォークフォワード検証</h2>
  <p class="muted">「直近N年で最適化して、その次の期間で運用する」を繰り返した結果。
     ここでの資産曲線は、一度も最適化に使っていない期間だけをつないだもの。</p>
  {deg_txt}
  {wf_chart}
  {_table(wf["folds"], fmts={"検証リターン": PCT, "検証最大DD": PCT})}
</section>""")

    if extras.get("monte_carlo"):
        mc = extras["monte_carlo"]
        if "判定" in mc:
            body = f"<p class='muted'>{html.escape(mc['判定'])}</p>"
        else:
            body = f"""<div class="tiles">
              {_tile("実績の最大DD", PCT(mc['実績最大DD']))}
              {_tile("想定DD 中央値", PCT(mc['想定最大DD(中央値)']))}
              {_tile("想定DD 95%タイル", PCT(mc['想定最大DD(95%タイル)']), "20回に1回はここまで落ちる", "warn")}
              {_tile("想定DD 最悪ケース", PCT(mc['想定最大DD(最悪)']), "", "bad")}
              {_tile("累積が負になる確率", PCT(mc['累積リターンが負になる確率']))}
            </div>"""
        sections.append(f"""
<section><h2>モンテカルロ（取引順序のシャッフル）</h2>
<p class="muted">同じ取引でも並び順が違えば資産曲線は変わる。実績のDDは運が良かっただけかもしれない。
   運用開始前に「覚悟すべきDD」を知るための数字。</p>{body}</section>""")

    if extras.get("sweep") is not None and len(extras["sweep"]):
        sw = extras["sweep"]
        sens = extras.get("sensitivity", {})
        sens_txt = ""
        if sens.get("判定"):
            sens_txt = (f"<p>最高 <b>{NUM(sens.get('最高'))}</b> / 中央値 <b>{NUM(sens.get('中央値'))}</b>　"
                        f"（比率 {PCT(sens.get('中央値/最高'))}）<br>"
                        f"<span class='verdict'>{html.escape(sens['判定'])}</span></p>")
        sections.append(f"""
<section><h2>パラメータ感度</h2>
<p class="muted">全組み合わせの成績一覧。ここで最高の行を選ぶのは過学習の入口。
   見るべきは「広い範囲でそこそこ勝てているか」。</p>
{sens_txt}
{_table(sw, max_rows=60, fmts={"年率リターン(CAGR)": PCT, "最大ドローダウン": PCT, "勝率": PCT})}
</section>""")

    cfg_block = (f'<section><h2>実行した設定</h2><pre class="cfg">{html.escape(config_text)}</pre></section>'
                 if config_text else "")

    return _shell(title, "".join(sections) + cfg_block, out_path)


def _shell(title: str, body: str, out_path: str) -> str:
    css = """
:root{color-scheme:light;--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
--muted:#898781;--grid:#e1e0d9;--base:#c3c2b7;--border:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--pos:#2a78d6;--neg:#d03b3b;--good:#0ca30c;--crit:#d03b3b;--warn:#fab219}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--base:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--pos:#3987e5}}
[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;
--grid:#2c2c2a;--base:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--pos:#3987e5}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
font-family:system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;line-height:1.7}
.page{max-width:1080px;margin:0 auto;padding:40px 22px 80px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:19px;margin:0 0 14px;padding-bottom:10px;border-bottom:1px solid var(--border)}
h3{font-size:14px;margin:26px 0 8px;color:var(--ink2);font-weight:600}
section{background:var(--surface);border:1px solid var(--border);border-radius:12px;
padding:22px 24px 26px;margin:22px 0}
.muted{color:var(--muted);font-size:13px}
.period{margin:-6px 0 16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:10px;margin:14px 0 4px}
.tile{background:var(--plane);border:1px solid var(--border);border-radius:9px;padding:11px 13px}
.tl{font-size:11px;color:var(--muted);letter-spacing:.02em}
.tv{font-size:21px;font-weight:650;margin:3px 0 1px;letter-spacing:-.02em}
.ts{font-size:11px;color:var(--ink2)}
.tile.ok .tv{color:var(--good)} .tile.bad .tv{color:var(--crit)} .tile.warn .tv{color:var(--ink)}
.chart-wrap{position:relative;margin:6px 0 4px}
.chart{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--grid);stroke-width:1}
.baseline{stroke:var(--base);stroke-width:1}
.tick{fill:var(--muted);font-size:11px;font-family:inherit;font-variant-numeric:tabular-nums}
.endlab{font-size:12px;font-weight:600;font-family:inherit}
.cellv{fill:#fff;font-size:10px;font-family:inherit;font-variant-numeric:tabular-nums}
.cellv.dark{fill:var(--ink2)}
.cellv.strong{font-weight:700}
.cell-na{fill:var(--grid);opacity:.35;rx:3}
.crosshair{stroke:var(--base);stroke-width:1;stroke-dasharray:3 3}
.tip{position:absolute;pointer-events:none;display:none;background:var(--surface);
border:1px solid var(--border);border-radius:7px;padding:7px 10px;font-size:12px;
box-shadow:0 4px 16px rgba(0,0,0,.14);white-space:nowrap;z-index:5}
.legend{display:flex;gap:16px;align-items:center;font-size:12px;color:var(--ink2);margin:2px 0 2px}
.legend .k{width:11px;height:11px;border-radius:3px;display:inline-block;margin-right:6px;
vertical-align:-1px}
.tw{overflow-x:auto;margin:6px 0;border:1px solid var(--border);border-radius:9px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th{background:var(--plane);text-align:left;padding:8px 11px;font-weight:600;color:var(--ink2);
white-space:nowrap;border-bottom:1px solid var(--border)}
td{padding:7px 11px;border-bottom:1px solid var(--border);white-space:nowrap;
font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
td.neg{color:var(--crit)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:22px}
@media(max-width:760px){.two{grid-template-columns:1fr}}
.verdict{font-weight:600}
.cfg{background:var(--plane);border:1px solid var(--border);border-radius:9px;padding:14px;
font-size:12px;overflow-x:auto;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;line-height:1.6}
.hd{font-size:12px;color:var(--muted);margin-bottom:26px}
"""
    js = """
document.querySelectorAll('.chart-wrap').forEach(w=>{
  let cfg; try{cfg=JSON.parse(w.dataset.chart)}catch(e){return}
  const svg=w.querySelector('svg'), tip=w.querySelector('.tip'),
        ch=w.querySelector('.crosshair'), keys=Object.keys(cfg.series);
  const fmt=v=>cfg.pct?(v*100).toFixed(2)+'%':v.toLocaleString(undefined,{maximumFractionDigits:0});
  function move(e){
    const r=svg.getBoundingClientRect();
    const px=(e.clientX-r.left)/r.width*1000;
    if(px<cfg.x0||px>cfg.x1){leave();return}
    const t=cfg.t0+(px-cfg.x0)/(cfg.x1-cfg.x0)*(cfg.t1-cfg.t0);
    let rows='';
    keys.forEach(k=>{
      const a=cfg.series[k]; if(!a.length)return;
      let lo=0,hi=a.length-1;
      while(lo<hi){const m=(lo+hi)>>1; a[m][0]<t?lo=m+1:hi=m}
      rows+='<div>'+k+' <b>'+fmt(a[lo][1])+'</b></div>';
      if(!rows.startsWith('<div style'))
        rows='<div style="color:var(--muted);margin-bottom:2px">'
             +new Date(a[lo][0]/1e6).toISOString().slice(0,10)+'</div>'+rows;
    });
    ch.setAttribute('x1',px);ch.setAttribute('x2',px);ch.style.display='';
    tip.innerHTML=rows;tip.style.display='block';
    const l=Math.min(Math.max(px/1000*r.width-60,0),r.width-140);
    tip.style.left=l+'px';tip.style.top='6px';
  }
  function leave(){tip.style.display='none';ch.style.display='none'}
  w.addEventListener('mousemove',move);w.addEventListener('mouseleave',leave);
});
"""
    doc = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{css}</style></head>
<body><div class="page">
<h1>{html.escape(title)}</h1>
<div class="hd">生成 {datetime.now().strftime('%Y-%m-%d %H:%M')}　/
数値はすべて手数料・スリッページ控除後。約定は翌営業日の寄付。</div>
{body}
</div><script>{js}</script></body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path

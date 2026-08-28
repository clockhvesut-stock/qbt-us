#!/usr/bin/env python3
"""
LINEにレポートを送る。

送る中身は2通り:
  1. outbox/{日付}_message.json があれば、それを送る（Claudeが解釈を書いたもの）
  2. なければ raw.json から機械的に要約を作って送る（フォールバック）

2があるので、AI側が動かなかった日でも最低限の情報は届く。
「何も来ない」が一番困るので、必ず何かを送る設計にしてある。

必要な環境変数（GitHubのSecretsに登録する）:
  LINE_CHANNEL_TOKEN : LINE Developers で発行するチャネルアクセストークン（長期）
  LINE_USER_ID       : 送信先のユーザーID（Uから始まる33文字）

使い方:
    python notify.py                # 今日のメッセージを送る
    python notify.py --dry-run      # 送らずに内容だけ表示
    python notify.py --test         # 疎通確認のテストメッセージを送る
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "reports")
OUTBOX = os.path.join(HERE, "outbox")

LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
MAX_CHARS = 4800          # LINEのテキスト上限は5000文字。余裕を持たせる
MAX_MESSAGES = 5          # 1リクエストで送れるメッセージ数


def pct(x, digits=1):
    try:
        return f"{float(x)*100:+.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------- 送信

def push(texts: list[str], token: str, user_id: str, dry: bool = False) -> bool:
    chunks = []
    for t in texts:
        while len(t) > MAX_CHARS:
            cut = t.rfind("\n", 0, MAX_CHARS)
            cut = cut if cut > MAX_CHARS * 0.5 else MAX_CHARS
            chunks.append(t[:cut])
            t = t[cut:].lstrip("\n")
        if t.strip():
            chunks.append(t)

    if dry:
        for i, c in enumerate(chunks, 1):
            print(f"----- メッセージ {i}/{len(chunks)} ({len(c)}文字) -----")
            print(c)
        return True

    import requests

    ok = True
    for i in range(0, len(chunks), MAX_MESSAGES):
        batch = chunks[i:i + MAX_MESSAGES]
        r = requests.post(
            LINE_PUSH_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"to": user_id, "messages": [{"type": "text", "text": c} for c in batch]},
            timeout=30)
        if r.status_code != 200:
            print(f"[エラー] LINE送信に失敗 ({r.status_code}): {r.text[:300]}")
            ok = False
        else:
            print(f"送信しました（{len(batch)}通）")
    return ok


# ---------------------------------------------------------------- フォールバック要約

def build_fallback(raw: dict) -> list[str]:
    """AI側の解釈が無い日に、機械的に作る要約"""
    r = raw.get("regime", {})
    macro = raw.get("macro", {})
    lines = [
        f"📊 {raw.get('date','')} 米国株レポート",
        f"（AIの解釈が未生成のため機械要約）",
        "",
        f"■ 相場局面: {r.get('regime','不明')}",
        f"{r.get('note','')}",
        "",
        "■ マクロ",
    ]
    for sym in ("^GSPC", "^VIX", "^TNX", "DX-Y.NYB", "CL=F", "GC=F"):
        m = macro.get(sym)
        if not m:
            continue
        lines.append(f"  {m['label']}: {m['last']} ({pct(m.get('chg_1d'))} / "
                     f"20日 {pct(m.get('chg_20d'))})")

    b = raw.get("breadth", {})
    if b:
        lines += ["", "■ 市場の内部",
                  f"  200日線超え: {pct(b.get('above_ma200_pct'),0)} / "
                  f"50日線超え: {pct(b.get('above_ma50_pct'),0)}"]

    pos = raw.get("positions", [])
    if pos:
        lines += ["", f"■ 保有 {len(pos)}銘柄"]
        for p in pos:
            mark = "⚠️" if p.get("exit_flags") else "　"
            lines.append(f"  {mark}{p['symbol']} {pct(p.get('pnl_pct'))} "
                         f"({p.get('bars_held')}日) {p.get('action','')}")
            for f in p.get("exit_flags", []):
                lines.append(f"     → {f}")

    sig = raw.get("track_a_signals", [])
    lines += ["", f"■ ルールベース候補 上位{min(8,len(sig))}銘柄"]
    for s in sig[:8]:
        lines.append(f"  {s['rank']}. {s['symbol']} ${s.get('price','—')} "
                     f"RSI{s.get('rsi14','—')} 21日{pct(s.get('chg_21d'))}")
    if not sig:
        lines.append("  該当なし")

    if raw.get("errors"):
        lines += ["", "■ エラー"] + [f"  {e}" for e in raw["errors"]]

    return ["\n".join(lines)]


# ---------------------------------------------------------------- 本体

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", action="store_true", help="疎通確認のみ")
    args = ap.parse_args()

    token = os.environ.get("LINE_CHANNEL_TOKEN", "")
    user_id = os.environ.get("LINE_USER_ID", "")

    if args.test:
        if not (token and user_id) and not args.dry_run:
            print("[エラー] LINE_CHANNEL_TOKEN と LINE_USER_ID を設定してください")
            return 1
        msg = ("✅ 接続テスト成功\n\n"
               "米国株の自動レポートがこのトークに届くようになりました。\n"
               f"送信時刻: {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M')}")
        return 0 if push([msg], token, user_id, args.dry_run) else 1

    if not (token and user_id) and not args.dry_run:
        print("[エラー] LINE_CHANNEL_TOKEN と LINE_USER_ID を設定してください")
        return 1

    date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1) Claudeが書いた解釈があればそれを使う
    msg_path = os.path.join(OUTBOX, f"{date}_message.json")
    texts = None
    if os.path.exists(msg_path):
        try:
            payload = json.load(open(msg_path, encoding="utf-8"))
            texts = payload.get("messages") or ([payload["text"]] if payload.get("text") else None)
            print(f"AIの解釈を使用: {msg_path}")
        except Exception as e:
            print(f"[警告] {msg_path} を読めませんでした: {e}")

    # 2) 無ければ機械要約にフォールバック
    if not texts:
        raw_path = os.path.join(REPORTS, f"{date}_raw.json")
        if not os.path.exists(raw_path):
            raw_path = os.path.join(REPORTS, "latest_raw.json")
        if not os.path.exists(raw_path):
            texts = [f"⚠️ {date}\nデータ収集が完了していません。"
                     f"GitHub Actions の実行ログを確認してください。"]
        else:
            raw = json.load(open(raw_path, encoding="utf-8"))
            texts = build_fallback(raw)
            print("機械要約にフォールバックしました")

    return 0 if push(texts, token, user_id, args.dry_run) else 1


if __name__ == "__main__":
    sys.exit(main())

"""
ペーパートレードの執行と記録。

バックテストと「まったく同じ約定ルール」で、1日ぶんだけ前に進める。
ここがずれると、検証で見た数字と実際の記録が比較できなくなり、
何ヶ月走らせても何も分からないままになる。

  バックテスト : 過去の全期間を一気に回す
  ペーパー     : 毎日1歩ずつ、同じ処理を進める

約定の約束ごとは engine.py と同一:
  1. シグナルは「その日の終値まで」の情報で計算し、注文は翌営業日の寄付で執行する
  2. 手数料とスリッページを往復で必ず引く
  3. 現金がなければ買えない。同時保有数の上限も守る
  4. 逆指値・利確はザラ場の高値安値で判定し、当日中に約定させる
  5. 損切りと利確が同日に当たったら、保守的に損切りを優先する

状態ファイル:
  state/pending.json   翌営業日の寄付で執行する注文
  state/positions.json 保有中の建玉
  state/trades.json    決済済みの取引
  state/account.json   現金残高
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def _read(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def _size(budget: float, price: float, lot: int, fractional: bool) -> float:
    """購入株数を決める。engine.py の Portfolio.size_shares と同じ計算"""
    if price <= 0:
        return 0.0
    if fractional:
        return round(math.floor(budget / price / 0.0001) * 0.0001, 6)
    return float(int(budget // (price * lot)) * lot)


class PaperBook:
    """
    1つの運用記録。ルールベース用とAI判断用で別々に持てるようにしてある。
    どちらが優れるかを同じ条件で比べるため、口座を分ける。
    """

    def __init__(self, state_dir: str, cfg: dict, name: str = "rule"):
        self.name = name
        self.dir = state_dir
        self.cfg = cfg
        pf = cfg.get("portfolio", {}) or {}
        co = cfg.get("costs", {}) or {}
        self.initial = float(pf.get("initial_cash", 2000))
        self.max_pos = int(pf.get("max_positions", 8))
        self.weight = pf.get("position_pct") or (1.0 / self.max_pos)
        self.lot = int(pf.get("lot_size", 1))
        self.fractional = bool(pf.get("allow_fractional", True))
        self.slip = float(co.get("slippage_bps", 8.0)) / 10000.0
        self.comm = float(co.get("commission_bps", 0.0)) / 10000.0
        self.min_comm = float(co.get("min_commission", 0.0))

        sfx = "" if name == "rule" else f"_{name}"
        self.p_pending = os.path.join(state_dir, f"pending{sfx}.json")
        self.p_positions = os.path.join(state_dir, f"positions{sfx}.json")
        self.p_trades = os.path.join(state_dir, f"trades{sfx}.json")
        self.p_account = os.path.join(state_dir, f"account{sfx}.json")

        self.pending = _read(self.p_pending, [])
        self.positions = _read(self.p_positions, [])
        self.trades = _read(self.p_trades, [])
        acct = _read(self.p_account, {})
        self.cash = float(acct.get("cash", self.initial))
        self.initial = float(acct.get("initial_cash", self.initial))
        self.log: list[str] = []

    # ------------------------------------------------------------------

    def fee(self, notional: float) -> float:
        return max(abs(notional) * self.comm, self.min_comm)

    def _close(self, pos: dict, price: float, date: str, reason: str):
        shares = float(pos["shares"])
        notional = shares * price
        fee = self.fee(notional)
        cost = shares * float(pos["entry_price"]) + float(pos.get("entry_fee", 0.0))
        pnl = notional - fee - cost
        self.cash += notional - fee
        self.trades.append({
            "symbol": pos["symbol"],
            "entry_date": pos["entry_date"], "exit_date": date,
            "entry_price": round(float(pos["entry_price"]), 4),
            "exit_price": round(price, 4),
            "shares": shares,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / cost, 4) if cost else 0.0,
            "bars_held": int(pos.get("bars_held", 0)),
            "exit_reason": reason,
            "book": self.name,
        })
        self.log.append(
            f"決済 {pos['symbol']} {price:.2f} ({pnl/cost*100:+.1f}%) — {reason}")

    # ------------------------------------------------------------------

    def step(self, date: str, bars: dict[str, dict], rules: dict) -> dict:
        """
        1営業日ぶん進める。

        bars: {シンボル: {"open":…, "high":…, "low":…, "close":…}}
        """
        stop_pct = rules.get("stop_loss_pct")
        take_pct = rules.get("take_profit_pct")
        trail_pct = rules.get("trail_stop_pct")
        max_hold = rules.get("max_hold_days")

        # 同じ日を二度処理しないための番人。
        # Actionsの再実行や手動実行で建玉が二重にできるのを防ぐ。
        last = _read(self.p_account, {}).get("last_step_date")
        if last == date:
            self.log.append(f"{date} はすでに処理済みのため、何もしません")
            return self.snapshot(bars, skipped=True)

        # ---- 1) 前日に決めた手仕舞いを、今日の寄付で執行 ----
        still = []
        for pos in self.positions:
            b = bars.get(pos["symbol"])
            if pos.get("exit_queued") and b and np.isfinite(b.get("open", np.nan)):
                self._close(pos, float(b["open"]) * (1 - self.slip), date,
                            pos.get("exit_reason") or "シグナル")
            else:
                pos.pop("exit_queued", None)
                still.append(pos)
        self.positions = still

        # ---- 2) 前日に決めた新規建てを、今日の寄付で執行 ----
        equity_before = self.cash + sum(
            float(p["shares"]) * float((bars.get(p["symbol"]) or {}).get("close")
                                       or p["entry_price"]) for p in self.positions)
        held = {p["symbol"] for p in self.positions}
        for order in self.pending:
            sym = order.get("symbol")
            if sym in held or len(self.positions) >= self.max_pos:
                continue
            b = bars.get(sym)
            if not b or not np.isfinite(b.get("open", np.nan)) or b["open"] <= 0:
                self.log.append(f"見送り {sym} — 寄付値が取得できませんでした")
                continue
            px = float(b["open"]) * (1 + self.slip)
            budget = min(equity_before * self.weight, self.cash)
            shares = _size(budget, px, self.lot, self.fractional)
            if shares <= 0:
                self.log.append(f"見送り {sym} — 資金不足（予算 ${budget:.0f} / 株価 ${px:.2f}）")
                continue
            notional = shares * px
            fee = self.fee(notional)
            if notional + fee > self.cash:
                shares = _size(self.cash * 0.999, px, self.lot, self.fractional)
                if shares <= 0:
                    continue
                notional = shares * px
                fee = self.fee(notional)
            self.cash -= notional + fee
            self.positions.append({
                "symbol": sym, "shares": shares,
                "entry_price": round(px, 4), "entry_date": date,
                "entry_fee": round(fee, 4), "bars_held": 0,
                "high_water": float(b.get("high") or px),
                "stop_price": round(px * (1 - stop_pct), 4) if stop_pct else None,
                "reason": order.get("reason", ""),
            })
            held.add(sym)
            self.log.append(f"新規 {sym} {px:.2f} × {shares:.4f}株 (${notional:.0f})")
        self.pending = []

        # ---- 3) ザラ場の逆指値・利確 ----
        still = []
        for pos in self.positions:
            b = bars.get(pos["symbol"]) or {}
            lo, hi = b.get("low"), b.get("high")
            if hi is not None and np.isfinite(hi):
                pos["high_water"] = max(float(pos.get("high_water", 0)), float(hi))
            hit, reason = None, ""
            ep = float(pos["entry_price"])
            if lo is not None and np.isfinite(lo):
                if stop_pct and float(lo) <= ep * (1 - stop_pct):
                    hit, reason = ep * (1 - stop_pct), "損切り"
                if hit is None and trail_pct:
                    ts = float(pos["high_water"]) * (1 - trail_pct)
                    if float(lo) <= ts:
                        hit, reason = ts, "トレーリングストップ"
            # 損切りと利確が同日なら損切りを優先する（保守的な側を採る）
            if hit is None and take_pct and hi is not None and np.isfinite(hi):
                if float(hi) >= ep * (1 + take_pct):
                    hit, reason = ep * (1 + take_pct), "利確"
            if hit is not None:
                self._close(pos, float(hit) * (1 - self.slip), date, reason)
            else:
                still.append(pos)
        self.positions = still

        # ---- 4) 大引け: 保有日数を進めて時価評価 ----
        for pos in self.positions:
            pos["bars_held"] = int(pos.get("bars_held", 0)) + 1

        return self.snapshot(bars)

    # ------------------------------------------------------------------

    def queue_exits(self, bars: dict, rules: dict, exit_flags: dict[str, list]):
        """明日の寄付で手仕舞う建玉に印をつける"""
        max_hold = rules.get("max_hold_days")
        for pos in self.positions:
            flags = list(exit_flags.get(pos["symbol"], []))
            if max_hold and int(pos.get("bars_held", 0)) >= int(max_hold):
                flags.append("期間満了")
            if flags:
                pos["exit_queued"] = True
                pos["exit_reason"] = flags[0]

    def queue_entries(self, candidates: list[dict], market_ok: bool = True):
        """
        明日の寄付で建てる注文を決める。
        すでに保有している銘柄と、枠を超える分は除く。
        """
        if not market_ok:
            self.pending = []
            self.log.append("相場フィルタが不通過のため、新規建ては見送ります")
            return
        held = {p["symbol"] for p in self.positions}
        leaving = sum(1 for p in self.positions if p.get("exit_queued"))
        slots = self.max_pos - len(self.positions) + leaving
        picked = []
        for c in candidates:
            if len(picked) >= slots:
                break
            sym = c.get("symbol")
            if sym and sym not in held:
                picked.append({"symbol": sym, "rank": c.get("rank"),
                               "reason": c.get("reason", "ルールのシグナル")})
        self.pending = picked

    # ------------------------------------------------------------------

    def snapshot(self, bars: dict, skipped: bool = False) -> dict:
        mv = 0.0
        rows = []
        for p in self.positions:
            b = bars.get(p["symbol"]) or {}
            px = float(b.get("close") or p["entry_price"])
            mv += float(p["shares"]) * px
            ep = float(p["entry_price"])
            rows.append({
                "symbol": p["symbol"], "shares": float(p["shares"]),
                "entry_price": ep, "entry_date": p["entry_date"],
                "price": round(px, 2), "pnl_pct": round(px / ep - 1, 4),
                "bars_held": int(p.get("bars_held", 0)),
                "stop_price": p.get("stop_price"),
                "exit_flags": (["翌営業日の寄付で手仕舞い予定"]
                               if p.get("exit_queued") else []),
                "action": "手仕舞い予定" if p.get("exit_queued") else "継続保有",
                "reason": p.get("reason", ""),
            })
        equity = self.cash + mv
        wins = [t for t in self.trades if t["pnl_pct"] > 0]
        streak = worst = 0
        for t in self.trades:
            streak = streak + 1 if t["pnl_pct"] <= 0 else 0
            worst = max(worst, streak)
        return {
            "book": self.name,
            "cash": round(self.cash, 2),
            "market_value": round(mv, 2),
            "equity": round(equity, 2),
            "initial_cash": self.initial,
            "return_pct": round(equity / self.initial - 1, 4) if self.initial else 0.0,
            "positions": rows,
            "pending": self.pending,
            "trades_total": len(self.trades),
            "win_rate": round(len(wins) / len(self.trades), 4) if self.trades else 0.0,
            "losing_streak": worst,
            "skipped": skipped,
            "log": self.log,
        }

    def save(self, date: str):
        _write(self.p_pending, self.pending)
        _write(self.p_positions, self.positions)
        _write(self.p_trades, self.trades[-500:])
        _write(self.p_account, {
            "cash": round(self.cash, 2),
            "initial_cash": self.initial,
            "last_step_date": date,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

"""
バックテストの中核エンジン。

設計上の約束ごと（ここが崩れると検証結果は全部ウソになる）:

 1. 未来を見ない
    シグナルは「その日の終値まで」の情報で計算し、約定は「翌営業日の寄付」で行う。
    終値でシグナルが出て同じ日の終値で約定する、という実運用不可能な仮定は使わない。

 2. コストを最初から入れる
    手数料とスリッページを往復で必ず引く。ゼロコスト前提の綺麗な資産曲線は意味がない。

 3. 資金制約を守る
    現金がなければ買えない。同時保有数の上限も守る。
    候補が枠より多いときは rank 式の順に採用する。

 4. 単元株に丸める
    日本株は100株単位。端数株で計算した非現実的な建玉を作らない。
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import INDICATOR_NAMESPACE


# ==================================================================== 式の評価

class ExpressionError(Exception):
    pass


class _BoolToBitwise(ast.NodeTransformer):
    """
    and/or/not を pandas 用の &/|/~ に変換する。

    単純な文字列置換ではダメ。`a > b & c > d` は演算子の優先順位の都合で
    `a > (b & c) > d` と解釈されてしまう。構文木の段階で変換して括弧を保証する。
    """

    def visit_BoolOp(self, node: ast.BoolOp):  # noqa: N802
        self.generic_visit(node)
        op = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
        out = node.values[0]
        for right in node.values[1:]:
            out = ast.BinOp(left=out, op=op, right=right)
        return ast.copy_location(out, node)

    def visit_UnaryOp(self, node: ast.UnaryOp):  # noqa: N802
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.copy_location(ast.UnaryOp(op=ast.Invert(), operand=node.operand), node)
        return node


def _compile_expr(expr: str):
    """式を構文木経由で安全にコンパイルする（結果はキャッシュされる）"""
    if expr in _EXPR_CACHE:
        return _EXPR_CACHE[expr]
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"式の文法エラー: {expr!r}\n  → {e}") from e

    for node in ast.walk(tree):
        # 属性アクセスと import 系は禁止（名前空間の外に出さない）
        if isinstance(node, (ast.Attribute, ast.Import, ast.ImportFrom, ast.Lambda)):
            raise ExpressionError(f"式の中では使えない構文です: {type(node).__name__}")
    tree = _BoolToBitwise().visit(tree)
    ast.fix_missing_locations(tree)
    code = compile(tree, "<strategy>", "eval")
    _EXPR_CACHE[expr] = code
    return code


_EXPR_CACHE: dict[str, object] = {}


# 指標計算の使い回し。
#
# パラメータ探索では同じ rsi(close,14) を何百回も計算することになる。
# 銘柄・関数・列・引数が同じなら結果も同じなので、一度だけ計算して覚えておく。
# これがないと528銘柄×360通りの探索が現実的な時間で終わらない。
#
# 前提: 同じ memo_key に対してデータが変わらないこと。
#       別のデータセットに移るときは clear_memo() を呼ぶ。
_MEMO: dict = {}
_MEMO_LIMIT = 200_000


def clear_memo():
    """データセットを切り替えるときに呼ぶ。呼び忘れると古い計算結果を使ってしまう"""
    _MEMO.clear()


def _memoized(fn, name: str, df: pd.DataFrame):
    """
    鍵は「銘柄コード」ではなく「DataFrameそのものの同一性」にする。

    銘柄コードを鍵にすると、別のデータセットに同じコードの銘柄があったとき、
    前のデータの計算結果を使ってしまう。テストで実際に踏んだ。
    値と一緒に DataFrame への参照も持つことで、
    ガベージコレクションによる id の使い回しも防ぐ。
    """
    def inner(*args, **kw):
        try:
            cols = tuple(getattr(a, "name", None) for a in args if isinstance(a, pd.Series))
            rest = tuple(a for a in args if not isinstance(a, pd.Series))
            k = (id(df), name, cols, rest, tuple(sorted(kw.items())))
            hash(k)
        except TypeError:
            return fn(*args, **kw)      # 鍵にできない引数が来たら素通し
        hit = _MEMO.get(k)
        if hit is not None:
            return hit[1]
        v = fn(*args, **kw)
        if len(_MEMO) < _MEMO_LIMIT:
            _MEMO[k] = (df, v)          # dfを掴んでおくと id が再利用されない
        return v
    return inner


def eval_expr(expr: str, df: pd.DataFrame, extra: dict | None = None,
              memo_key: str | None = None) -> pd.Series:
    """
    戦略ファイルに書かれた1行の式を、1銘柄のOHLCVに対して評価する。

    式の中では open/high/low/close/volume と、indicators.py の関数が使える。
    例: "close > sma(close, 200) and rsi(close, 14) < 30"

    memo_key に銘柄コードを渡すと、指標の計算結果を使い回す。
    パラメータ探索のように同じ指標を何度も計算する場面で効く。
    """
    if memo_key:
        ns = {n: (_memoized(f, n, df) if callable(f) else f)
              for n, f in INDICATOR_NAMESPACE.items()}
    else:
        ns = dict(INDICATOR_NAMESPACE)
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            ns[c] = df[c]
    if {"high", "low", "close"} <= set(df.columns):
        ns["typical"] = (df["high"] + df["low"] + df["close"]) / 3.0
    if extra:
        ns.update(extra)

    code = _compile_expr(expr)
    try:
        out = eval(code, {"__builtins__": {}}, ns)  # noqa: S307 — 名前空間を白リストで制限済み
    except Exception as e:
        raise ExpressionError(f"式の評価に失敗しました: {expr!r}\n  → {e}") from e

    if np.isscalar(out) or isinstance(out, (bool, int, float)):
        out = pd.Series(out, index=df.index)
    return pd.Series(out, index=df.index)


def build_signal_frame(expr: str | None, data: dict[str, pd.DataFrame],
                       calendar: pd.DatetimeIndex, boolean: bool = True) -> pd.DataFrame:
    """全銘柄に対して式を評価し、日付×銘柄 の行列にまとめる"""
    if not expr:
        return pd.DataFrame(False if boolean else np.nan, index=calendar,
                            columns=list(data.keys()))
    cols = {}
    for sym, df in data.items():
        s = eval_expr(expr, df, memo_key=sym)
        if boolean:
            s = s.fillna(False).astype(bool)
        cols[sym] = s.reindex(calendar)
    out = pd.DataFrame(cols, index=calendar)
    return out.fillna(False).astype(bool) if boolean else out


# ==================================================================== 設定

@dataclass
class Costs:
    commission_bps: float = 5.0     # 片道手数料（bps。5.0 = 0.05%）
    slippage_bps: float = 10.0      # 片道スリッページ（bps）
    min_commission: float = 0.0     # 最低手数料（円/ドル）

    def buy_price(self, px: float) -> float:
        return px * (1.0 + self.slippage_bps / 10000.0)

    def sell_price(self, px: float) -> float:
        return px * (1.0 - self.slippage_bps / 10000.0)

    def fee(self, notional: float) -> float:
        return max(abs(notional) * self.commission_bps / 10000.0, self.min_commission)


@dataclass
class Rules:
    entry: str                       # 買いシグナルの条件式
    exit: str | None = None          # 手仕舞い条件式
    rank: str | None = None          # 候補が枠より多いときの優先順位（大きい順）
    rank_ascending: bool = False     # Trueなら小さい順を優先
    universe_filter: str | None = None   # 流動性フィルタ等。Falseの銘柄は売買対象外
    market_filter: str | None = None     # 相場全体のフィルタ（ベンチマークに対して評価）
    stop_loss_pct: float | None = None   # 逆指値（0.08 = 建値から-8%）
    take_profit_pct: float | None = None # 利確（0.20 = +20%）
    trail_stop_pct: float | None = None  # トレーリングストップ
    max_hold_days: int | None = None     # 最大保有日数（時間切れ手仕舞い）


@dataclass
class Portfolio:
    initial_cash: float = 3_000_000.0
    max_positions: int = 10
    position_pct: float | None = None   # 1銘柄あたりの資金比率。Noneなら 1/max_positions
    lot_size: int = 100                 # 単元株数（日本株=100, 米国株=1）
    allow_partial: bool = True          # 資金不足時に買える分だけ買うか

    # 端株（フラクショナル株）を許可するか。
    # 少額運用では必須に近い。2,000ドルを8銘柄に分けると1銘柄250ドルなので、
    # 株価300ドルの銘柄は1株も買えず、ユニバースが勝手に低位株に偏ってしまう。
    # moomoo・IBKR・Alpaca はいずれも米国株の端株売買に対応している。
    allow_fractional: bool = False
    min_fraction: float = 0.0001        # 端株の最小単位

    def size_shares(self, budget: float, price: float) -> float:
        """予算と株価から購入株数を決める"""
        if price <= 0:
            return 0.0
        if self.allow_fractional:
            n = math.floor(budget / price / self.min_fraction) * self.min_fraction
            return round(n, 6)
        return float(int(budget // (price * self.lot_size)) * self.lot_size)


@dataclass
class Position:
    symbol: str
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    high_water: float = 0.0
    bars_held: int = 0


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    fees: float
    bars_held: int
    exit_reason: str


@dataclass
class Result:
    equity: pd.Series = field(default_factory=pd.Series)
    trades: list[Trade] = field(default_factory=list)
    positions_count: pd.Series = field(default_factory=pd.Series)
    exposure: pd.Series = field(default_factory=pd.Series)
    benchmark: pd.Series | None = None
    label: str = ""
    skipped_unaffordable: int = 0

    @property
    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=[f.name for f in Trade.__dataclass_fields__.values()])
        return pd.DataFrame([t.__dict__ for t in self.trades])


# ==================================================================== エンジン本体

class Backtest:
    def __init__(self, data: dict[str, pd.DataFrame], rules: Rules,
                 portfolio: Portfolio, costs: Costs,
                 benchmark: pd.DataFrame | None = None):
        self.data = data
        self.rules = rules
        self.pf = portfolio
        self.costs = costs
        self.benchmark = benchmark
        self.calendar = self._calendar()

    def _calendar(self) -> pd.DatetimeIndex:
        idx = None
        for df in self.data.values():
            idx = df.index if idx is None else idx.union(df.index)
        return pd.DatetimeIndex(sorted(idx))

    # ------------------------------------------------------------------

    def run(self, start: str | None = None, end: str | None = None, label: str = "") -> Result:
        cal = self.calendar
        if start:
            cal = cal[cal >= pd.Timestamp(start)]
        if end:
            cal = cal[cal <= pd.Timestamp(end)]

        # --- 価格行列とシグナル行列を先に作る（ベクトル化できる部分は全部ここで） ---
        px_open = pd.DataFrame({s: d["open"] for s, d in self.data.items()}).reindex(cal)
        px_high = pd.DataFrame({s: d["high"] for s, d in self.data.items()}).reindex(cal)
        px_low = pd.DataFrame({s: d["low"] for s, d in self.data.items()}).reindex(cal)
        px_close = pd.DataFrame({s: d["close"] for s, d in self.data.items()}).reindex(cal)

        entry_sig = build_signal_frame(self.rules.entry, self.data, cal)
        exit_sig = build_signal_frame(self.rules.exit, self.data, cal) if self.rules.exit else None
        uni_ok = (build_signal_frame(self.rules.universe_filter, self.data, cal)
                  if self.rules.universe_filter else None)
        rank_val = (build_signal_frame(self.rules.rank, self.data, cal, boolean=False)
                    if self.rules.rank else None)

        # 相場全体のフィルタはベンチマーク（TOPIX/S&P500等）に対して評価する
        market_ok = None
        if self.rules.market_filter:
            if self.benchmark is None:
                raise ValueError("market_filter を使うには benchmark の指定が必要です")
            market_ok = eval_expr(self.rules.market_filter, self.benchmark
                                  ).reindex(cal).ffill().fillna(False).astype(bool)

        cash = self.pf.initial_cash
        positions: dict[str, Position] = {}
        trades: list[Trade] = []
        eq_vals, npos_vals, expo_vals = [], [], []

        pending_entries: list[str] = []
        pending_exits: list[tuple[str, str]] = []
        # 「シグナルは出たが資金が足りず1株も買えなかった」回数。
        # これが多いなら、同時保有数を減らすか端株を有効にする必要がある。
        skipped_unaffordable = 0

        weight = self.pf.position_pct if self.pf.position_pct else 1.0 / self.pf.max_positions

        for i, date in enumerate(cal):
            o = px_open.iloc[i]
            h = px_high.iloc[i]
            lo = px_low.iloc[i]
            c = px_close.iloc[i]

            # ---------- 1) 前日に決まった手仕舞いを、今日の寄付で執行 ----------
            for sym, reason in pending_exits:
                if sym not in positions or not np.isfinite(o.get(sym, np.nan)):
                    continue
                cash += self._close_position(positions, sym, self.costs.sell_price(float(o[sym])),
                                             date, reason, trades)
            pending_exits = []

            # ---------- 2) 前日に決まった新規建てを、今日の寄付で執行 ----------
            equity_prev = cash + sum(p.shares * float(c.get(p.symbol, p.entry_price))
                                     for p in positions.values())
            for sym in pending_entries:
                if len(positions) >= self.pf.max_positions or sym in positions:
                    continue
                price_raw = o.get(sym, np.nan)
                if not np.isfinite(price_raw) or price_raw <= 0:
                    continue
                px = self.costs.buy_price(float(price_raw))
                budget = min(equity_prev * weight, cash)
                shares = self.pf.size_shares(budget, px)
                if shares <= 0:
                    skipped_unaffordable += 1
                    continue
                notional = shares * px
                fee = self.costs.fee(notional)
                if notional + fee > cash:
                    if not self.pf.allow_partial:
                        continue
                    shares = self.pf.size_shares(cash * 0.999, px)
                    if shares <= 0:
                        continue
                    notional = shares * px
                    fee = self.costs.fee(notional)
                cash -= notional + fee
                positions[sym] = Position(sym, shares, px, date, high_water=px)
                positions[sym]._entry_fee = fee  # type: ignore[attr-defined]
            pending_entries = []

            # ---------- 3) 日中の逆指値・利確判定（ザラ場で当たったものは当日約定） ----------
            for sym in list(positions.keys()):
                p = positions[sym]
                hi_v, lo_v = h.get(sym, np.nan), lo.get(sym, np.nan)
                if not np.isfinite(lo_v):
                    continue
                p.high_water = max(p.high_water, float(hi_v) if np.isfinite(hi_v) else p.high_water)

                stop_px = None
                reason = ""
                if self.rules.stop_loss_pct:
                    sl = p.entry_price * (1 - self.rules.stop_loss_pct)
                    if lo_v <= sl:
                        stop_px, reason = sl, "損切り"
                if stop_px is None and self.rules.trail_stop_pct:
                    ts = p.high_water * (1 - self.rules.trail_stop_pct)
                    if lo_v <= ts and ts > p.entry_price * 0.0:
                        stop_px, reason = ts, "トレーリングストップ"
                # 損切りと利確が同日に両方当たった場合は、保守的に損切りを優先する
                if stop_px is None and self.rules.take_profit_pct and np.isfinite(hi_v):
                    tp = p.entry_price * (1 + self.rules.take_profit_pct)
                    if hi_v >= tp:
                        stop_px, reason = tp, "利確"

                if stop_px is not None:
                    cash += self._close_position(positions, sym, self.costs.sell_price(float(stop_px)),
                                                 date, reason, trades)

            # ---------- 4) 大引け: 時価評価して資産曲線に記録 ----------
            mv = 0.0
            for p in positions.values():
                px_c = c.get(p.symbol, np.nan)
                mv += p.shares * (float(px_c) if np.isfinite(px_c) else p.entry_price)
                p.bars_held += 1
            equity = cash + mv
            eq_vals.append(equity)
            npos_vals.append(len(positions))
            expo_vals.append(mv / equity if equity > 0 else 0.0)

            # ---------- 5) 大引け: 明日の注文を決める（ここで使えるのは今日までの情報だけ） ----------
            if i + 1 >= len(cal):
                continue

            for sym, p in positions.items():
                want_exit = bool(exit_sig.iloc[i].get(sym, False)) if exit_sig is not None else False
                if self.rules.max_hold_days and p.bars_held >= self.rules.max_hold_days:
                    want_exit, r = True, "期間満了"
                else:
                    r = "シグナル"
                if want_exit:
                    pending_exits.append((sym, r))

            slots = self.pf.max_positions - len(positions) + len(pending_exits)
            if slots > 0 and (market_ok is None or bool(market_ok.iloc[i])):
                row = entry_sig.iloc[i]
                cands = [s for s in row.index
                         if bool(row.get(s, False)) and s not in positions
                         and np.isfinite(px_close.iloc[i].get(s, np.nan))]
                if uni_ok is not None:
                    cands = [s for s in cands if bool(uni_ok.iloc[i].get(s, False))]
                if rank_val is not None and cands:
                    rv = rank_val.iloc[i]
                    cands = sorted(cands,
                                   key=lambda s: (float(rv.get(s, np.nan))
                                                  if np.isfinite(rv.get(s, np.nan))
                                                  else (np.inf if self.rules.rank_ascending else -np.inf)),
                                   reverse=not self.rules.rank_ascending)
                pending_entries = cands[:slots]

        eq = pd.Series(eq_vals, index=cal, name="equity")
        bench = None
        if self.benchmark is not None:
            b = self.benchmark["close"].reindex(cal).ffill()
            bench = b / b.iloc[0] * self.pf.initial_cash
        if skipped_unaffordable > 0:
            print(f"    [注意] 資金不足で見送ったシグナルが {skipped_unaffordable} 回ありました。"
                  f"同時保有数を減らすか allow_fractional を有効にしてください")

        return Result(equity=eq, trades=trades,
                      positions_count=pd.Series(npos_vals, index=cal),
                      exposure=pd.Series(expo_vals, index=cal),
                      benchmark=bench, label=label,
                      skipped_unaffordable=skipped_unaffordable)

    # ------------------------------------------------------------------

    def _close_position(self, positions: dict[str, Position], sym: str, px: float,
                        date: pd.Timestamp, reason: str, trades: list[Trade]) -> float:
        p = positions.pop(sym)
        notional = p.shares * px
        fee = self.costs.fee(notional)
        entry_fee = getattr(p, "_entry_fee", 0.0)
        pnl = notional - fee - (p.shares * p.entry_price + entry_fee)
        trades.append(Trade(
            symbol=sym, entry_date=p.entry_date, exit_date=date,
            entry_price=p.entry_price, exit_price=px, shares=p.shares,
            pnl=pnl, pnl_pct=pnl / (p.shares * p.entry_price),
            fees=fee + entry_fee, bars_held=p.bars_held, exit_reason=reason,
        ))
        return notional - fee

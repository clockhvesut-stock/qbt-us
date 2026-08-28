#!/usr/bin/env python3
"""
売買対象の銘柄リストを作る。

    python build_universe.py            # S&P500 + ETF + 中型株（流動性フィルタつき）
    python build_universe.py --quick    # S&P500 + ETF のみ（数秒で終わる）

中型株のフィルタは全上場6000銘柄の出来高を見るので10〜20分かかる。
一度作れば四半期に1回作り直す程度で十分。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qbt import universe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="流動性フィルタを飛ばし、S&P500とETFだけで作る")
    ap.add_argument("--min-dollar-volume", type=float, default=20_000_000,
                    help="1日の平均売買代金の下限（ドル）")
    ap.add_argument("--min-price", type=float, default=5.0)
    ap.add_argument("--max-midcap", type=int, default=500)
    args = ap.parse_args()

    include = ("sp500", "etf") if args.quick else ("sp500", "midcap", "etf")
    print(f"ユニバースを構築します: {' + '.join(include)}")
    df = universe.build(
        include=include,
        min_dollar_volume=args.min_dollar_volume,
        min_price=args.min_price,
        max_midcap=args.max_midcap,
    )
    print("\n内訳:")
    print(df.groupby("group").size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

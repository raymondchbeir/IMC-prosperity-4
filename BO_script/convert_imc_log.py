"""Convert an IMC Prosperity submission log (.log JSON) into the
prices_*.csv / trades_*.csv layout that rust_backtester consumes.

The IMC log is JSON with two relevant top-level keys:
  - activitiesLog : str — already a semicolon-CSV with the exact columns
                          rust_backtester expects.  Written verbatim.
  - tradeHistory  : list[dict] — converted to a semicolon-CSV.  Trades where
                                 buyer or seller == "SUBMISSION" are filtered
                                 out (they are the previous bot's own fills;
                                 a fresh backtest will generate its own).

Output filenames follow rust_backtester's pattern:
    prices_round_<R>_day_<D>.csv
    trades_round_<R>_day_<D>_nn.csv

Default round=0, day=0 to match the EVAL sentinel in rust_backtest_wrapper.py.

Usage:
    python convert_imc_log.py path/to/513705.log -o data/round4_eval_100k
    # then point EVAL_DATA_PATH at data/round4_eval_100k
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TRADES_HEADER = ["timestamp", "buyer", "seller", "symbol", "currency", "price", "quantity"]


def convert(log_path: Path, out_dir: Path, round_num: int, day_num: int,
            include_submission_trades: bool = False) -> None:
    obj = json.loads(log_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- prices file: activitiesLog is already CSV in the right shape -------
    activities_text = obj["activitiesLog"].rstrip("\n")
    # Optionally rewrite the day column so it matches the chosen day_num
    # (the rust binary looks at filename, not the column, but keeps things tidy).
    lines = activities_text.split("\n")
    header, *rows = lines
    rewritten = [header]
    for row in rows:
        parts = row.split(";", 2)            # only split off "day;timestamp;..."
        if len(parts) >= 2:
            parts[0] = str(day_num)
            rewritten.append(";".join(parts))
        else:
            rewritten.append(row)
    prices_path = out_dir / f"prices_round_{round_num}_day_{day_num}.csv"
    prices_path.write_text("\n".join(rewritten) + "\n")

    # --- trades file: convert JSON list to semicolon-CSV --------------------
    trades_path = out_dir / f"trades_round_{round_num}_day_{day_num}.csv"
    n_kept = n_dropped = 0
    with trades_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(TRADES_HEADER)
        for tr in obj["tradeHistory"]:
            buyer  = tr.get("buyer")  or ""
            seller = tr.get("seller") or ""
            if not include_submission_trades and (buyer == "SUBMISSION" or seller == "SUBMISSION"):
                n_dropped += 1
                continue
            w.writerow([
                tr["timestamp"], buyer, seller, tr["symbol"],
                tr.get("currency") or "XIRECS",
                tr["price"], tr["quantity"],
            ])
            n_kept += 1

    products = sorted({r.split(";", 3)[2] for r in rows if ";" in r})
    timestamps = sorted({int(r.split(";", 2)[1]) for r in rows if ";" in r})
    print(f"wrote {prices_path}")
    print(f"  rows         : {len(rows)}")
    print(f"  timestamps   : {len(timestamps)}  ({timestamps[0]}..{timestamps[-1]})")
    print(f"  products ({len(products)}): {', '.join(products)}")
    print(f"wrote {trades_path}")
    print(f"  trades kept  : {n_kept}")
    print(f"  SUBMISSION dropped: {n_dropped}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, help="Path to the IMC submission .log file")
    ap.add_argument("-o", "--out-dir", type=Path, required=True,
                    help="Directory to write the converted CSVs into")
    ap.add_argument("--round", dest="round_num", type=int, default=0,
                    help="Round number to use in filenames (default: 0)")
    ap.add_argument("--day", dest="day_num", type=int, default=0,
                    help="Day number to use in filenames (default: 0)")
    ap.add_argument("--include-submission-trades", action="store_true",
                    help="Keep SUBMISSION-side trades.  Off by default — those "
                         "are the previous bot's own fills, not market flow.")
    args = ap.parse_args()
    convert(args.log, args.out_dir, args.round_num, args.day_num,
            include_submission_trades=args.include_submission_trades)

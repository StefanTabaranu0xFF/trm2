#!/usr/bin/env python3
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen, Request


BASE_URL = "https://api.binance.com/api/v3/klines"


def fetch_klines(symbol: str, interval: str, limit: int) -> list:
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }
    url = f"{BASE_URL}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "trm2-fetch/1.0"})
    with urlopen(req, timeout=30) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)


def to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def write_csv(klines: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("timestamp,open,high,low,close,volume\n")
        for row in klines:
            ts = int(row[0])
            f.write(
                f"{ts},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Binance OHLCV candle data and write CSV."
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol.")
    parser.add_argument("--interval", default="1h", help="Kline interval, e.g. 1h.")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of candles to fetch (max 1000).",
    )
    parser.add_argument(
        "--output",
        default="binance_ohlcv.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if args.limit < 1 or args.limit > 1000:
        print("Limit must be between 1 and 1000.", file=sys.stderr)
        return 1

    try:
        klines = fetch_klines(args.symbol, args.interval, args.limit)
    except Exception as exc:
        print(f"Failed to fetch klines: {exc}", file=sys.stderr)
        return 1

    if not klines:
        print("No klines returned.", file=sys.stderr)
        return 1

    write_csv(klines, args.output)
    first_ts = int(klines[0][0])
    last_ts = int(klines[-1][0])
    print(
        f"Wrote {len(klines)} candles for {args.symbol} "
        f"({to_iso(first_ts)} -> {to_iso(last_ts)}) to {args.output}"
    )
    time.sleep(0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

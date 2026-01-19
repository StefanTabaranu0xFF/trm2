#!/usr/bin/env python3
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen, Request


BASE_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000
INTERVAL_MS = {
    "1s": 1000,
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
    "1M": 30 * 24 * 60 * 60_000,
}


def fetch_klines(
    symbol: str,
    interval: str,
    limit: int,
    start_time: int | None = None,
    end_time: int | None = None,
) -> list:
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time
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


def parse_time_arg(value: str | None) -> int | None:
    if value is None:
        return None
    if value.isdigit():
        return int(value)
    cleaned = value.replace("Z", "+00:00")
    return int(datetime.fromisoformat(cleaned).timestamp() * 1000)


def collect_klines(
    symbol: str,
    interval: str,
    total: int,
    start_ms: int | None,
    end_ms: int | None,
) -> list:
    interval_ms = INTERVAL_MS.get(interval)
    if interval_ms is None:
        raise ValueError(f"Unsupported interval {interval}.")

    candles: dict[int, list] = {}
    remaining = total
    backward = start_ms is None and remaining > MAX_LIMIT
    if backward and end_ms is None:
        end_ms = None

    while remaining > 0:
        limit = min(MAX_LIMIT, remaining)
        if backward:
            batch = fetch_klines(
                symbol, interval, limit, start_time=None, end_time=end_ms
            )
        else:
            batch = fetch_klines(
                symbol, interval, limit, start_time=start_ms, end_time=end_ms
            )

        if not batch:
            break

        for row in batch:
            candles[int(row[0])] = row

        remaining = total - len(candles)
        if backward:
            earliest = min(int(row[0]) for row in batch)
            next_end = earliest - interval_ms
            if end_ms is not None and next_end >= end_ms:
                break
            end_ms = next_end
        else:
            latest = max(int(row[0]) for row in batch)
            start_ms = latest + interval_ms
            if end_ms is not None and start_ms > end_ms:
                break

    sorted_rows = [candles[k] for k in sorted(candles.keys())]
    if total and len(sorted_rows) > total:
        sorted_rows = sorted_rows[-total:]
    return sorted_rows


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
        "--total",
        type=int,
        default=1000,
        help="Total number of candles to fetch (can exceed 1000).",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start time (ms since epoch or ISO-8601).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End time (ms since epoch or ISO-8601).",
    )
    parser.add_argument(
        "--output",
        default="binance_ohlcv.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if args.limit < 1 or args.limit > MAX_LIMIT:
        print("Limit must be between 1 and 1000.", file=sys.stderr)
        return 1
    if args.total < 1:
        print("Total must be >= 1.", file=sys.stderr)
        return 1

    start_ms = parse_time_arg(args.start)
    end_ms = parse_time_arg(args.end)

    try:
        if args.total > args.limit:
            klines = collect_klines(
                args.symbol,
                args.interval,
                args.total,
                start_ms,
                end_ms,
            )
        else:
            klines = fetch_klines(
                args.symbol,
                args.interval,
                args.limit,
                start_time=start_ms,
                end_time=end_ms,
            )
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

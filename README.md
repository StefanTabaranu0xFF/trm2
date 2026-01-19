# trm2

## Fetch Binance candle data

```bash
python3 scripts/fetch_binance_ohlcv.py --symbol BTCUSDT --interval 1h --limit 1000 --output binance_ohlcv.csv
```

## Train on real OHLCV CSV

```bash
make
./trm binance_ohlcv.csv
```

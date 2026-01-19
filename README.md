# trm2

## Fetch Binance candle data

```bash
python3 scripts/fetch_binance_ohlcv.py --symbol BTCUSDT --interval 1h --total 5000 --output binance_ohlcv.csv
```

## Train on real OHLCV CSV

```bash
make
python3 scripts/fetch_binance_ohlcv.py --symbol BTCUSDT --interval 1h --total 5000 --output binance_ohlcv.csv
./trm binance_ohlcv.csv
```

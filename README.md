# trm2

## Fetch Binance candle data

```bash
python3 scripts/fetch_binance_ohlcv.py --symbol BTCUSDT --interval 1h --total 5000 --output binance_ohlcv.csv
```

## Train on real OHLCV CSV

Second argument is the number of MLP layers on top of the recursive block (0-4).
Use `--save` to write model weights and `--load` to resume from a saved model.
Use `--eval` to skip training and only evaluate a loaded model.

```bash
make
./trm binance_ohlcv.csv 2 --save model.bin
./trm binance_ohlcv.csv 2 --load model.bin --save model.bin
./trm binance_ohlcv.csv 2 --load model.bin --eval
```

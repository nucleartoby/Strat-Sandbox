# Strat-Sandbox

Notebooks for building and backtesting classic technical trading strategies across FX, equities, and crypto.

## Overview

Each folder implements and tests a specific indicator or strategy, moving average crossovers, MACD, RSI, Bollinger Band variants, and the Puell Multiple (a Bitcoin on-chain valuation metric)  against real price data (tested so far on GBP/USD, SPY, and BTC) to see how they perform out of sample rather than just in theory.

## Contents

- **Moving Average** — MA, MA crossover, and MACD strategies, including a Bollinger Band variant, backtested on GBP/USD and SPY
- **Relative Strength Index** — RSI and RSI + Bollinger Band strategies
- **Puell Multiple** — calculator for the Puell Multiple, used as a cyclical valuation signal for Bitcoin

## Approach

Each notebook pulls historical price data, computes the relevant indicator, defines entry/exit rules, and plots the resulting strategy performance against a buy-and-hold baseline.

## Stack

Python (Jupyter), pandas, NumPy, matplotlib.

## Setup

```bash
git clone https://github.com/nucleartoby/Strat-Sandbox.git
cd Strat-Sandbox
pip install -r requirements.txt
jupyter notebook
```

## Notes

This is an exploratory sandbox rather than a production backtesting framework. Each notebook is self-contained, so results and assumptions (fees, slippage, position sizing) vary by strategy and are documented inline.

## License

MIT
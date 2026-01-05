# Machine Learning for Trading Signals  
_Logistic Regression & Support Vector Machines_

This project demonstrates a **discriminative machine learning pipeline** for financial return classification,
combining **technical indicators**, **time-series aware cross-validation**, and **strategy backtesting**.

The repository is adapted from a coursework project and rewritten as a **standalone, reproducible ML system**
suitable for portfolio and research presentation.

---

## Problem setup

- Asset: **Microsoft (MSFT)** daily data
- Task: classify future returns into **long / neutral / short** signals
- Labels:
  - `1`   → return ≥ 1%
  - `0`   → −1% < return < 1%
  - `-1`  → return ≤ −1%

---

## Feature engineering

Technical indicators are computed using **lagged prices** only:

- Bollinger Bands (20)
- RSI (14)
- MACD (12, 26, 9)
- Momentum (12)
- OBV
- ATR
- CCI
- Lagged OHLC prices
- Log returns

All missing values are dropped after feature construction.

---

## Models

### Logistic Regression
- Solver: `liblinear`
- Hyperparameters:
  - `penalty`: L1, L2
  - `C`: {0.1, 0.5, 1, 2, 3, 4, 5, 10}
- Cross-validation:
  - `TimeSeriesSplit`
  - 5 splits

### Support Vector Machine
- Kernels: `linear`, `rbf`
- Hyperparameters:
  - `C`: {0.3, 0.5, 1, 5, 10, 20, 30, 50, 100}
- Cross-validation:
  - `TimeSeriesSplit`
  - Randomized search
- Scoring: **macro F1**

---

## Evaluation

- Train / test split:
  - Training: all observations except last 30 days
  - Test: last 30 days
- Metrics:
  - Accuracy
  - Precision
  - Recall
- Strategy comparison:
  - Logistic Regression strategy
  - SVM strategy
  - Buy-and-hold benchmark

Cumulative returns are plotted for direct comparison.

---

## Repository structure

```
.
├── src/
├── data/
│   └── sample/
├── reports/
│   └── figures/
├── notebook_model_experiments.ipynb
├── requirements.txt
└── README.md
```

---

## How to run

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open and run:

```
notebook_model_experiments.ipynb
```

---

## Key learning outcomes

- Time-series–aware ML validation
- Feature engineering from financial data
- Model comparison under different scoring rules
- Translating classification outputs into trading strategies

---

## Notes on data

Market data is downloaded programmatically using `yfinance`.  
No proprietary datasets are redistributed.

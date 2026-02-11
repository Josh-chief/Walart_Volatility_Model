
#  Walmart Volatility Modeling Using GARCH(1,1)

##  Project Overview

This project implements a complete **volatility modeling pipeline** for Walmart Inc. (WMT) using high-frequency (hourly) financial data and a **GARCH(1,1)** framework.

The objective is to:

* Model time-varying volatility
* Detect and confirm ARCH effects
* Perform rigorous pre- and post-estimation diagnostics
* Produce interpretable volatility dynamics
* Save reproducible outputs for further analysis

The implementation follows industry-standard quantitative finance methodology used in risk management, algorithmic trading, and financial econometrics.

---

##  Data Description

* **Asset:** Walmart Inc. (Ticker: WMT)
* **Frequency:** Hourly
* **Period:** 2025-02-10 to 2026-02-10
* **Source:** Yahoo Finance (via `yfinance` API)

Log returns are computed as:

[
r_t = \log\left(\frac{P_t}{P_{t-1}}\right)
]

This transformation ensures:

* Stationarity (under mild assumptions)
* Additive return structure
* Statistical suitability for volatility modeling

---

##  Methodological Framework

The project follows a structured econometric workflow:

---

### 1️ Data Collection & Return Construction

* Download hourly price data
* Save raw dataset (`wmt_data.csv`)
* Compute log returns
* Generate descriptive statistics

---

### 2️ Pre-Fit Diagnostics

Before estimating a GARCH model, we verify key assumptions:

#### ✔ ADF Test (Stationarity)

* Confirms returns are stationary
* Required for GARCH estimation validity

#### ✔ ACF of Squared Returns

* Visual detection of volatility clustering
* Persistence in squared returns indicates ARCH effects

#### ✔ Engle’s ARCH-LM Test

* Formal test for conditional heteroskedasticity
* Significant result justifies GARCH modeling

---

### 3️ GARCH(1,1) Model Specification

The model estimated:

[
r_t = \epsilon_t
]

[
\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
]

Where:

* ( \omega ) = Long-run variance component
* ( \alpha ) = Short-run shock response
* ( \beta ) = Volatility persistence

The model assumes:

* Zero conditional mean
* Normally distributed innovations
* Volatility clustering dynamics

GARCH(1,1) is widely recognized as the industry benchmark for volatility modeling.

---

### 4️ Post-Fit Diagnostics

After estimation, residual diagnostics ensure model adequacy:

#### ✔ Ljung-Box Test (Standardized Residuals)

* Tests for remaining serial correlation

#### ✔ Ljung-Box Test (Squared Residuals)

* Tests for remaining ARCH effects
* Most critical validation step

A well-specified GARCH model should remove autocorrelation in squared residuals.

---

### 5️ Volatility Visualization

The conditional volatility series is plotted and saved:

* `WMT_Conditional_Volatility.png`

This visualizes:

* Volatility clustering
* Shock persistence
* Periods of market stress

---

## Project Structure

```
├── Walmart.py
├── wmt_data.csv
├── ACF_Squared_Returns_WMT.png
├── WMT_Conditional_Volatility.png
├── garch_full_results.txt
└── README.md
```

---

## Installation & Requirements

Install required packages:

```bash
pip install yfinance pandas numpy matplotlib statsmodels arch
```

---

## How to Run

```bash
python Walmart.py
```

Outputs generated:

* Raw data CSV
* Diagnostic plots
* Volatility plot
* Full model results text file

---

## Why GARCH(1,1)?

GARCH(1,1) is preferred because:

* It captures volatility clustering
* It models persistence efficiently
* It is parsimonious
* It performs robustly across financial assets

In practice, over 70% of applied financial volatility models begin with GARCH(1,1) as a baseline.

---

## Key Insights This Model Can Provide

* Volatility persistence measurement
* Risk regime identification
* Risk forecasting foundation
* Basis for Value-at-Risk (VaR) estimation
* Input for portfolio optimization

---

## Potential Extensions

This project can be extended by:

* GARCH with Student-t distribution
* EGARCH (asymmetry modeling)
* GJR-GARCH (leverage effects)
* ARMA-GARCH mean specification
* Multi-asset volatility modeling
* Out-of-sample volatility forecasting
* VaR backtesting

---

## Academic & Professional Relevance

This project demonstrates competence in:

* Financial econometrics
* Time series diagnostics
* Risk modeling
* Quantitative Python implementation
* Reproducible research practices

It is suitable for:

* MSc Data Science portfolios
* Quant finance applications
* Research publications
* Risk analytics demonstrations

---

## Conclusion

This project provides a rigorous, end-to-end volatility modeling pipeline for Walmart stock using hourly data and a GARCH(1,1) framework.

Through structured diagnostics and validation, it ensures statistical validity and professional robustness,aligning with best practices in financial risk modeling.

---

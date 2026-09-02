# Weather Trend Forecasting — PM Accelerator Technical Assessment

## PM Accelerator Mission

> *[Paste the current PM Accelerator mission statement here, copied verbatim from their official LinkedIn page (linkedin.com/company/product-manager-accelerator) or website — the assessment asks you to source this yourself so it's accurate and current.]*

## Project Overview

This project analyzes the **[Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)** dataset from Kaggle to forecast global weather trends, completing both the basic and advanced tiers of the assessment: data cleaning, exploratory data analysis, forecasting model development, anomaly detection, and a multi-model ensemble.

- **Dataset:** 162,939 records × 41 features, covering 211 countries and 268 locations, updated daily from May 2024 onward
- **Target:** Global average daily temperature (°C)
- **Tools:** Python — pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, XGBoost

## Repository Structure

```
├── README.md
├── requirements.txt
├── data/
│   └── GlobalWeatherRepository.csv        # not committed — see Setup below
├── notebooks/
│   ├── 01_cleaning_eda.ipynb
│   ├── 02_basic_forecasting_model.ipynb
│   ├── 03_anomaly_detection.ipynb
│   └── 04_ensemble_models.ipynb
└── charts/                                 # exported figures used in the report
```
*(Adjust this section to match your actual file/folder names.)*

## Setup

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

Download `GlobalWeatherRepository.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository) and place it in `data/`. The raw CSV is not committed to the repo due to size and because it updates daily — anyone running this later will see slightly different recent dates than what's reported below.

## Methodology & Results

### 1. Data Cleaning & Preprocessing

- **Missing values:** none found (0 of 162,939 × 41 cells) — plausible given the dataset is scraped live from a weather API on a schedule rather than hand-entered.
- **Corrupted readings:** a min/max sanity check (not caught by the missing-value check) surfaced 4 physically impossible single-row readings: a 79.3°C temperature (Fiji), a 1,841 mph wind speed (Burundi), and two ~3,000 mb pressure readings (Honduras, Iran). All four were dropped.
- **False positives avoided:** an initial low-pressure outlier flag (770–870 mb, 14 rows) turned out to be mostly legitimate — those readings came from high-altitude cities (Addis Ababa, Mexico City, Nairobi, Kigali, Guatemala City, Windhoek), where lower barometric pressure is expected physics, not an error. Only 2 of the 14 flagged rows were genuine corruption.
- **Precipitation** is heavily right-skewed (mostly 0mm with a long tail); a `log1p` transform is applied before any distance-based modeling.

### 2. Exploratory Data Analysis

- Temperature is roughly normally distributed around 24–26°C.
- A clear annual seasonal cycle is visible in global average temperature (peak ~June–July, trough ~December–January).
- Core correlations align with physical expectation: humidity vs. UV index (−0.54), temperature vs. UV index (+0.48), humidity vs. cloud cover (+0.50).
- Daily averages are unreliable on days with very few readings (e.g. one day had a single reading for the entire globe) — a minimum sample-size threshold (≥50 readings/day) is applied before any day is used in time-series analysis.

### 3. Basic Forecasting Model

Linear Regression using a linear trend term plus annual Fourier (sin/cos) seasonal terms, evaluated against a naive persistence baseline on a 60-day temporal holdout (never randomly shuffled — order matters for time series).

| Model | MAE (°C) | RMSE (°C) | MAPE |
|---|---|---|---|
| Naive persistence (baseline) | 0.644 | 0.991 | 2.87% |
| Linear Regression + seasonality | 0.503 | 0.879 | 2.32% |

### 4. Anomaly Detection

Multivariate **Isolation Forest** across 7 features (temperature, humidity, wind, pressure, precipitation, cloud cover, UV index), flagging the top 1% as anomalous.

- Most flagged readings are genuine severe-weather signatures (monsoon conditions in Dhaka/Hanoi, North Atlantic storms in Iceland/Ireland/Denmark) — physically coherent combinations, not noise.
- One date, **2026-08-17**, stands out sharply: a 14.4% anomaly rate vs. a ~1% baseline average (nearly 3× the next-worst day). Investigation showed multiple Northern Hemisphere cities reporting winter-like conditions in the middle of August (e.g. Baghdad dropping from 40.6°C to 17.3°C week-over-week, London and Dublin recording "Blizzard"). This is treated as a data pipeline artifact for that date and excluded from subsequent time-series modeling.

### 5. Multiple Models + Ensemble

Four models built on the same cleaned daily series (2026-08-17 excluded): Linear Regression, SARIMAX (seasonal terms as exogenous regressors), XGBoost (seasonal features), and a lag-based "momentum" model using recursive multi-step forecasting.

| Model | MAE (°C) | RMSE (°C) | MAPE |
|---|---|---|---|
| **Linear Regression + seasonal** | **0.366** | **0.429** | **1.58%** |
| SARIMAX | 0.434 | 0.526 | 1.90% |
| XGBoost (seasonal) | 0.447 | 0.526 | 1.94% |
| Momentum (lag-based, recursive) | 0.410 | 0.462 | 1.78% |
| Ensemble (4-model average) | 0.397 | 0.461 | 1.73% |

**Finding:** the ensemble does not outperform the single best model. Error-correlation analysis showed all four models' errors were highly correlated (0.79–0.95) — they all key off the same dominant seasonal signal, so averaging them pulls the best model's prediction toward the weaker ones rather than cancelling error. Ensembling adds value in proportion to model *diversity*, and this problem doesn't have much diversity left to extract once seasonality is accounted for.

## Key Insights

1. "Zero missing values" is not the same as "clean data" — disguised corruption (physically impossible single-row readings) required a separate physical-plausibility check to catch.
2. Statistical outlier rules need domain context before acting on them — geography (altitude) explained most of what looked like pressure anomalies.
3. A documented, isolated data-quality event (2026-08-17) was traced from an unexplained dip in a forecasting chart to a specific, evidenced root cause via anomaly detection.
4. More models ≠ better forecasts. An ensemble is only as good as the diversity between its members; this project shows a case where a single well-specified model outperformed a 4-model ensemble, and explains why.

## Limitations & Future Work

- Unique advanced analyses (long-term regional climate patterns, air-quality/weather correlation, formal feature-importance analysis, and spatial/geographic pattern visualization) are natural next steps beyond what's covered here.
- A mild day-of-week signal in forecast residuals (weekends running ~0.2–0.3°C cooler) was observed but not confirmed — sample size in the test window is too small to distinguish from noise.
- Evaluation used a single train/test split; rolling-origin backtesting (refitting across multiple split points) would give a more robust estimate of model performance than one fixed 60-day holdout.

## Author

Joseph — [add contact/LinkedIn if you want it here]

# Time Series Forecasting Portfolio

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/license-MIT-2ea44f)](LICENSE)

一个可复现的小时级城市需求预测作品集项目。它把基础 Demo 提升为一条完整的时间序列建模链路：季节性基线、可解释模型、树模型、扩展窗口回测、残差分位数预测区间和诊断看板。

This repository is a compact, reproducible forecasting study built on synthetic hourly demand data. It emphasizes honest chronological validation, baseline comparison, uncertainty communication, and inspectable residual diagnostics rather than production claims.

## Why This Project

- **Decision-oriented evaluation:** every candidate is evaluated on the same expanding-window backtests.
- **Baseline discipline:** `seasonal_naive` predicts the value from 24 hours earlier, making model lift measurable.
- **Leakage-aware forecasting:** recursive multi-step prediction only feeds model-generated demand back into future lags.
- **Uncertainty made visible:** prediction bands use empirical residual quantiles from historical backtests.
- **Reproducible by default:** deterministic data generation, pinned dependency ranges, CSV artifacts, and pytest coverage.

## Architecture

```mermaid
flowchart LR
    A[generate_data.py\nsynthetic hourly demand] --> B[data/demand.csv]
    B --> C[Feature engineering\ncalendar + lags]
    C --> D[Candidate models\nSeasonal Naive | Linear | Random Forest]
    D --> E[Expanding-window backtest]
    E --> F[Leaderboard + residuals]
    F --> G[Intervals + diagnostics]
    G --> H[Streamlit dashboard]
```

## Feature Matrix

| Capability | Implementation | Output |
|---|---|---|
| Seasonal baseline | 24-hour lag (`seasonal_naive`) | Transparent reference forecast |
| Candidate comparison | Same chronological windows for all models | `data/leaderboard.csv` |
| Forecasting | Recursive multi-step horizon | Future demand predictions |
| Uncertainty | Empirical residual quantiles | `lower <= prediction <= upper` |
| Diagnostics | Bias, sample standard deviation, quantiles, lag-1 autocorrelation | Dashboard and Python API |
| Evaluation | MAE, RMSE, MAPE, SMAPE | Window-level and mean scores |

## Real Run Results

The following values were produced locally from the deterministic dataset with a 60-day initial window, 7-day test window, and 30-day step:

| Model | MAE | RMSE | MAPE | SMAPE | Windows |
|---|---:|---:|---:|---:|---:|
| seasonal_naive | 12.148 | 15.111 | 11.037% | 11.742% | 4 |
| random_forest | 12.659 | 15.310 | 11.313% | 12.238% | 4 |
| linear | 13.513 | 17.573 | 11.836% | 12.800% | 4 |

These scores are a reproducibility check on synthetic data, not evidence of real-world performance.

## Quick Start

```bash
python -m pip install -r requirements.txt
python generate_data.py
python train.py
python -m pytest -q
streamlit run app.py
```

Open the local Streamlit URL. Use the sidebar to select a model, horizon, training and backtest windows, and interval coverage. The dashboard shows the leaderboard, observed/forecast timeline with an uncertainty band, forecast CSV, residual diagnostics, and rolling error details.

## Repository Guide

- `generate_data.py`: deterministic synthetic data generator.
- `train.py`: feature engineering, models, recursive forecasts, backtesting, comparison, intervals, and diagnostics.
- `app.py`: Streamlit analysis workspace.
- `tests/test_core.py`: reproducibility, leakage, model comparison, interval ordering, diagnostics, and metric tests.
- `data/`: generated input and evaluation artifacts.

## Evaluation Notes

Each rolling window trains only on observations before the test window. Forecasts for later steps use earlier predictions for `lag_1`, `lag_24`, and the rolling mean. The demo holds weather features at the latest observed values when creating future rows; a real deployment would require weather forecasts and stronger feature governance.

Residual intervals are empirical and marginal: they summarize historical backtest errors and do not model changing variance, hierarchical uncertainty, or probabilistic dependence across the horizon.

## Scope And Limitations

This is a portfolio and teaching project, not a production forecasting service. It uses synthetic data and does not include holidays, promotions, exogenous forecast uncertainty, data-quality monitoring, model registry, scheduled retraining, authentication, or an API. Those are deliberate next steps for a real system.

## Resume Wording

**Built a reproducible hourly demand forecasting workflow in Python with seasonal-naive benchmarking, expanding-window backtesting, recursive multi-step inference, residual-quantile prediction intervals, and a Streamlit diagnostics dashboard; added pytest coverage for leakage prevention and uncertainty ordering.**

## License

MIT. See `LICENSE` if present in the surrounding project distribution.

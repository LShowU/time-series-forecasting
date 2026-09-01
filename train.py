"""Train and chronologically evaluate demand forecasters without future leakage."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from generate_data import save_data

FEATURES = ["temperature", "precipitation", "is_weekend", "hour", "day_of_week", "day_of_year", "lag_1", "lag_24", "rolling_24"]
MODEL_NAMES = ("seasonal_naive", "linear", "random_forest")


class SeasonalNaive:
    """A transparent daily seasonal baseline: forecast each hour from t-24."""

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "SeasonalNaive":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["lag_24"].to_numpy(dtype=float)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    required = {"timestamp", "demand"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").set_index("timestamp")
    data["hour"] = data.index.hour
    data["day_of_week"] = data.index.dayofweek
    data["day_of_year"] = data.index.dayofyear
    data["lag_1"] = data["demand"].shift(1)
    data["lag_24"] = data["demand"].shift(24)
    data["rolling_24"] = data["demand"].shift(1).rolling(24).mean()
    return data.dropna().reset_index()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denominator = np.maximum(np.abs(y_true), 1e-9)
    smape_denominator = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-9)
    return {"MAE": float(mean_absolute_error(y_true, y_pred)), "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))), "MAPE": float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100), "SMAPE": float(np.mean(2 * np.abs(y_true - y_pred) / smape_denominator) * 100)}


def make_model(model_name: str):
    if model_name == "seasonal_naive":
        return SeasonalNaive()
    if model_name == "linear":
        return LinearRegression()
    if model_name == "random_forest":
        return RandomForestRegressor(n_estimators=40, random_state=42, n_jobs=-1, min_samples_leaf=2)
    raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}")


def fit_model(history: pd.DataFrame, model_name: str = "linear"):
    prepared = make_features(history)
    if prepared.empty:
        raise ValueError("At least 25 historical rows are required")
    model = make_model(model_name)
    model.fit(prepared[FEATURES], prepared["demand"])
    return model


def forecast(model, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """Recursively forecast future rows; only predicted demand feeds future lags."""
    if future.empty:
        return future.assign(prediction=pd.Series(dtype=float))
    history = history.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    future = future.copy().sort_values("timestamp")
    demand_history = list(history.sort_values("timestamp")["demand"].astype(float))
    if len(demand_history) < 24:
        raise ValueError("At least 24 historical rows are required for forecasting")
    predictions = []
    for _, row in future.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        values = {"temperature": row.get("temperature", 0), "precipitation": row.get("precipitation", 0), "is_weekend": row.get("is_weekend", int(timestamp.dayofweek >= 5)), "hour": timestamp.hour, "day_of_week": timestamp.dayofweek, "day_of_year": timestamp.dayofyear, "lag_1": demand_history[-1], "lag_24": demand_history[-24], "rolling_24": float(np.mean(demand_history[-24:]))}
        prediction = float(model.predict(pd.DataFrame([values])[FEATURES])[0])
        predictions.append(prediction)
        demand_history.append(prediction)
    future["prediction"] = predictions
    return future


def backtest_predictions(data: pd.DataFrame, model_name: str = "linear", initial_train: int = 24 * 30, test_size: int = 24 * 7, step: int = 24 * 7) -> pd.DataFrame:
    """Return every out-of-sample prediction from expanding-window backtesting."""
    if model_name not in MODEL_NAMES or initial_train < 25 or test_size < 1 or step < 1:
        raise ValueError("Invalid model or rolling evaluation parameters")
    data = data.copy().sort_values("timestamp").reset_index(drop=True)
    if initial_train + test_size > len(data):
        raise ValueError("Not enough rows for the requested rolling evaluation")
    records = []
    for window, start in enumerate(range(initial_train, len(data) - test_size + 1, step)):
        train, test = data.iloc[:start], data.iloc[start:start + test_size]
        predicted = forecast(fit_model(train, model_name), train, test)
        for timestamp, actual, prediction in zip(test["timestamp"], test["demand"], predicted["prediction"]):
            records.append({"window": window, "timestamp": timestamp, "actual": float(actual), "prediction": float(prediction), "residual": float(actual - prediction), "model": model_name})
    return pd.DataFrame(records)


def rolling_evaluate(data: pd.DataFrame, model_name: str = "linear", initial_train: int = 24 * 30, test_size: int = 24 * 7, step: int = 24 * 7) -> pd.DataFrame:
    predictions = backtest_predictions(data, model_name, initial_train, test_size, step)
    ordered = data.copy().sort_values("timestamp").reset_index(drop=True)
    rows = []
    for window, group in predictions.groupby("window", sort=True):
        score = metrics(group["actual"], group["prediction"])
        train_end = pd.to_datetime(ordered["timestamp"].iloc[initial_train + int(window) * step - 1])
        score.update(model=model_name, window=int(window), train_end=train_end, test_start=group["timestamp"].iloc[0], test_end=group["timestamp"].iloc[-1])
        rows.append(score)
    return pd.DataFrame(rows)


def compare_models(data: pd.DataFrame, model_names: tuple[str, ...] = MODEL_NAMES, initial_train: int = 24 * 30, test_size: int = 24 * 7, step: int = 24 * 7) -> pd.DataFrame:
    """Evaluate all candidate models and rank them by mean RMSE then MAE."""
    rows = []
    for name in model_names:
        scores = rolling_evaluate(data, name, initial_train, test_size, step)
        average = scores[["MAE", "RMSE", "MAPE", "SMAPE"]].mean().to_dict()
        rows.append({"model": name, **{key: float(value) for key, value in average.items()}, "windows": len(scores)})
    return pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)


def residual_diagnostics(residuals: np.ndarray | pd.Series) -> dict[str, float]:
    """Summarize calibration and serial dependence of backtest residuals."""
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("At least one finite residual is required")
    lag1 = float(np.corrcoef(values[:-1], values[1:])[0, 1]) if len(values) > 2 and np.std(values[:-1]) > 0 and np.std(values[1:]) > 0 else 0.0
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"bias": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "q05": float(quantiles[0]), "q25": float(quantiles[1]), "q50": float(quantiles[2]), "q75": float(quantiles[3]), "q95": float(quantiles[4]), "autocorr_lag1": lag1}


def prediction_intervals(predictions: np.ndarray | pd.Series, residuals: np.ndarray | pd.Series, coverage: float = 0.9) -> pd.DataFrame:
    """Create residual-quantile prediction intervals with guaranteed ordering."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1")
    pred = np.asarray(predictions, dtype=float)
    residual = np.asarray(residuals, dtype=float)
    residual = residual[np.isfinite(residual)]
    if not len(residual):
        raise ValueError("At least one finite residual is required")
    tail = (1 - coverage) / 2
    lower = pred + np.quantile(residual, tail)
    upper = pred + np.quantile(residual, 1 - tail)
    lower, upper = np.minimum(lower, pred), np.maximum(upper, pred)
    return pd.DataFrame({"prediction": pred, "lower": lower, "upper": upper})


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline synthetic demand forecasting model comparison")
    parser.add_argument("--data", default="data/demand.csv")
    parser.add_argument("--model", choices=list(MODEL_NAMES), default=None, help="Evaluate only this model; default evaluates all candidates")
    args = parser.parse_args()
    path = Path(args.data)
    if not path.exists():
        save_data(path)
    leaderboard = compare_models(pd.read_csv(path), (args.model,) if args.model else MODEL_NAMES)
    print(leaderboard[["model", "MAE", "RMSE", "MAPE", "SMAPE"]].round(3).to_string(index=False))
    leaderboard.to_csv(path.parent / "leaderboard.csv", index=False)


if __name__ == "__main__":
    main()

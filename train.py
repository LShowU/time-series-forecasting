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
MODEL_NAMES = ("linear", "random_forest")


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
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100),
        "SMAPE": float(np.mean(2 * np.abs(y_true - y_pred) / smape_denominator) * 100),
    }


def make_model(model_name: str):
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
    predictions = []
    for _, row in future.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        values = {
            "temperature": row.get("temperature", 0), "precipitation": row.get("precipitation", 0),
            "is_weekend": row.get("is_weekend", int(timestamp.dayofweek >= 5)),
            "hour": timestamp.hour, "day_of_week": timestamp.dayofweek, "day_of_year": timestamp.dayofyear,
            "lag_1": demand_history[-1], "lag_24": demand_history[-24],
            "rolling_24": float(np.mean(demand_history[-24:])),
        }
        prediction = float(model.predict(pd.DataFrame([values])[FEATURES])[0])
        predictions.append(prediction)
        demand_history.append(prediction)
    future["prediction"] = predictions
    return future


def rolling_evaluate(data: pd.DataFrame, model_name: str = "linear", initial_train: int = 24 * 30, test_size: int = 24 * 7, step: int = 24 * 7) -> pd.DataFrame:
    if model_name not in MODEL_NAMES or initial_train < 25 or test_size < 1 or step < 1:
        raise ValueError("Invalid model or rolling evaluation parameters")
    data = data.copy().sort_values("timestamp").reset_index(drop=True)
    if initial_train + test_size > len(data):
        raise ValueError("Not enough rows for the requested rolling evaluation")
    records = []
    for start in range(initial_train, len(data) - test_size + 1, step):
        train, test = data.iloc[:start], data.iloc[start:start + test_size]
        model = fit_model(train, model_name)
        predicted = forecast(model, train, test)
        score = metrics(test["demand"].to_numpy(), predicted["prediction"].to_numpy())
        score.update(model=model_name, train_end=train["timestamp"].iloc[-1], test_start=test["timestamp"].iloc[0], test_end=test["timestamp"].iloc[-1])
        records.append(score)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline synthetic demand forecasting baseline")
    parser.add_argument("--data", default="data/demand.csv")
    parser.add_argument("--model", choices=list(MODEL_NAMES), default="linear")
    args = parser.parse_args()
    path = Path(args.data)
    if not path.exists():
        save_data(path)
    scores = rolling_evaluate(pd.read_csv(path), args.model)
    print(scores[["MAE", "RMSE", "MAPE", "SMAPE"]].mean().round(3).to_string())
    scores.to_csv(path.parent / "rolling_metrics.csv", index=False)


if __name__ == "__main__":
    main()

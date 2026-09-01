"""Generate deterministic synthetic hourly urban demand data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_demand_data(n_days: int = 180, seed: int = 42) -> pd.DataFrame:
    """Return hourly synthetic demand for an independent demonstration project."""
    if n_days < 14:
        raise ValueError("n_days must be at least 14")
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_days * 24, freq="h")
    hour = timestamps.hour.to_numpy()
    day_index = np.arange(len(timestamps)) // 24
    dow = timestamps.dayofweek.to_numpy()
    is_weekend = (dow >= 5).astype(int)
    seasonal = 12 * np.sin(2 * np.pi * day_index / 30) + 7 * np.cos(2 * np.pi * day_index / 180)
    temperature = 15 + 10 * np.sin(2 * np.pi * (day_index - 30) / 365) + rng.normal(0, 1.5, len(timestamps))
    precipitation = np.clip(rng.gamma(1.2, 0.8, len(timestamps)) - 0.7, 0, None)
    commute = 30 * np.exp(-((hour - 8) / 2.2) ** 2) + 34 * np.exp(-((hour - 17) / 2.8) ** 2)
    daytime = 18 * np.exp(-((hour - 13) / 5) ** 2)
    weekend_effect = is_weekend * 10
    weather_effect = -7 * precipitation - 0.12 * (temperature - 20) ** 2
    trend = day_index * 0.025
    noise = rng.normal(0, 5, len(timestamps))
    demand = np.maximum(8, 85 + commute + daytime + seasonal + weekend_effect + weather_effect + trend + noise)
    return pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temperature.round(2),
        "precipitation": precipitation.round(2),
        "is_weekend": is_weekend,
        "demand": demand.round(2),
    })


def save_data(path: str | Path = "data/demand.csv", n_days: int = 180, seed: int = 42) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_demand_data(n_days=n_days, seed=seed).to_csv(output, index=False)
    return output


if __name__ == "__main__":
    saved = save_data()
    print(f"Generated {saved}")

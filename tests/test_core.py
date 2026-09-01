import numpy as np
import pandas as pd
import pytest

from generate_data import generate_demand_data
from train import FEATURES, fit_model, forecast, make_features, metrics, rolling_evaluate


def test_generation_is_reproducible_and_hourly():
    first = generate_demand_data(30, 7)
    second = generate_demand_data(30, 7)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 30 * 24
    assert first.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert (first.demand > 0).all()


def test_features_use_past_only_and_have_expected_columns():
    original = generate_demand_data(20, 3)
    frame = make_features(original)
    assert set(FEATURES).issubset(frame.columns)
    assert np.isclose(frame.loc[0, "lag_1"], original.loc[23, "demand"])
    assert len(frame) == len(original) - 24


def test_recursive_forecast_has_requested_horizon_and_ignores_future_demand():
    data = generate_demand_data(40, 5)
    history, future = data.iloc[:-24], data.iloc[-24:].copy()
    model = fit_model(history, "linear")
    first = forecast(model, history, future)
    altered = future.copy()
    altered["demand"] = altered["demand"] + 10000
    second = forecast(model, history, altered)
    assert len(first) == 24
    assert first.timestamp.is_monotonic_increasing
    np.testing.assert_allclose(first.prediction, second.prediction)


def test_rolling_evaluation_returns_finite_metrics():
    scores = rolling_evaluate(generate_demand_data(60, 9), initial_train=24 * 14, test_size=24 * 3, step=24 * 3)
    assert len(scores) > 1
    assert scores[["MAE", "RMSE", "MAPE", "SMAPE"]].apply(np.isfinite).all().all()
    assert (scores[["MAE", "RMSE", "MAPE", "SMAPE"]] >= 0).all().all()


def test_metrics_known_values_and_zero_targets():
    result = metrics(np.array([1.0, 2.0]), np.array([2.0, 2.0]))
    assert result["MAE"] == 0.5
    assert result["RMSE"] == np.sqrt(0.5)
    assert result["MAPE"] == 50.0
    zero_result = metrics(np.array([0.0, 2.0]), np.array([0.0, 1.0]))
    assert np.isfinite(list(zero_result.values())).all()


def test_invalid_model_and_short_history_fail_loudly():
    data = generate_demand_data(30, 1)
    with pytest.raises(ValueError, match="Unknown model"):
        fit_model(data, "arima")
    with pytest.raises(ValueError, match="25 historical"):
        fit_model(data.iloc[:20], "linear")
    with pytest.raises(ValueError, match="Invalid model"):
        rolling_evaluate(data, "arima")

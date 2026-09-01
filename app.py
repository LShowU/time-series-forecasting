"""Streamlit dashboard for the synthetic demand forecasting demo."""
from pathlib import Path

import pandas as pd

from generate_data import save_data
from train import fit_model, forecast, rolling_evaluate

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "demand.csv"


def main() -> None:
    try:
        import plotly.express as px
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Install requirements.txt, then run: streamlit run app.py") from exc

    st.set_page_config(page_title="Demand Forecast", page_icon="DF", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    :root { --ink:#17212b; --muted:#667784; --line:#dce6e8; --teal:#198f88; --teal-soft:#e8f5f3; --indigo:#4655a8; --surface:#ffffff; }
    .stApp { background:#f6f9f9; color:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    .block-container { max-width:1440px; padding:2.2rem 3rem 3rem; }
    [data-testid="stSidebar"] { background:#eef5f4; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label { color:var(--ink); }
    .hero { display:flex; justify-content:space-between; align-items:flex-start; gap:24px; padding:4px 0 22px; border-bottom:1px solid var(--line); margin-bottom:22px; }
    .eyebrow { color:var(--teal); font-size:.74rem; font-weight:750; letter-spacing:.08em; text-transform:uppercase; margin-bottom:8px; }
    .hero h1 { color:var(--ink); font-size:2.05rem; line-height:1.15; margin:0 0 8px; letter-spacing:0; }
    .hero p { color:var(--muted); margin:0; font-size:.98rem; }
    .status { display:inline-flex; align-items:center; gap:8px; color:#176c67; background:var(--teal-soft); border:1px solid #c4e5e1; border-radius:999px; padding:8px 13px; font-size:.82rem; font-weight:650; white-space:nowrap; }
    .status-dot { width:8px; height:8px; background:var(--teal); border-radius:50%; }
    .section-title { color:var(--ink); font-weight:720; font-size:1.05rem; margin:1.3rem 0 .65rem; }
    .section-note { color:var(--muted); font-size:.83rem; margin:-.35rem 0 .8rem; }
    [data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); border-left:4px solid var(--teal); border-radius:8px; padding:15px 17px; box-shadow:0 2px 8px rgba(23,33,43,.035); }
    [data-testid="stMetricLabel"] { color:var(--muted); font-size:.8rem; }
    [data-testid="stMetricValue"] { color:var(--ink); font-size:1.55rem; }
    .meta-strip { color:var(--muted); background:#fff; border:1px solid var(--line); border-radius:8px; padding:11px 15px; margin-bottom:16px; font-size:.86rem; }
    .empty { color:var(--muted); background:#fff; border:1px dashed #b8cbcd; border-radius:8px; padding:28px; text-align:center; }
    .stButton button, .stDownloadButton button { border-radius:6px; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="hero">
      <div><div class="eyebrow">Forecast intelligence workspace</div><h1>Demand Forecast</h1><p>Chronological backtesting with recursive multi-step forecasts</p></div>
      <div class="status"><span class="status-dot"></span>Offline · reproducible</div>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def load_data(path: str) -> pd.DataFrame:
        if not Path(path).exists():
            save_data(path)
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        if not {"timestamp", "demand"}.issubset(frame.columns):
            raise ValueError("CSV must contain timestamp and demand columns")
        return frame.sort_values("timestamp").reset_index(drop=True)

    @st.cache_data
    def evaluate(frame: pd.DataFrame, model: str, train_days: int, test_days: int, step_days: int) -> pd.DataFrame:
        return rolling_evaluate(frame, model, 24 * train_days, 24 * test_days, 24 * step_days)

    data = load_data(str(DATA_PATH))
    with st.sidebar:
        st.header("Forecast controls")
        model_name = st.selectbox("Model", ["linear", "random_forest"])
        horizon = st.slider("Forecast horizon (hours)", 24, 24 * 14, 24 * 7, 24)
        train_days = st.slider("Initial train window (days)", 14, 120, 30)
        test_days = st.slider("Backtest window (days)", 1, 14, 7)
        step_days = st.slider("Backtest step (days)", 1, 14, 7)

    st.markdown(f'<div class="meta-strip"><b>Dataset</b> &nbsp; {len(data):,} hourly records &nbsp; · &nbsp; {data["timestamp"].min():%Y-%m-%d} to {data["timestamp"].max():%Y-%m-%d} &nbsp; · &nbsp; <b>Model</b> {model_name.replace("_", " ").title()} &nbsp; · &nbsp; <b>Horizon</b> {horizon} hours</div>', unsafe_allow_html=True)
    try:
        scores = evaluate(data, model_name, train_days, test_days, step_days)
        model = fit_model(data, model_name)
        last_time = data["timestamp"].max()
        future_times = pd.date_range(last_time + pd.Timedelta(hours=1), periods=horizon, freq="h")
        latest = data.iloc[-1]
        future = pd.DataFrame({"timestamp": future_times, "temperature": latest.get("temperature", 0), "precipitation": latest.get("precipitation", 0)})
        future["is_weekend"] = future.timestamp.dt.dayofweek.ge(5).astype(int)
        prediction = forecast(model, data, future)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.markdown('<div class="section-title">Model performance</div><div class="section-note">Mean error across chronological rolling windows</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, label, fmt in zip(cols, ["MAE", "RMSE", "MAPE", "SMAPE"], ["{:.2f}", "{:.2f}", "{:.2f}%", "{:.2f}%"]):
        col.metric(label, fmt.format(average[label]))

    history = data.tail(min(len(data), 24 * 30)).rename(columns={"demand": "value"})[["timestamp", "value"]]
    future_plot = prediction.rename(columns={"prediction": "value"})[["timestamp", "value"]]
    chart_data = pd.concat([history.assign(series="Observed"), future_plot.assign(series="Forecast")])
    st.markdown('<div class="section-title">Demand timeline</div><div class="section-note">Observed demand and recursive forecast for the selected horizon</div>', unsafe_allow_html=True)
    if chart_data.empty:
        st.markdown('<div class="empty">No timeline data is available for this configuration.</div>', unsafe_allow_html=True)
    else:
        fig = px.line(chart_data, x="timestamp", y="value", color="series", title=None, color_discrete_map={"Observed":"#198f88", "Forecast":"#4655a8"})
        fig.update_layout(height=390, margin=dict(l=10,r=10,t=20,b=10), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", legend_title_text="", hovermode="x unified")
        fig.update_xaxes(showgrid=False, title=None)
        fig.update_yaxes(gridcolor="#edf2f2", title="Demand")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown('<div class="section-title">Forecast values</div>', unsafe_allow_html=True)
        if prediction.empty:
            st.markdown('<div class="empty">No forecast rows yet. Increase the horizon or check the data file.</div>', unsafe_allow_html=True)
        else:
            st.dataframe(prediction[["timestamp", "prediction"]], use_container_width=True, hide_index=True, height=300)
            st.download_button("Download forecast CSV", prediction.to_csv(index=False), "forecast.csv", "text/csv")
    with right:
        st.markdown('<div class="section-title">Run configuration</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="meta-strip"><b>Training window</b><br>{train_days} days<br><br><b>Backtest window</b><br>{test_days} days<br><br><b>Step</b><br>{step_days} days</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Backtest diagnostics</div><div class="section-note">Window-level error helps reveal model stability over time</div>', unsafe_allow_html=True)
    if scores.empty:
        st.markdown('<div class="empty">No backtest windows are available. Reduce the training or test window.</div>', unsafe_allow_html=True)
    else:
        diag_left, diag_right = st.columns([1.15, .85], gap="large")
        metric_long = scores.reset_index(names="window").melt(id_vars="window", value_vars=["MAE", "RMSE", "MAPE", "SMAPE"], var_name="metric", value_name="value")
        with diag_left:
            fig = px.line(metric_long, x="window", y="value", color="metric", markers=True, title=None, color_discrete_sequence=["#198f88", "#4655a8", "#d97745", "#6d7c8c"])
            fig.update_layout(height=330, margin=dict(l=10,r=10,t=20,b=10), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", legend_title_text="")
            fig.update_xaxes(showgrid=False, title="Rolling window")
            fig.update_yaxes(gridcolor="#edf2f2", title="Error")
            st.plotly_chart(fig, use_container_width=True)
        with diag_right:
            st.dataframe(scores, use_container_width=True, hide_index=True, height=280)
            st.download_button("Download backtest CSV", scores.to_csv(index=False), "rolling_metrics.csv", "text/csv")


if __name__ == "__main__":
    main()

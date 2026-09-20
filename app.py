"""
Weather Rainfall Prediction Dashboard  (Experiment 8)
Run:  streamlit run app.py
"""
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Rainfall Prediction Dashboard", page_icon="🌧️", layout="wide")

MODEL_PATH = "best_weather_rainfall_model.pkl"


# --------------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:  # usually a scikit-learn version mismatch
    import sklearn
    st.error(
        f"Could not load the model ({type(e).__name__}: {e}).\n\n"
        f"Installed scikit-learn = {sklearn.__version__}. The model was saved with a different version - "
        "pin the same version in requirements.txt (e.g. scikit-learn==1.8.0) and reboot the app."
    )
    st.stop()
FEATURES = list(model.feature_names_in_)


# --------------------------------------------------------------------------
# Feature engineering  (!! must match what you did in your training notebook !!)
# --------------------------------------------------------------------------
def build_features(temp, hum, wind, wind_dir_deg, pressure, cloud, dt,
                   hum_l3, hum_l6, pres_l3, pres_l6, cloud_l3, cloud_l6):
    doy = dt.timetuple().tm_yday
    row = {
        "Temperature": temp,
        "Humidity": hum,
        "Wind_Speed": wind,
        "Wind_Direction_sin": np.sin(np.deg2rad(wind_dir_deg)),
        "Wind_Direction_cos": np.cos(np.deg2rad(wind_dir_deg)),
        "Atmospheric_Pressure": pressure,
        "Cloud_Cover": cloud,
        "Month": dt.month,
        "Day": dt.day,
        "DayOfWeek": dt.weekday(),
        "Hour_sin": np.sin(2 * np.pi * dt.hour / 24),
        "Hour_cos": np.cos(2 * np.pi * dt.hour / 24),
        "DayOfYear_sin": np.sin(2 * np.pi * doy / 365),
        "DayOfYear_cos": np.cos(2 * np.pi * doy / 365),
        "Humidity_Lag_3": hum_l3,
        "Humidity_Lag_6": hum_l6,
        "Pressure_Lag_3": pres_l3,
        "Pressure_Lag_6": pres_l6,
        "Cloud_Lag_3": cloud_l3,
        "Cloud_Lag_6": cloud_l6,
        "Humidity_Change_3": hum - hum_l3,
        "Pressure_Change_3": pressure - pres_l3,
        "Cloud_Change_3": cloud - cloud_l3,
    }
    return pd.DataFrame([row])[FEATURES]


@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(model)


# --------------------------------------------------------------------------
# Drift helpers
# --------------------------------------------------------------------------
def psi(reference, current, bins=10):
    """Population Stability Index. <0.1 stable, 0.1-0.25 moderate, >0.25 major drift."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct, cur_pct = np.clip(ref_pct, 1e-6, None), np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(ref_df, cur_df):
    rows = []
    for col in FEATURES:
        if col in ref_df and col in cur_df:
            p = psi(ref_df[col].dropna(), cur_df[col].dropna())
            ks = ks_2samp(ref_df[col].dropna(), cur_df[col].dropna())
            status = "🟢 Stable" if p < 0.1 else ("🟡 Moderate" if p < 0.25 else "🔴 Drift")
            rows.append({"Feature": col, "PSI": round(p, 4),
                         "KS statistic": round(ks.statistic, 4),
                         "KS p-value": round(ks.pvalue, 4), "Status": status})
    return pd.DataFrame(rows).sort_values("PSI", ascending=False)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🌧️ Rainfall Prediction Dashboard")
st.caption("Gradient Boosting Regressor · predictions, SHAP explanations, metrics, drift checks and Responsible AI notes")

tab_pred, tab_shap, tab_metrics, tab_drift, tab_rai = st.tabs(
    ["🔮 Predict", "🧠 SHAP Explainability", "📊 Model Metrics", "🌊 Drift Check", "⚖️ Responsible AI"]
)

# --------------------------------------------------------------------------
# Tab 1: Predict
# --------------------------------------------------------------------------
with tab_pred:
    st.subheader("Enter current weather conditions")
    c1, c2, c3 = st.columns(3)
    with c1:
        temp = st.number_input("Temperature (°C)", -10.0, 55.0, 28.0, 0.5)
        hum = st.slider("Humidity (%)", 0.0, 100.0, 75.0)
        wind = st.number_input("Wind speed", 0.0, 150.0, 12.0, 0.5)
        wind_dir = st.slider("Wind direction (°)", 0, 360, 180)
    with c2:
        pressure = st.number_input("Atmospheric pressure (hPa)", 900.0, 1100.0, 1005.0, 0.5)
        cloud = st.slider("Cloud cover (%)", 0.0, 100.0, 60.0)
        date = st.date_input("Date", datetime.now().date())
        hour = st.slider("Hour of day", 0, 23, datetime.now().hour)
    with c3:
        st.markdown("**Readings from earlier (lag values)**")
        hum_l3 = st.number_input("Humidity 3 h ago", 0.0, 100.0, 70.0)
        hum_l6 = st.number_input("Humidity 6 h ago", 0.0, 100.0, 65.0)
        pres_l3 = st.number_input("Pressure 3 h ago", 900.0, 1100.0, 1006.0)
        pres_l6 = st.number_input("Pressure 6 h ago", 900.0, 1100.0, 1007.0)
        cloud_l3 = st.number_input("Cloud cover 3 h ago", 0.0, 100.0, 50.0)
        cloud_l6 = st.number_input("Cloud cover 6 h ago", 0.0, 100.0, 40.0)

    dt = datetime.combine(date, datetime.min.time()).replace(hour=hour)
    X_one = build_features(temp, hum, wind, wind_dir, pressure, cloud, dt,
                           hum_l3, hum_l6, pres_l3, pres_l6, cloud_l3, cloud_l6)
    pred = float(model.predict(X_one)[0])
    pred = max(pred, 0.0)  # rainfall cannot be negative

    m1, m2 = st.columns(2)
    m1.metric("Predicted rainfall", f"{pred:.2f}")
    m2.metric("Category", "🌂 Rain likely" if pred >= 1 else "☀️ Little/no rain")
    with st.expander("Model input row (after feature engineering)"):
        st.dataframe(X_one.T.rename(columns={0: "value"}))

    st.session_state["X_one"] = X_one

# --------------------------------------------------------------------------
# Tab 2: SHAP
# --------------------------------------------------------------------------
with tab_shap:
    st.subheader("Why did the model predict this?")
    explainer = get_explainer()

    st.markdown("#### Local explanation (the prediction on the *Predict* tab)")
    sv_one = explainer(st.session_state["X_one"])
    fig = plt.figure()
    shap.plots.waterfall(sv_one[0], max_display=12, show=False)
    st.pyplot(fig, clear_figure=True)

    st.markdown("#### Global explanation")
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.plot.barh(ax=ax)
    ax.set_title("Built-in feature importance")
    st.pyplot(fig, clear_figure=True)

    st.markdown("**SHAP summary plot** — upload your processed feature data "
                "(e.g. `X_test.to_csv('reference_data.csv', index=False)`)")
    up_shap = st.file_uploader("Feature CSV for SHAP", type="csv", key="shap_csv")
    if up_shap:
        Xs = pd.read_csv(up_shap)
        missing = [c for c in FEATURES if c not in Xs.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            Xs = Xs[FEATURES].sample(min(500, len(Xs)), random_state=42)
            sv = explainer(Xs)
            fig = plt.figure()
            shap.summary_plot(sv.values, Xs, show=False)
            st.pyplot(fig, clear_figure=True)

# --------------------------------------------------------------------------
# Tab 3: Metrics
# --------------------------------------------------------------------------
with tab_metrics:
    st.subheader("Evaluate the model on a labelled test set")
    st.write("Upload a CSV containing the 23 feature columns **plus** the true rainfall column.")
    up_test = st.file_uploader("Test CSV", type="csv", key="test_csv")
    if up_test:
        df = pd.read_csv(up_test)
        target = st.selectbox("Target (actual rainfall) column",
                              [c for c in df.columns if c not in FEATURES])
        if all(c in df.columns for c in FEATURES):
            y_true, y_pred = df[target], np.clip(model.predict(df[FEATURES]), 0, None)
            a, b, c = st.columns(3)
            a.metric("MAE", f"{mean_absolute_error(y_true, y_pred):.3f}")
            b.metric("RMSE", f"{np.sqrt(mean_squared_error(y_true, y_pred)):.3f}")
            c.metric("R²", f"{r2_score(y_true, y_pred):.3f}")

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].scatter(y_true, y_pred, alpha=0.4)
            lim = [0, max(y_true.max(), y_pred.max())]
            axes[0].plot(lim, lim, "r--")
            axes[0].set(xlabel="Actual", ylabel="Predicted", title="Actual vs Predicted")
            axes[1].hist(y_true - y_pred, bins=40)
            axes[1].set(title="Residuals (actual − predicted)")
            st.pyplot(fig, clear_figure=True)
        else:
            st.error("The CSV is missing some of the required feature columns.")
    else:
        st.info("Upload a test CSV to see MAE, RMSE, R² and residual plots.")

# --------------------------------------------------------------------------
# Tab 4: Drift
# --------------------------------------------------------------------------
with tab_drift:
    st.subheader("Has the incoming data changed since training?")
    col_a, col_b = st.columns(2)
    ref_file = col_a.file_uploader("Reference data (training features CSV)", type="csv", key="ref")
    cur_file = col_b.file_uploader("New / recent data (features CSV)", type="csv", key="cur")
    if ref_file and cur_file:
        ref_df, cur_df = pd.read_csv(ref_file), pd.read_csv(cur_file)
        rep = drift_report(ref_df, cur_df)
        n_drift = int((rep["PSI"] >= 0.25).sum())
        st.metric("Features with major drift (PSI ≥ 0.25)", f"{n_drift} / {len(rep)}")
        st.dataframe(rep, use_container_width=True, hide_index=True)
        feat = st.selectbox("Compare distribution of", rep["Feature"])
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(ref_df[feat].dropna(), bins=30, alpha=0.6, density=True, label="Reference")
        ax.hist(cur_df[feat].dropna(), bins=30, alpha=0.6, density=True, label="New")
        ax.legend(); ax.set_title(feat)
        st.pyplot(fig, clear_figure=True)
        st.caption("PSI < 0.1 stable · 0.1–0.25 moderate shift · > 0.25 significant drift → consider retraining.")
    else:
        st.info("Upload both files to run PSI and Kolmogorov–Smirnov drift tests.")

# --------------------------------------------------------------------------
# Tab 5: Responsible AI
# --------------------------------------------------------------------------
with tab_rai:
    st.subheader("Responsible AI summary")
    try:
        with open("Responsible_AI.md", encoding="utf-8") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.warning("Responsible_AI.md not found in the app folder.")

"""
==============================================================================
Project 9 - Smart City Traffic Forecasting
Flask Web Application & Prediction Server
==============================================================================
This module serves the interactive frontend dashboard and connects directly
to the trained machine learning model and feature engineering pipeline.

Architecture:
  Frontend (HTML/CSS/JS)
         ↓
  Flask Server (app.py)
         ↓
  Pretrained Model (models/traffic_model.pkl)
         ↓
  Real-Time Prediction & Operational Classification
==============================================================================
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Flask Application
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "traffic_model.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Exactly the 22 feature columns used during model training
FEATURE_COLS = [
    "Junction", "Year", "Month", "Day", "Hour", "DayOfWeek", "DayOfYear", "WeekOfYear",
    "IsWeekend", "Traffic_Period", "Hour_Sin", "Hour_Cos", "DayOfWeek_Sin", "DayOfWeek_Cos",
    "Lag_1", "Lag_2", "Lag_3", "Lag_24", "Lag_48", "Lag_168",
    "Rolling_Mean_24", "Rolling_Mean_168"
]

# Global cache for model and historical series
MODEL = None
JUNCTION_SERIES = {}
BASELINE_HOURLY = {}
JUNCTION_STATS = {}
HOURLY_PROFILE = []
MODEL_COMPARISON = []


def find_data_file(filename):
    """Locate CSV dataset robustly regardless of working directory or double extensions."""
    base_name = os.path.splitext(filename)[0]
    candidate_paths = [
        os.path.join(BASE_DIR, "data", filename),
        os.path.join(BASE_DIR, "data", filename.lower()),
        os.path.join(BASE_DIR, "data", f"{base_name.lower()}.csv.csv"),
        os.path.join(BASE_DIR, "data", f"{filename}.csv"),
        os.path.join("data", filename),
        os.path.join("data", f"{base_name.lower()}.csv.csv"),
    ]
    for path in candidate_paths:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Cannot locate dataset file: {filename}")


def load_application_resources():
    """
    Loads:
    1. Saved production model (models/traffic_model.pkl)
    2. Historical training data & predictions to construct a continuous lookup series per junction
    3. Hourly diurnal patterns and model evaluation records
    """
    global MODEL, JUNCTION_SERIES, BASELINE_HOURLY, JUNCTION_STATS, HOURLY_PROFILE, MODEL_COMPARISON

    # 1. Load Pretrained Model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Pretrained model file not found at: {MODEL_PATH}. "
            "Please run 'python traffic_forecasting.py --no-plots' first to train and export the model."
        )

    print(f">> Loading pretrained model from: {MODEL_PATH} ...")
    with open(MODEL_PATH, "rb") as f:
        MODEL = pickle.load(f)
    print(">> Model loaded successfully!")

    # 2. Build Continuous Junction Time-Series Lookup
    try:
        train_path = find_data_file("Train.csv")
        train_df = pd.read_csv(train_path)
        dt_col = "DateTime" if "DateTime" in train_df.columns else "datetime"
        train_df[dt_col] = pd.to_datetime(train_df[dt_col])
        train_df = train_df.sort_values(by=[dt_col, "Junction"]).reset_index(drop=True)

        # Baseline hourly profile for fallback/generalization
        train_df["Hour"] = train_df[dt_col].dt.hour
        train_df["DayOfWeek"] = train_df[dt_col].dt.dayofweek
        hourly_grp = train_df.groupby("Hour")["Vehicles"].mean()
        HOURLY_PROFILE = [round(float(hourly_grp.get(h, 20.0)), 2) for h in range(24)]

        # Junction statistics
        j_train_stats = train_df.groupby("Junction")["Vehicles"].mean().to_dict()

        # Load predictions if available
        pred_path = os.path.join(BASE_DIR, "traffic_predictions.csv")
        pred_df = None
        j_pred_stats = {}
        if os.path.exists(pred_path):
            pred_df = pd.read_csv(pred_path)
            pred_df[dt_col] = pd.to_datetime(pred_df[dt_col])
            j_pred_stats = pred_df.groupby("Junction")["Predicted_Traffic"].mean().to_dict()

        # Combine training and test predictions into continuous series per junction
        for j in [1, 2, 3, 4]:
            t_subset = train_df[train_df["Junction"] == j][[dt_col, "Vehicles"]].rename(columns={"Vehicles": "Traffic"})
            if pred_df is not None:
                p_subset = pred_df[pred_df["Junction"] == j][[dt_col, "Predicted_Traffic"]].rename(columns={"Predicted_Traffic": "Traffic"})
                full_junc_df = pd.concat([t_subset, p_subset], ignore_index=True)
            else:
                full_junc_df = t_subset

            full_junc_df = full_junc_df.drop_duplicates(subset=[dt_col]).sort_values(by=dt_col)
            series = pd.Series(full_junc_df["Traffic"].values, index=full_junc_df[dt_col])
            JUNCTION_SERIES[j] = series

            # Baseline lookup: (Junction, DayOfWeek, Hour)
            j_train_df = train_df[train_df["Junction"] == j]
            base_map = j_train_df.groupby(["DayOfWeek", "Hour"])["Vehicles"].median().to_dict()
            BASELINE_HOURLY[j] = base_map

            JUNCTION_STATS[j] = {
                "historical_mean": round(float(j_train_stats.get(j, 20.0)), 2),
                "predicted_mean": round(float(j_pred_stats.get(j, j_train_stats.get(j, 20.0))), 2)
            }

        print(">> Historical junction time series initialized.")
    except Exception as e:
        print(f">> Warning during time series loading: {e}")

    # 3. Load Model Comparison Table
    comp_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        MODEL_COMPARISON = comp_df.to_dict(orient="records")
    else:
        MODEL_COMPARISON = [
            {"Model": "Random Forest", "MAE": 2.9065, "RMSE": 4.7729, "R2": 0.9692},
            {"Model": "Linear Regression", "MAE": 3.0821, "RMSE": 4.8003, "R2": 0.9689},
            {"Model": "Decision Tree", "MAE": 4.2498, "RMSE": 6.7260, "R2": 0.9389}
        ]


# Initialize resources on module import
load_application_resources()


def compute_pipeline_features(junction: int, target_dt: pd.Timestamp) -> dict:
    """
    Applies the EXACT SAME preprocessing and feature-engineering logic
    used during model training to construct the 22 required features.
    
    Strictly prevents target leakage by only utilizing historical timestamps (t - 1 to t - 168).
    """
    year = target_dt.year
    month = target_dt.month
    day = target_dt.day
    hour = target_dt.hour
    dayofweek = target_dt.dayofweek
    dayofyear = target_dt.dayofyear
    weekofyear = int(target_dt.isocalendar().week)
    is_weekend = 1 if dayofweek >= 5 else 0

    # Traffic period encoding: Night=0, Morning=1, Afternoon=2, Evening=3
    if 0 <= hour <= 5:
        traffic_period = 0
    elif 6 <= hour <= 11:
        traffic_period = 1
    elif 12 <= hour <= 17:
        traffic_period = 2
    else:
        traffic_period = 3

    # Cyclical trigonometric transformations
    hour_sin = float(np.sin(2.0 * np.pi * hour / 24.0))
    hour_cos = float(np.cos(2.0 * np.pi * hour / 24.0))
    dow_sin = float(np.sin(2.0 * np.pi * dayofweek / 7.0))
    dow_cos = float(np.cos(2.0 * np.pi * dayofweek / 7.0))

    # Retrieve junction series and fallback baseline
    series = JUNCTION_SERIES.get(junction)
    baseline_map = BASELINE_HOURLY.get(junction, {})

    def get_past_value(offset_hours: int) -> float:
        past_dt = target_dt - pd.Timedelta(hours=offset_hours)
        if series is not None and past_dt in series.index:
            return float(series.loc[past_dt])
        # Fallback to seasonal historical profile for that junction
        past_dow = past_dt.dayofweek
        past_hr = past_dt.hour
        return float(baseline_map.get((past_dow, past_hr), 20.0))

    # Calculate exact historical lag features
    lag_1 = get_past_value(1)
    lag_2 = get_past_value(2)
    lag_3 = get_past_value(3)
    lag_24 = get_past_value(24)
    lag_48 = get_past_value(48)
    lag_168 = get_past_value(168)

    # Calculate rolling statistics (strictly looking backward)
    if series is not None:
        start_24 = target_dt - pd.Timedelta(hours=24)
        end_1 = target_dt - pd.Timedelta(hours=1)
        window_24 = series.loc[start_24:end_1]
        rolling_24 = float(window_24.mean()) if len(window_24) > 0 else (lag_1 + lag_24) / 2.0

        start_168 = target_dt - pd.Timedelta(hours=168)
        window_168 = series.loc[start_168:end_1]
        rolling_168 = float(window_168.mean()) if len(window_168) > 0 else (lag_24 + lag_168) / 2.0
    else:
        rolling_24 = (lag_1 + lag_24) / 2.0
        rolling_168 = (lag_24 + lag_168) / 2.0

    return {
        "Junction": int(junction),
        "Year": int(year),
        "Month": int(month),
        "Day": int(day),
        "Hour": int(hour),
        "DayOfWeek": int(dayofweek),
        "DayOfYear": int(dayofyear),
        "WeekOfYear": int(weekofyear),
        "IsWeekend": int(is_weekend),
        "Traffic_Period": int(traffic_period),
        "Hour_Sin": round(hour_sin, 6),
        "Hour_Cos": round(hour_cos, 6),
        "DayOfWeek_Sin": round(dow_sin, 6),
        "DayOfWeek_Cos": round(dow_cos, 6),
        "Lag_1": round(lag_1, 2),
        "Lag_2": round(lag_2, 2),
        "Lag_3": round(lag_3, 2),
        "Lag_24": round(lag_24, 2),
        "Lag_48": round(lag_48, 2),
        "Lag_168": round(lag_168, 2),
        "Rolling_Mean_24": round(rolling_24, 2),
        "Rolling_Mean_168": round(rolling_168, 2),
    }


def classify_traffic_level(predicted_traffic: float) -> str:
    """
    Derived operational interpretation thresholds:
      - Low Traffic:      < 20 vehicles / hour
      - Moderate Traffic: 20 to 50 vehicles / hour
      - Heavy Traffic:    > 50 vehicles / hour
    (Explicit operational categorization; not a classification model prediction)
    """
    if predicted_traffic < 20.0:
        return "Low"
    elif predicted_traffic <= 50.0:
        return "Moderate"
    else:
        return "Heavy"


# ==============================================================================
# ROUTES
# ==============================================================================

@app.route("/")
def index():
    """Renders the main smart-city traffic dashboard."""
    return render_template(
        "index.html",
        selected_model_name="Random Forest",
        best_mae="2.9065",
        best_rmse="4.7729",
        best_r2="0.9692",
        comparison_table=MODEL_COMPARISON
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint:
    Accepts JSON: { "junction": 1, "date": "2017-08-15", "time": "09:00" }
    Validates input, runs identical feature engineering, queries the saved model,
    and returns predicted vehicle counts and derived classification.
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "Request payload must be valid JSON."}), 400

    data = request.get_json()

    # 1. Input Validation
    junction_raw = data.get("junction")
    date_str = data.get("date")
    time_str = data.get("time")

    if not junction_raw or not date_str or not time_str:
        return jsonify({"success": False, "error": "Missing required fields: junction, date, or time."}), 400

    try:
        junction = int(junction_raw)
        if junction not in [1, 2, 3, 4]:
            return jsonify({"success": False, "error": "Junction must be 1, 2, 3, or 4."}), 400
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Invalid junction identifier. Expected integer 1-4."}), 400

    try:
        target_dt = pd.to_datetime(f"{date_str} {time_str}")
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid date or time format: {e}"}), 400

    if MODEL is None:
        return jsonify({"success": False, "error": "ML model is not loaded."}), 500

    try:
        # 2. Extract pipeline features in exact order
        features_dict = compute_pipeline_features(junction, target_dt)
        feature_vector = [features_dict[col] for col in FEATURE_COLS]
        X = pd.DataFrame([feature_vector], columns=FEATURE_COLS)

        # 3. Model Inference
        raw_pred = MODEL.predict(X)[0]
        # Bounded to non-negative traffic volume
        prediction = float(max(0.0, round(raw_pred, 2)))

        # 4. Derived Operational Traffic Level
        traffic_level = classify_traffic_level(prediction)

        return jsonify({
            "success": True,
            "prediction": prediction,
            "level": traffic_level,
            "junction": junction,
            "datetime": target_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "features": features_dict
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Inference execution failed: {str(e)}"}), 500


@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    """Returns actual project statistics and historical curves for Chart.js."""
    return jsonify({
        "success": True,
        "junction_stats": JUNCTION_STATS,
        "hourly_profile": HOURLY_PROFILE,
        "model_comparison": MODEL_COMPARISON
    })


@app.route("/results/figures/<path:filename>")
def serve_figure(filename):
    """Serves high-resolution diagnostic figures from results/figures/."""
    return send_from_directory(FIGURES_DIR, filename)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚦 Smart City Traffic Forecasting Web Application")
    print("=" * 60)
    print(f"Loaded Model : {MODEL_PATH}")
    print("Server URL   : http://127.0.0.1:5000")
    print("Press Ctrl+C to terminate the server.")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)

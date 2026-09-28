# Project 9 - Smart City Traffic Forecasting

## 1. Project Overview
This project focuses on forecasting smart-city traffic patterns using machine learning regression techniques. By analyzing historical hourly vehicle counts collected across multiple city junctions, the system learns temporal traffic dynamics, diurnal rhythms, and day-of-week fluctuations to forecast future hourly traffic volumes across the city's road network.

The entire solution is implemented as a modular, standalone Python pipeline without relying on notebook environments.

---

## 2. Problem Statement
Urban traffic congestion leads to travel delays, elevated fuel consumption, increased emissions, and safety hazards. Modern smart cities rely on intelligent transportation systems (ITS) to optimize signal timings, allocate traffic management resources, and prevent bottleneck formations.

Accurate traffic forecasting across different junctions presents key challenges:
- **Spatial Variation:** Traffic density and peak hours vary significantly between different junctions.
- **Temporal Seasonality:** Traffic patterns exhibit strong hourly cycles (rush hours vs. overnight) and weekly patterns (weekdays vs. weekends).
- **Trend and Evolution:** Urban traffic volume shifts over months as vehicle adoption and commuter patterns evolve.

Forecasting traffic volumes enables municipal planners and automated traffic control systems to anticipate congestion hours in advance and take proactive measures.

---

## 3. Project Objective
The primary objective of this project is to build an end-to-end machine learning pipeline that:
1. Ingests and preprocesses multi-junction hourly traffic datasets.
2. Performs comprehensive Exploratory Data Analysis (EDA) to uncover temporal and junction-specific trends.
3. Engineers predictive calendar, cyclical, lag, and rolling statistics while strictly preventing data leakage.
4. Trains and evaluates multiple regression models on a chronological validation split.
5. Employs the best-performing model to generate recursive multi-step autoregressive traffic forecasts for the unlabelled test horizon (`Test.csv`).
6. Exports structured model comparisons and visualization artifacts for reporting and deployment.

---

## 4. Dataset
The project utilizes historical traffic data structured across two CSV files located in the `data/` directory:

### Train Dataset (`Train.csv`)
- **Records:** 48,120 rows
- **Columns:** 4 (`DateTime`, `Junction`, `Vehicles`, `ID`)
- **Date Range:** November 1, 2015, 00:00:00 to June 30, 2017, 23:00:00 (hourly frequency)
- **Target Variable:** `Vehicles` (non-negative integer representing the hourly vehicle count)
- **Junctions:** 4 distinct junctions (Junctions 1, 2, 3, and 4)
- **Missing / Duplicate Values:** 0 missing values, 0 duplicate records

### Test Dataset (`Test.csv`)
- **Records:** 11,808 rows
- **Columns:** 3 (`DateTime`, `Junction`, `ID`)
- **Date Range:** July 1, 2017, 00:00:00 to October 31, 2017, 23:00:00 (4-month hourly forecast horizon)
- **Target Variable:** Not present (unlabelled evaluation dataset)
- **Missing / Duplicate Values:** 0 missing values, 0 duplicate records

---

## 5. Technologies Used
The project is built entirely with Python and its core scientific computing libraries:
- **Python:** Core programming language
- **Pandas:** Structured data manipulation, datetime indexing, rolling window calculations, and tabular exports
- **NumPy:** Numerical computations and trigonometric transformations (sine/cosine)
- **Matplotlib:** Static figure creation, line charts, and metric comparison plots
- **Seaborn:** Statistical data visualizations, residual distributions, and bar charts
- **Scikit-learn:** Data modeling, linear and tree-based regression algorithms, and validation metrics (MAE, RMSE, $R^2$)

---

## 6. Project Workflow

```
Dataset (Train.csv & Test.csv)
       ↓
Data Preprocessing (Parsing, Sorting, Integrity Checks)
       ↓
Exploratory Data Analysis (EDA & Pattern Discovery)
       ↓
Feature Engineering (Calendar, Cyclical, Lags, Rolling Means)
       ↓
Model Training (Chronological Train/Validation Split)
       ↓
Model Evaluation & Comparison (MAE, RMSE, R² on Validation Set)
       ↓
Final Prediction (Recursive Multi-Step Autoregressive Forecasting)
```

---

## 7. Data Preprocessing
The preprocessing phase prepares the raw data for modeling:
1. **Dynamic File Detection:** Resolves file paths dynamically to handle potential filesystem formatting variations.
2. **Datetime Parsing:** Converts `DateTime` strings into native `pandas.Timestamp` objects.
3. **Chronological Ordering:** Sorts records sequentially by `DateTime` and `Junction` to preserve the natural arrow of time.
4. **Data Hygiene Verification:** Formally confirms zero missing values, zero duplicates, correct column data types, and contiguous junction tracking.

---

## 8. Exploratory Data Analysis
EDA uncovers several key traffic patterns across the network:
- **Junction Disparity:** 
  - Junction 1 exhibits the highest traffic density (mean ~46.5 vehicles/hour).
  - Junction 2 shows moderate density (mean ~14.4 vehicles/hour).
  - Junction 3 shows steady, lower volume (mean ~13.7 vehicles/hour).
  - Junction 4 represents a newly monitored junction starting in early 2017 with lower vehicle counts (mean ~7.3 vehicles/hour).
- **Diurnal Cycles:** Clear double-peak commute profile on weekdays—a morning peak around 09:00 and an evening peak between 17:00 and 19:00. Overnight hours (01:00 to 05:00) demonstrate consistent troughs.
- **Weekly Patterns:** Weekend traffic displays lower morning peaks and smoother midday curves compared to sharp weekday commuter peaks.
- **Long-Term Trend:** Year-over-year analysis demonstrates a continuous upward trend in total vehicle counts from 2015 through 2017.

---

## 9. Feature Engineering
A total of **22 predictive features** are engineered from the temporal and junction data:

### Date & Time Features
- `Year`, `Month`, `Day`, `Hour`: Basic calendar representations.
- `DayOfWeek`: Numerical day index (0 = Monday, 6 = Sunday).
- `DayOfYear`: Continuous progression through the year (1–366).
- `WeekOfYear`: ISO week number (1–53).
- `IsWeekend`: Binary indicator (1 for Saturday/Sunday, 0 for weekdays).
- `Traffic_Period`: Categorical time-of-day binning (`0: Night [00-05]`, `1: Morning [06-11]`, `2: Afternoon [12-17]`, `3: Evening [18-23]`).

### Cyclical Features
To preserve cyclical continuity (e.g., ensuring 23:00 and 00:00 are recognized as adjacent hours):
- `Hour_Sin`, `Hour_Cos`: $\sin\left(\frac{2\pi \cdot \text{Hour}}{24}\right)$ and $\cos\left(\frac{2\pi \cdot \text{Hour}}{24}\right)$
- `DayOfWeek_Sin`, `DayOfWeek_Cos`: $\sin\left(\frac{2\pi \cdot \text{DayOfWeek}}{7}\right)$ and $\cos\left(\frac{2\pi \cdot \text{DayOfWeek}}{7}\right)$

### Junction-Specific Historical Lag Features
To capture temporal autocorrelation without mixing observations across junctions, lag variables are computed grouped strictly by `Junction`:
- `Lag_1`, `Lag_2`, `Lag_3`: Immediate prior hourly traffic counts.
- `Lag_24`: Vehicle count at the exact same hour on the previous day.
- `Lag_48`: Vehicle count at the exact same hour two days prior.
- `Lag_168`: Vehicle count at the exact same hour and day of the previous week (full weekly cycle).

### Junction-Specific Rolling Statistics (Leakage-Free)
Rolling statistics are calculated grouped by `Junction` using `.shift(1)` to ensure the current observation is never included in its own rolling window:
- `Rolling_Mean_24`: Trailing 24-hour moving average traffic.
- `Rolling_Mean_168`: Trailing 168-hour (7-day) moving average traffic.

*Warm-up Period:* The initial 168 hours per junction (672 rows total) containing initial `NaN` lag/rolling values are trimmed, leaving **47,448 clean training observations**.

---

## 10. Machine Learning Models
To predict hourly vehicle counts, three regression architectures were implemented and trained using an 80/20 chronological split:
- **Training Set:** 37,956 samples (November 8, 2015 to March 24, 2017)
- **Validation Set:** 9,492 samples (March 24, 2017 to June 30, 2017)

### 1. Linear Regression
A standard parametric baseline model that fits linear coefficients to identify primary trend lines and direct feature effects.

### 2. Decision Tree Regressor
A non-parametric tree algorithm (`random_state=42`) capable of capturing non-linear interactions between junction identity, hour of the day, and rolling traffic trends.

### 3. Random Forest Regressor
An ensemble bagging algorithm (`n_estimators=100`, `random_state=42`, `n_jobs=-1`) that constructs multiple uncorrelated decision trees and aggregates their predictions, reducing variance and mitigating overfitting.

---

## 11. Model Evaluation
Model performance was evaluated on the unseen chronological validation split using three standard regression metrics:
- **Mean Absolute Error (MAE):** Average magnitude of prediction errors in vehicles.
- **Root Mean Squared Error (RMSE):** Error metric penalizing larger deviations more severely.
- **Coefficient of Determination ($R^2$):** Proportion of variance in traffic counts explained by the model.

### Validation Results

| Model | MAE | RMSE | R² |
| :--- | ---: | ---: | ---: |
| **Random Forest** | **2.9065** | **4.7729** | **0.9692** |
| Linear Regression | 3.0821 | 4.8003 | 0.9689 |
| Decision Tree | 4.2498 | 6.7260 | 0.9389 |

> **Note on Evaluation Methodology:**  
> All evaluation metrics were computed strictly on the historical chronological validation partition where ground-truth vehicle counts were available. `Test.csv` contains unlabelled future records without target values; therefore, it was not and cannot be used to calculate error metrics.

---

## 12. Final Prediction
Based on the validation results, **Random Forest Regressor** was selected as the final production model due to achieving the lowest RMSE (4.7729), lowest MAE (2.9065), and highest $R^2$ (0.9692).

### Autoregressive Multi-Step Test Forecasting
Generating predictions across a 4-month horizon (11,808 hourly records) where lag features depend on prior traffic counts requires recursive forecasting:
1. For each junction, predictions are generated step-by-step in chronological sequence.
2. At each time step, calendar and cyclical features are computed from the test timestamp.
3. Lag and rolling window values are dynamically updated using a rolling history buffer that incorporates both actual historical counts and newly predicted values.
4. Predictions are rounded to two decimal places and bounded to non-negative values.

### Output File
The final predictions are exported to the project root:
- **File:** `traffic_predictions.csv`
- **Rows:** 11,808 records
- **Columns:** `ID`, `DateTime`, `Junction`, `Predicted_Traffic`

---

## 13. Results & Visualizations
The pipeline generates and saves eight analytical charts into the `results/figures/` directory:

1. **`model_mae_comparison.png`:** Bar chart comparing validation MAE across the evaluated models.
2. **`model_rmse_comparison.png`:** Bar chart comparing validation RMSE across the evaluated models.
3. **`model_r2_comparison.png`:** Bar chart illustrating $R^2$ goodness-of-fit scores.
4. **`actual_vs_predicted.png`:** Scatter plot comparing actual vs. predicted validation traffic with the $y = x$ reference line.
5. **`residual_distribution.png`:** Histogram and KDE curve of prediction errors centered closely around zero.
6. **`traffic_by_junction.png`:** Historical mean vehicle volume comparison across the four city junctions.
7. **`predicted_traffic_distribution.png`:** Distribution profile of predicted test traffic counts.
8. **`predicted_traffic_over_time.png`:** Full chronological time-series visualization of forecasted traffic across the test horizon.

Additionally, the evaluation metrics are exported in tabular format to:
- `results/model_comparison.csv`

---

## 14. Project Structure
The repository is organized as follows:

```
Project-09-Smart-City-Traffic-Forecasting/
│
├── data/
│   ├── Train.csv
│   └── Test.csv
│
├── models/
│   └── traffic_model.pkl
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
├── templates/
│   └── index.html
│
├── results/
│   ├── model_comparison.csv
│   └── figures/
│       ├── actual_vs_predicted.png
│       ├── model_mae_comparison.png
│       ├── model_r2_comparison.png
│       ├── model_rmse_comparison.png
│       ├── predicted_traffic_distribution.png
│       ├── predicted_traffic_over_time.png
│       ├── residual_distribution.png
│       └── traffic_by_junction.png
│
├── traffic_forecasting.py
├── app.py
├── traffic_predictions.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 15. How to Run the Project

### Prerequisites & Installation
Ensure Python 3.8+ is installed on your system. Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### 1. Running the Machine Learning Pipeline
To execute the complete ML pipeline, train models, evaluate validation metrics, generate the test forecasts, export all diagnostic visualizations, and save the production model:

```bash
python traffic_forecasting.py
```

To run in non-interactive/headless mode:

```bash
python traffic_forecasting.py --no-plots
```

This ensures `models/traffic_model.pkl`, `traffic_predictions.csv`, and all figures in `results/figures/` are generated.

### 2. Launching the Interactive Web Application
Launch the Flask dashboard to interactively forecast traffic flows and inspect live analytics:

```bash
python app.py
```

Once running, navigate to **`http://127.0.0.1:5000`** in any web browser.

---

## 16. Key Learnings
- **Time-Series Validation Discipline:** Traditional random cross-validation leaks future information into past predictions. Preserving a strict chronological split is essential for obtaining honest generalization estimates.
- **Autoregressive Feature Engineering:** Lags and rolling averages provide substantial predictive power for short-term and medium-term traffic patterns, capturing diurnal rhythms (24h) and weekly cycles (168h).
- **Target Leakage Prevention:** Computing rolling aggregates must be coupled with explicit shifting (`shift(1)`) so that the target variable of the current observation is never factored into its own feature set.
- **Multi-Entity Time Series:** Features like lags and moving averages must be calculated independently per junction to avoid cross-junction signal corruption.
- **Recursive Forecasting Mechanics:** Multi-step forecasting into unlabelled horizons requires an autoregressive simulation loop where earlier predictions dynamically serve as lag inputs for subsequent steps.

---

## 17. Internship Context
This project was developed as part of the **Machine Learning Internship by UCT** under **Project 9: Forecasting of Smart City Traffic Patterns**.

---

## 18. Future Improvements
The following technical enhancements can be considered for future iterations:
- **Hyperparameter Optimization:** Conduct systematic tuning (e.g., via Optuna or Bayesian Search) for tree depth, minimum samples per split, and feature subsampling.
- **Gradient Boosting Models:** Benchmark gradient-boosted decision trees such as LightGBM, XGBoost, or CatBoost for improved inference speed and potential metric gains.
- **Exogenous Holiday Calendars:** Integrate formal national and local holiday calendars to better capture unusual traffic drops during public holidays.
- **External Weather & Incident Feeds:** Incorporate real-time meteorological observations (precipitation, fog, temperature) and reported road closures.
- **Service Deployment:** Wrap the forecasting model in a lightweight REST API (such as FastAPI) to serve live inference requests to traffic management control systems.

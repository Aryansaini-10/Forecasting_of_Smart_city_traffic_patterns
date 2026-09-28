/**
 * Project 9 - Smart City Traffic Forecasting
 * Frontend Controller & Chart Visualizer
 */

document.addEventListener("DOMContentLoaded", () => {
    initQuickPresets();
    initPredictionForm();
    initFeatureToggle();
    loadAnalyticsCharts();
});

/**
 * Configure Quick Selection Buttons for common traffic scenarios
 */
function initQuickPresets() {
    const presetBtns = document.querySelectorAll(".preset-btn");
    const dateInput = document.getElementById("forecast-date");
    const timeInput = document.getElementById("forecast-time");
    const junctionSelect = document.getElementById("forecast-junction");

    presetBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const time = btn.dataset.time;
            const junction = btn.dataset.junction;
            const date = btn.dataset.date;

            if (time && timeInput) timeInput.value = time;
            if (junction && junctionSelect) junctionSelect.value = junction;
            if (date && dateInput) dateInput.value = date;

            // Trigger prediction automatically on preset selection
            submitPrediction();
        });
    });
}

/**
 * Initialize Prediction Form submission via fetch()
 */
function initPredictionForm() {
    const form = document.getElementById("prediction-form");
    if (!form) return;

    form.addEventListener("submit", (e) => {
        e.preventDefault();
        submitPrediction();
    });
}

/**
 * Submit prediction request to Flask /predict route
 */
async function submitPrediction() {
    const junctionSelect = document.getElementById("forecast-junction");
    const dateInput = document.getElementById("forecast-date");
    const timeInput = document.getElementById("forecast-time");
    const btnForecast = document.getElementById("btn-forecast");
    const spinner = document.getElementById("btn-spinner");
    const btnText = document.getElementById("btn-text");
    const errorAlert = document.getElementById("prediction-error");

    // Hide previous error
    if (errorAlert) {
        errorAlert.style.display = "none";
        errorAlert.textContent = "";
    }

    const junctionVal = parseInt(junctionSelect.value, 10);
    const dateVal = dateInput.value.trim();
    const timeVal = timeInput.value.trim();

    // Client-side validation
    if (!junctionVal || junctionVal < 1 || junctionVal > 4) {
        showError("Please select a valid junction (1-4).");
        return;
    }
    if (!dateVal) {
        showError("Please specify a valid forecast date.");
        return;
    }
    if (!timeVal) {
        showError("Please specify a forecast time (HH:MM).");
        return;
    }

    // Set UI loading state
    btnForecast.disabled = true;
    if (spinner) spinner.style.display = "inline-block";
    if (btnText) btnText.textContent = "Forecasting...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                junction: junctionVal,
                date: dateVal,
                time: timeVal
            })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Prediction request failed.");
        }

        renderPredictionResult(data);
    } catch (err) {
        showError(err.message || "An unexpected error occurred while forecasting traffic.");
    } finally {
        btnForecast.disabled = false;
        if (spinner) spinner.style.display = "none";
        if (btnText) btnText.textContent = "Forecast Traffic";
    }
}

/**
 * Display prediction output and feature breakdown
 */
function renderPredictionResult(data) {
    const placeholder = document.getElementById("result-placeholder");
    const display = document.getElementById("result-display");
    const numberEl = document.getElementById("predicted-number");
    const levelBadge = document.getElementById("traffic-level");
    const metaJunction = document.getElementById("result-junction");
    const metaDateTime = document.getElementById("result-datetime");
    const featureBody = document.getElementById("feature-table-body");

    if (placeholder) placeholder.style.display = "none";
    if (display) display.style.display = "block";

    // Format vehicle count
    if (numberEl) {
        numberEl.textContent = Number(data.prediction).toFixed(2);
    }

    // Apply derived traffic level badge
    if (levelBadge) {
        levelBadge.textContent = `${data.level} Traffic`;
        levelBadge.className = "traffic-level-badge";
        if (data.level === "Low") {
            levelBadge.classList.add("level-low");
        } else if (data.level === "Moderate") {
            levelBadge.classList.add("level-moderate");
        } else {
            levelBadge.classList.add("level-heavy");
        }
    }

    if (metaJunction) metaJunction.textContent = `Junction ${data.junction}`;
    if (metaDateTime) metaDateTime.textContent = data.datetime;

    // Populate feature values breakdown table
    if (featureBody && data.features) {
        featureBody.innerHTML = "";
        const features = data.features;
        for (const [key, val] of Object.entries(features)) {
            const row = document.createElement("tr");
            const th = document.createElement("th");
            th.textContent = key;
            const td = document.createElement("td");
            td.textContent = typeof val === "number" ? (Number.isInteger(val) ? val : val.toFixed(4)) : val;
            row.appendChild(th);
            row.appendChild(td);
            featureBody.appendChild(row);
        }
    }
}

/**
 * Feature details expand/collapse toggle
 */
function initFeatureToggle() {
    const toggleBtn = document.getElementById("feature-toggle");
    const table = document.getElementById("feature-table");

    if (toggleBtn && table) {
        toggleBtn.addEventListener("click", () => {
            const isHidden = table.style.display === "none" || table.style.display === "";
            table.style.display = isHidden ? "table" : "none";
            toggleBtn.innerHTML = isHidden 
                ? "<span>Hide Pipeline Features ▲</span>" 
                : "<span>View 22 Pipeline Features ▼</span>";
        });
    }
}

/**
 * Display error banner
 */
function showError(msg) {
    const errorAlert = document.getElementById("prediction-error");
    if (errorAlert) {
        errorAlert.textContent = msg;
        errorAlert.style.display = "block";
    }
}

/**
 * Load and render interactive Chart.js analytics charts
 */
async function loadAnalyticsCharts() {
    try {
        const response = await fetch("/api/analytics");
        if (!response.ok) return;
        const data = await response.json();

        renderJunctionChart(data.junction_stats);
        renderHourlyChart(data.hourly_profile);
        renderPerformanceChart(data.model_comparison);
    } catch (e) {
        console.warn("Analytics data fetch notice:", e);
    }
}

/**
 * Chart 1: Traffic by Junction (Historical vs. Predicted)
 */
function renderJunctionChart(junctionData) {
    const ctx = document.getElementById("junctionChart");
    if (!ctx || !junctionData) return;

    const labels = ["Junction 1", "Junction 2", "Junction 3", "Junction 4"];
    const histMeans = [junctionData[1].historical_mean, junctionData[2].historical_mean, junctionData[3].historical_mean, junctionData[4].historical_mean];
    const predMeans = [junctionData[1].predicted_mean, junctionData[2].predicted_mean, junctionData[3].predicted_mean, junctionData[4].predicted_mean];

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Historical Average (Train)",
                    data: histMeans,
                    backgroundColor: "rgba(59, 130, 246, 0.7)",
                    borderColor: "rgba(59, 130, 246, 1)",
                    borderWidth: 1,
                    borderRadius: 4
                },
                {
                    label: "Predicted Average (Test Horizon)",
                    data: predMeans,
                    backgroundColor: "rgba(6, 182, 212, 0.7)",
                    borderColor: "rgba(6, 182, 212, 1)",
                    borderWidth: 1,
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: "#94a3b8", font: { size: 11 } }
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw} vehicles/hr`
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: "#94a3b8" },
                    grid: { color: "rgba(255, 255, 255, 0.05)" }
                },
                y: {
                    title: { display: true, text: "Average Vehicles / Hour", color: "#94a3b8" },
                    ticks: { color: "#94a3b8" },
                    grid: { color: "rgba(255, 255, 255, 0.08)" }
                }
            }
        }
    });
}

/**
 * Chart 2: Hourly Diurnal Traffic Pattern
 */
function renderHourlyChart(hourlyProfile) {
    const ctx = document.getElementById("hourlyChart");
    if (!ctx || !hourlyProfile) return;

    const hours = Array.from({ length: 24 }, (_, i) => `${String(i).padStart(2, "0")}:00`);
    const values = hourlyProfile;

    new Chart(ctx, {
        type: "line",
        data: {
            labels: hours,
            datasets: [{
                label: "Average Vehicle Count",
                data: values,
                borderColor: "#06b6d4",
                backgroundColor: "rgba(6, 182, 212, 0.15)",
                fill: true,
                tension: 0.35,
                borderWidth: 2,
                pointBackgroundColor: "#06b6d4",
                pointBorderColor: "#fff",
                pointRadius: 3,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: "#94a3b8", font: { size: 11 } }
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` Hourly Average: ${ctx.raw} vehicles`
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: "#94a3b8", maxTicksLimit: 12 },
                    grid: { color: "rgba(255, 255, 255, 0.05)" }
                },
                y: {
                    title: { display: true, text: "Vehicles Count", color: "#94a3b8" },
                    ticks: { color: "#94a3b8" },
                    grid: { color: "rgba(255, 255, 255, 0.08)" }
                }
            }
        }
    });
}

/**
 * Chart 3: Model Performance Comparison
 */
function renderPerformanceChart(modelData) {
    const ctx = document.getElementById("performanceChart");
    if (!ctx || !modelData) return;

    const labels = modelData.map(m => m.Model);
    const mae = modelData.map(m => m.MAE);
    const rmse = modelData.map(m => m.RMSE);

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "MAE (Lower is Better)",
                    data: mae,
                    backgroundColor: "rgba(16, 185, 129, 0.7)",
                    borderColor: "rgba(16, 185, 129, 1)",
                    borderWidth: 1,
                    borderRadius: 4
                },
                {
                    label: "RMSE (Lower is Better)",
                    data: rmse,
                    backgroundColor: "rgba(244, 63, 94, 0.7)",
                    borderColor: "rgba(244, 63, 94, 1)",
                    borderWidth: 1,
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: "#94a3b8", font: { size: 11 } }
                }
            },
            scales: {
                x: {
                    ticks: { color: "#94a3b8" },
                    grid: { color: "rgba(255, 255, 255, 0.05)" }
                },
                y: {
                    title: { display: true, text: "Error in Vehicles", color: "#94a3b8" },
                    ticks: { color: "#94a3b8" },
                    grid: { color: "rgba(255, 255, 255, 0.08)" }
                }
            }
        }
    });
}

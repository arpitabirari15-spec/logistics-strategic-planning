"""Predictive modeling for delivery cycle time (Week 4).

Trains and compares Linear Regression, Random Forest, and Gradient Boosting
models to forecast cycle_time_days from order/shipment features, then
produces evaluation metrics, feature importance, and diagnostic plots.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = ["#1B2A4A", "#3D5A80", "#E07A3E", "#7A9CC6", "#B0783F", "#5B6472"]
OUT = "/home/claude/week4/charts"

df = pd.read_csv("/home/claude/week4/data/logistics_dataset.csv", parse_dates=["order_date"])
df = df.dropna(subset=["order_value", "transportation_cost"]).copy()

CAT_FEATURES = ["region", "courier", "pincode_tier", "payment_mode"]
NUM_FEATURES = ["distance_km", "shipment_volume", "order_value"]
TARGET = "cycle_time_days"

X = df[CAT_FEATURES + NUM_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
], remainder="passthrough")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

results = []
predictions = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    predictions[name] = preds
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
    results.append({
        "Model": name, "RMSE": round(rmse, 3), "MAE": round(mae, 3), "R2": round(r2, 3),
        "CV RMSE (mean)": round(-cv_scores.mean(), 3), "CV RMSE (std)": round(cv_scores.std(), 3),
    })

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT}/model_comparison.csv", index=False)
print(results_df)

# ---------- Hyperparameter tuning on the best-performing model (Gradient Boosting) ----------
gb_pipe = Pipeline([("prep", preprocess), ("model", GradientBoostingRegressor(random_state=42))])
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [2, 3, 4],
    "model__learning_rate": [0.05, 0.1],
}
grid = GridSearchCV(gb_pipe, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
best_preds = best_model.predict(X_test)
best_rmse = np.sqrt(mean_squared_error(y_test, best_preds))
best_mae = mean_absolute_error(y_test, best_preds)
best_r2 = r2_score(y_test, best_preds)
print("Best params:", grid.best_params_)
print(f"Tuned GB -> RMSE={best_rmse:.3f}, MAE={best_mae:.3f}, R2={best_r2:.3f}")

pd.DataFrame([{
    "Best Params": str(grid.best_params_), "RMSE": round(best_rmse, 3),
    "MAE": round(best_mae, 3), "R2": round(best_r2, 3),
}]).to_csv(f"{OUT}/tuned_model_result.csv", index=False)

# ---------- Chart 1: Model comparison bar chart (RMSE) ----------
fig, ax = plt.subplots(figsize=(7, 4.2))
sns.barplot(data=results_df, x="Model", y="RMSE", ax=ax, palette=PALETTE[:3])
ax.set_title("Model Comparison: Test RMSE (Cycle Time, days)", fontsize=13, weight="bold", color=PALETTE[0])
ax.set_ylabel("RMSE (days)")
plt.tight_layout()
plt.savefig(f"{OUT}/01_model_comparison_rmse.png", dpi=150)
plt.close()

# ---------- Chart 2: Actual vs Predicted (tuned GB model) ----------
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.scatter(y_test, best_preds, alpha=0.4, color=PALETTE[1], s=20)
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
ax.plot(lims, lims, color=PALETTE[2], linestyle="--", linewidth=2, label="Perfect prediction")
ax.set_xlabel("Actual Cycle Time (days)")
ax.set_ylabel("Predicted Cycle Time (days)")
ax.set_title("Actual vs. Predicted Cycle Time (Tuned Gradient Boosting)", fontsize=12, weight="bold", color=PALETTE[0])
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/02_actual_vs_predicted.png", dpi=150)
plt.close()

# ---------- Chart 3: Residual plot ----------
residuals = y_test.values - best_preds
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(best_preds, residuals, alpha=0.4, color=PALETTE[1], s=20)
ax.axhline(0, color=PALETTE[2], linestyle="--", linewidth=2)
ax.set_xlabel("Predicted Cycle Time (days)")
ax.set_ylabel("Residual (Actual \u2212 Predicted)")
ax.set_title("Residual Plot (Tuned Gradient Boosting)", fontsize=12, weight="bold", color=PALETTE[0])
plt.tight_layout()
plt.savefig(f"{OUT}/03_residual_plot.png", dpi=150)
plt.close()

# ---------- Chart 4: Feature importance ----------
feature_names = best_model.named_steps["prep"].get_feature_names_out()
importances = best_model.named_steps["model"].feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(10)
imp_df["feature"] = imp_df["feature"].str.replace("cat__", "").str.replace("remainder__", "")
fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.barplot(data=imp_df, y="feature", x="importance", ax=ax, palette=[PALETTE[0]] * len(imp_df))
ax.set_title("Top 10 Feature Importances (Tuned Gradient Boosting)", fontsize=12, weight="bold", color=PALETTE[0])
ax.set_xlabel("Relative Importance")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT}/04_feature_importance.png", dpi=150)
plt.close()

print("Charts saved to", OUT)

"""Region-level order-volume forecasting with a gradient-boosted regressor."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

FEATURES = ["week", "region_encoded", "rolling_avg_3w", "is_festival_week", "pincode_tier"]


def train_demand_model(weekly_demand: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        weekly_demand[FEATURES],
        weekly_demand["order_count"],
        test_size=0.2,
        shuffle=False,  # preserve time order
    )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"MAE: {mae:.2f}")
    return model


if __name__ == "__main__":
    weekly_demand = pd.read_csv("weekly_demand_features.csv")
    train_demand_model(weekly_demand)

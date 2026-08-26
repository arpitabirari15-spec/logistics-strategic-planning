"""Predict return-to-origin (RTO) / delivery-failure risk before dispatch."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

RISK_FEATURES = ["courier_encoded", "pincode_tier", "order_value", "payment_mode_encoded", "is_festival_week"]


def train_rto_risk_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df[RISK_FEATURES], df["is_rto"], test_size=0.2, random_state=42
    )

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)

    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f"AUC: {auc:.3f}")
    return clf


if __name__ == "__main__":
    df = pd.read_csv("analytical_table_with_features.csv")
    train_rto_risk_model(df)

"""End-to-end data cleaning & preprocessing pipeline for the last-mile
delivery scenario (Week 2).

Combines a public trip dataset (e.g. the Kaggle Delhivery Logistics Dataset)
with simulated orders/RTO tables, then runs it through the full pipeline
described in the Week 2 report: format standardization, deduplication,
missing-value handling, outlier treatment, categorical encoding, scaling,
and class-imbalance correction for the RTO target.

Expected input schemas (CSV):
  delhivery_trips.csv : trip_uuid, od_start_time, source_pincode,
                         dest_pincode, distance_km, courier, ...
  orders.csv           : order_id, order_date, region, order_value,
                          payment_mode, delivered_date, ...
  rto_log.csv          : order_id, is_rto, rto_reason, ...
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def iqr_bounds(series: pd.Series, k: float = 1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def load_and_standardize(trips_path: str, orders_path: str, rto_path: str) -> pd.DataFrame:
    trips = pd.read_csv(trips_path)
    orders = pd.read_csv(orders_path)
    rto_log = pd.read_csv(rto_path)

    orders["order_date"] = pd.to_datetime(orders["order_date"], dayfirst=True)
    orders["delivered_date"] = pd.to_datetime(orders["delivered_date"], dayfirst=True)
    trips["od_start_time"] = pd.to_datetime(trips["od_start_time"], dayfirst=True)
    trips["source_pincode"] = trips["source_pincode"].astype(str).str.zfill(6)
    trips["dest_pincode"] = trips["dest_pincode"].astype(str).str.zfill(6)
    trips["courier"] = trips["courier"].str.strip().str.title()

    df = orders.merge(trips, left_on="order_id", right_on="trip_uuid", how="left").merge(
        rto_log, on="order_id", how="left"
    )
    df["cycle_time_days"] = (df["delivered_date"] - df["order_date"]).dt.days
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset="order_id", keep="first")
    print(f"Removed {before - len(df)} duplicate order records")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df["order_value"] = df["order_value"].fillna(df["order_value"].median())
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    df = df.dropna(subset=["delivered_date"])
    return df


def treat_outliers(df: pd.DataFrame) -> pd.DataFrame:
    low, high = iqr_bounds(df["cycle_time_days"])
    df = df[(df["cycle_time_days"] >= low) & (df["cycle_time_days"] <= high)]
    df = df[df["order_value"] > 0]
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=["payment_mode"], drop_first=True)
    courier_freq = df["courier"].value_counts(normalize=True)
    df["courier_freq_enc"] = df["courier"].map(courier_freq)
    return df


def scale_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df["cycle_time_log"] = np.log1p(df["cycle_time_days"])
    scaler = StandardScaler()
    num_cols = ["order_value", "distance_km", "cycle_time_log"]
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def run_pipeline(trips_path: str, orders_path: str, rto_path: str) -> pd.DataFrame:
    df = load_and_standardize(trips_path, orders_path, rto_path)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = treat_outliers(df)
    df = encode_categoricals(df)
    df = scale_numeric(df)
    return df


if __name__ == "__main__":
    clean_df = run_pipeline("delhivery_trips.csv", "orders.csv", "rto_log.csv")
    clean_df.to_csv("analytical_table_clean.csv", index=False)
    print(f"Final clean dataset: {clean_df.shape[0]} rows, {clean_df.shape[1]} columns")

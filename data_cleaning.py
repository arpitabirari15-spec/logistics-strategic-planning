"""Merge order, courier-trip, and RTO records into one analytical table.

Expected input schemas (CSV):
  orders.csv         : order_id, order_date, region, order_value, payment_mode, ...
  courier_trips.csv  : order_id, pincode, courier, delivered_date, ...
  rto_log.csv        : order_id, is_rto, rto_reason, ...
"""
import pandas as pd


def build_analytical_table(orders_path: str, courier_trips_path: str, rto_log_path: str) -> pd.DataFrame:
    orders = pd.read_csv(orders_path)
    courier_trips = pd.read_csv(courier_trips_path)
    rto_log = pd.read_csv(rto_log_path)

    orders["order_date"] = pd.to_datetime(orders["order_date"])
    courier_trips["pincode"] = courier_trips["pincode"].astype(str).str.zfill(6)

    df = (
        orders.merge(courier_trips, on="order_id", how="left")
        .merge(rto_log, on="order_id", how="left")
    )

    df["cycle_time_days"] = (df["delivered_date"] - df["order_date"]).dt.days
    df = df.dropna(subset=["cycle_time_days"])
    return df


if __name__ == "__main__":
    result = build_analytical_table("orders.csv", "courier_trips.csv", "rto_log.csv")
    print(result.head())

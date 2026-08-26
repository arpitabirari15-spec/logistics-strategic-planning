"""Simulate a hypothetical last-mile delivery dataset for a D2C e-commerce
brand (consistent with the Week 1-2 scenario), with realistic distributions
and correlations between shipment volume, distance, transportation cost,
delivery time, and RTO risk.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 3000

REGIONS = ["North", "South", "East", "West", "Central"]
COURIERS = ["Delhivery", "Bluedart", "Ecom Express", "Xpressbees", "Shadowfax"]
PINCODE_TIERS = ["Metro", "Tier-2", "Tier-3"]
TIER_WEIGHTS = [0.45, 0.35, 0.20]
PAYMENT_MODES = ["Prepaid", "COD"]

df = pd.DataFrame({
    "order_id": [f"ORD{100000+i}" for i in range(N)],
    "region": np.random.choice(REGIONS, N),
    "courier": np.random.choice(COURIERS, N, p=[0.30, 0.15, 0.25, 0.18, 0.12]),
    "pincode_tier": np.random.choice(PINCODE_TIERS, N, p=TIER_WEIGHTS),
    "payment_mode": np.random.choice(PAYMENT_MODES, N, p=[0.62, 0.38]),
})

# Order date over a 6-month window, with extra order density in June (sale event)
all_days = pd.date_range("2026-03-01", "2026-08-25", freq="D")
day_weights = np.where(all_days.month == 6, 2.2, 1.0)
day_weights = day_weights / day_weights.sum()
df["order_date"] = np.random.choice(all_days, size=N, p=day_weights)
df["order_date"] = pd.to_datetime(df["order_date"])
df["month"] = df["order_date"].dt.month

# Distance depends on pincode tier (Tier-2/3 tend to be farther from hubs)
tier_distance_base = df["pincode_tier"].map({"Metro": 12, "Tier-2": 45, "Tier-3": 95})
df["distance_km"] = np.round(np.random.gamma(shape=3.0, scale=tier_distance_base / 3.0), 1)

# Shipment volume (units per order, mostly 1-2, occasional bulk)
df["shipment_volume"] = np.random.choice([1, 2, 3, 4, 5], N, p=[0.55, 0.25, 0.10, 0.06, 0.04])

# Order value correlated with shipment volume, with noise
df["order_value"] = np.round(
    (df["shipment_volume"] * np.random.normal(650, 150, N)).clip(150, None), 2
)

# Transportation cost: base + per-km + per-unit-volume + courier premium/discount
courier_cost_factor = df["courier"].map({
    "Delhivery": 1.00, "Bluedart": 1.25, "Ecom Express": 0.90, "Xpressbees": 0.85, "Shadowfax": 0.80
})
df["transportation_cost"] = np.round(
    (35 + df["distance_km"] * 3.8 + df["shipment_volume"] * 12) * courier_cost_factor
    + np.random.normal(0, 15, N),
    2,
).clip(20, None)

# Delivery cycle time: driven by distance, pincode tier, and a bit of courier + season noise
tier_delay = df["pincode_tier"].map({"Metro": 0.4, "Tier-2": 1.3, "Tier-3": 2.6})
sale_event_bump = df["month"].isin([6]).astype(int) * np.random.uniform(0.5, 1.5, N)
df["cycle_time_days"] = np.round(
    1.2 + df["distance_km"] / 40 + tier_delay + sale_event_bump + np.random.gamma(1.5, 0.4, N), 1
)

# Inject a few outliers/edge cases to mirror real messy data
outlier_idx = np.random.choice(df.index, size=25, replace=False)
df.loc[outlier_idx, "cycle_time_days"] = np.random.uniform(15, 30, 25)

# RTO probability rises with cycle time, Tier-3, and COD
rto_logit = (
    -3.2
    + 0.08 * df["cycle_time_days"]
    + df["pincode_tier"].map({"Metro": 0, "Tier-2": 0.5, "Tier-3": 1.1})
    + df["payment_mode"].map({"Prepaid": 0, "COD": 0.9})
)
rto_prob = 1 / (1 + np.exp(-rto_logit))
df["is_rto"] = (np.random.rand(N) < rto_prob).astype(int)

# A few missing values to mirror real-world data quality issues (handled in Week 2 pipeline)
for col, frac in [("order_value", 0.01), ("transportation_cost", 0.008)]:
    miss_idx = np.random.choice(df.index, size=int(N * frac), replace=False)
    df.loc[miss_idx, col] = np.nan

df = df.drop(columns=["month"])
df.to_csv("/home/claude/week3/data/logistics_dataset.csv", index=False)
print(df.shape)
print(df.head())
print(df.isna().sum())

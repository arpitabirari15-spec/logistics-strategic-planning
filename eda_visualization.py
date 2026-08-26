"""Exploratory data analysis and visualization for the last-mile delivery
dataset (Week 3). Produces summary statistics and a set of charts saved as
PNG for embedding in the Week 3 report.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = ["#1B2A4A", "#3D5A80", "#E07A3E", "#7A9CC6", "#B0783F", "#5B6472"]
sns.set_palette(PALETTE)

OUT = "/home/claude/week3/charts"

df = pd.read_csv("/home/claude/week3/data/logistics_dataset.csv", parse_dates=["order_date"])
df_clean = df.dropna(subset=["order_value", "transportation_cost"]).copy()

# ---------- 1. Central tendency / summary stats ----------
summary = df_clean[["distance_km", "shipment_volume", "order_value",
                     "transportation_cost", "cycle_time_days"]].describe().T
summary["skew"] = df_clean[summary.index].skew()
summary.to_csv(f"{OUT}/summary_stats.csv")
print(summary)

# ---------- 2. Distribution of delivery cycle time ----------
fig, ax = plt.subplots(figsize=(7, 4.2))
sns.histplot(df_clean["cycle_time_days"], bins=40, kde=True, color=PALETTE[1], ax=ax)
ax.axvline(df_clean["cycle_time_days"].median(), color=PALETTE[2], linestyle="--", label="Median")
ax.set_title("Distribution of Delivery Cycle Time", fontsize=13, weight="bold", color=PALETTE[0])
ax.set_xlabel("Cycle Time (days)")
ax.set_ylabel("Number of Orders")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/01_cycle_time_distribution.png", dpi=150)
plt.close()

# ---------- 3. Transportation cost by courier (boxplot) ----------
fig, ax = plt.subplots(figsize=(7.5, 4.2))
order = df_clean.groupby("courier")["transportation_cost"].median().sort_values().index
sns.boxplot(data=df_clean, x="courier", y="transportation_cost", order=order, ax=ax, palette=PALETTE)
ax.set_title("Transportation Cost by Courier Partner", fontsize=13, weight="bold", color=PALETTE[0])
ax.set_xlabel("Courier")
ax.set_ylabel("Transportation Cost (₹)")
plt.tight_layout()
plt.savefig(f"{OUT}/02_cost_by_courier.png", dpi=150)
plt.close()

# ---------- 4. Cycle time by pincode tier ----------
fig, ax = plt.subplots(figsize=(7, 4.2))
tier_order = ["Metro", "Tier-2", "Tier-3"]
sns.violinplot(data=df_clean, x="pincode_tier", y="cycle_time_days", order=tier_order, ax=ax, palette=PALETTE)
ax.set_title("Delivery Cycle Time by Pincode Tier", fontsize=13, weight="bold", color=PALETTE[0])
ax.set_xlabel("Pincode Tier")
ax.set_ylabel("Cycle Time (days)")
plt.tight_layout()
plt.savefig(f"{OUT}/03_cycle_time_by_tier.png", dpi=150)
plt.close()

# ---------- 5. Distance vs cycle time scatter (with RTO hue) ----------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
plot_df = df_clean.sample(900, random_state=1).copy()
plot_df["RTO Outcome"] = plot_df["is_rto"].map({0: "Delivered", 1: "Returned (RTO)"})
sns.scatterplot(data=plot_df, x="distance_km", y="cycle_time_days",
                 hue="RTO Outcome", hue_order=["Delivered", "Returned (RTO)"],
                 palette={"Delivered": PALETTE[1], "Returned (RTO)": PALETTE[2]},
                 alpha=0.7, ax=ax, s=32)
ax.set_title("Distance vs. Delivery Cycle Time (colored by RTO)", fontsize=13, weight="bold", color=PALETTE[0])
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Cycle Time (days)")
ax.legend(title="RTO Outcome")
plt.tight_layout()
plt.savefig(f"{OUT}/04_distance_vs_cycletime.png", dpi=150)
plt.close()

# ---------- 6. Correlation heatmap ----------
num_cols = ["distance_km", "shipment_volume", "order_value", "transportation_cost", "cycle_time_days", "is_rto"]
corr = df_clean[num_cols].corr()
fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix of Key Logistics Variables", fontsize=13, weight="bold", color=PALETTE[0])
plt.tight_layout()
plt.savefig(f"{OUT}/05_correlation_heatmap.png", dpi=150)
plt.close()

# ---------- 7. Weekly shipment volume trend ----------
weekly = df_clean.set_index("order_date").resample("W")["shipment_volume"].sum().reset_index()
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(weekly["order_date"], weekly["shipment_volume"], color=PALETTE[0], linewidth=2, marker="o", markersize=3)
ax.fill_between(weekly["order_date"], weekly["shipment_volume"], color=PALETTE[1], alpha=0.15)
ax.set_title("Weekly Shipment Volume Trend", fontsize=13, weight="bold", color=PALETTE[0])
ax.set_xlabel("Week")
ax.set_ylabel("Total Units Shipped")
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(f"{OUT}/06_weekly_volume_trend.png", dpi=150)
plt.close()

# ---------- 8. RTO rate by region and payment mode ----------
rto_pivot = df_clean.groupby(["region", "payment_mode"])["is_rto"].mean().reset_index()
fig, ax = plt.subplots(figsize=(7.5, 4.2))
sns.barplot(data=rto_pivot, x="region", y="is_rto", hue="payment_mode", ax=ax, palette=[PALETTE[1], PALETTE[2]])
ax.set_title("RTO Rate by Region and Payment Mode", fontsize=13, weight="bold", color=PALETTE[0])
ax.set_xlabel("Region")
ax.set_ylabel("RTO Rate")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
plt.tight_layout()
plt.savefig(f"{OUT}/07_rto_rate_region_payment.png", dpi=150)
plt.close()

print("All charts saved to", OUT)

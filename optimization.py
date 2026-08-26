"""Courier-allocation optimization (Week 4).

Uses the predictive-modeling insight that distance and pincode tier are the
dominant cost/delay drivers to reformulate courier assignment as a linear
program: minimize total transportation cost by reallocating order volume
across couriers, subject to each courier's handling-capacity limit.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pulp

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = ["#1B2A4A", "#3D5A80", "#E07A3E", "#7A9CC6", "#B0783F"]
OUT = "/home/claude/week4/charts"

df = pd.read_csv("/home/claude/week4/data/logistics_dataset.csv")
df = df.dropna(subset=["order_value", "transportation_cost"]).copy()

TIERS = ["Metro", "Tier-2", "Tier-3"]
COURIERS = sorted(df["courier"].unique())

# Average cost per (tier, courier) combination, observed in the current (effectively
# random) assignment -- this is the cost matrix the optimizer will route against.
cost_matrix = df.pivot_table(index="pincode_tier", columns="courier",
                              values="transportation_cost", aggfunc="mean").reindex(TIERS)

# Current demand (order count) per tier
demand = df.groupby("pincode_tier").size().reindex(TIERS)

# Current cost under the existing (~random) assignment
current_total_cost = df["transportation_cost"].sum()

# Current volume handled per courier -> sets a realistic capacity ceiling
# (allow 15% headroom above current load, i.e. couriers can absorb some
# reallocated volume but not an unbounded amount)
current_volume = df.groupby("courier").size().reindex(COURIERS)
capacity = (current_volume * 1.15).round().astype(int)

# ---------- Linear program: minimize total cost ----------
prob = pulp.LpProblem("CourierAllocation", pulp.LpMinimize)
assign = {(t, c): pulp.LpVariable(f"x_{t}_{c}", lowBound=0) for t in TIERS for c in COURIERS}

prob += pulp.lpSum(assign[(t, c)] * cost_matrix.loc[t, c] for t in TIERS for c in COURIERS)

# Demand satisfaction: all orders in each tier must be assigned
for t in TIERS:
    prob += pulp.lpSum(assign[(t, c)] for c in COURIERS) == demand[t]

# Capacity constraint: each courier cannot exceed its handling capacity
for c in COURIERS:
    prob += pulp.lpSum(assign[(t, c)] for t in TIERS) <= capacity[c]

prob.solve(pulp.PULP_CBC_CMD(msg=0))

optimized_total_cost = pulp.value(prob.objective)
savings = current_total_cost - optimized_total_cost
savings_pct = savings / current_total_cost * 100

print(f"Current total cost:   Rs {current_total_cost:,.0f}")
print(f"Optimized total cost: Rs {optimized_total_cost:,.0f}")
print(f"Savings: Rs {savings:,.0f} ({savings_pct:.1f}%)")

# Extract the optimized allocation for reporting
alloc_rows = []
for t in TIERS:
    for c in COURIERS:
        v = assign[(t, c)].value()
        if v and v > 0.5:
            alloc_rows.append({"Tier": t, "Courier": c, "Orders Assigned": round(v)})
alloc_df = pd.DataFrame(alloc_rows)
alloc_df.to_csv(f"{OUT}/optimized_allocation.csv", index=False)
print(alloc_df)

pd.DataFrame([{
    "Current Total Cost": round(current_total_cost, 0),
    "Optimized Total Cost": round(optimized_total_cost, 0),
    "Savings (Rs)": round(savings, 0),
    "Savings (%)": round(savings_pct, 1),
}]).to_csv(f"{OUT}/optimization_summary.csv", index=False)

# ---------- Chart: current vs optimized total cost ----------
fig, ax = plt.subplots(figsize=(5.5, 4.5))
bars = ax.bar(["Current Allocation", "Optimized Allocation"],
              [current_total_cost, optimized_total_cost],
              color=[PALETTE[2], PALETTE[0]], width=0.5)
for b in bars:
    ax.annotate(f"\u20b9{b.get_height():,.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=11)
ax.set_ylabel("Total Transportation Cost (\u20b9)")
ax.set_title(f"Cost Impact of Optimized Courier Allocation\n({savings_pct:.1f}% reduction)",
             fontsize=12, weight="bold", color=PALETTE[0])
plt.tight_layout()
plt.savefig(f"{OUT}/05_optimization_savings.png", dpi=150)
plt.close()

# ---------- Chart: allocation shift heatmap (optimized volume share by tier x courier) ----------
alloc_pivot = alloc_df.pivot_table(index="Tier", columns="Courier", values="Orders Assigned", fill_value=0).reindex(TIERS)
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.heatmap(alloc_pivot, annot=True, fmt=".0f", cmap="Blues", ax=ax, cbar_kws={"label": "Orders Assigned"})
ax.set_title("Optimized Order Allocation by Tier and Courier", fontsize=12, weight="bold", color=PALETTE[0])
plt.tight_layout()
plt.savefig(f"{OUT}/06_optimized_allocation_heatmap.png", dpi=150)
plt.close()

print("Optimization charts saved to", OUT)

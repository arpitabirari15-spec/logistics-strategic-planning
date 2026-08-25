# Logistics Strategic Planning & Data Exploration

Week 1 internship deliverable: strategic plan for a data-science-driven logistics
analysis project, using a D2C e-commerce brand's last-mile delivery network
(regional warehouses -> metro/Tier-2/Tier-3 customers via a multi-courier panel)
as the scenario.

## Contents
- `docs/Logistics_Strategic_Planning_Report.docx` — full strategic planning report
  (scenario, KPIs, literature review, roadmap, and code walkthroughs).
- `src/` — runnable versions of the Python snippets referenced in the report:
  - `data_cleaning.py` — merges orders/courier-trips/RTO logs into one analytical table
  - `eda.py` — delay distribution by courier and weekly regional demand seasonality
  - `demand_forecasting.py` — GradientBoostingRegressor region-level demand forecast
  - `zone_clustering.py` — K-Means delivery-zone (pincode) clustering
  - `rto_risk_classification.py` — GradientBoostingClassifier RTO/delivery-failure risk model
  - `route_optimization.py` — OR-Tools CVRP solver skeleton for courier allocation

## KPIs tracked
On-Time Delivery Rate, First-Attempt Delivery Success (FADS), Average Delivery
Cycle Time, Cost per Delivery, Return-to-Origin (RTO) Rate.

## Status
Planning-stage deliverable (Week 1). Scripts are structured against expected
column schemas; real/proxy datasets (e.g. the Kaggle Delhivery dataset) to be
wired in during Week 2.

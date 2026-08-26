# Logistics Data Science Internship — Weeks 1–4

Internship project on a D2C e-commerce brand's last-mile delivery network
(regional warehouses -> metro/Tier-2/Tier-3 customers via a multi-courier panel).

## Week 1 — Strategic Planning and Data Exploration
`week1/docs/Logistics_Strategic_Planning_Report.docx` — scenario definition, KPIs,
literature review, seven-stage roadmap, and Python code illustrations.

`week1/src/` — data_cleaning.py, eda.py, demand_forecasting.py, zone_clustering.py,
rto_risk_classification.py, route_optimization.py

## Week 2 — Data Collection, Cleaning, and Preprocessing
`week2/docs/Logistics_Data_Cleaning_Preprocessing_Report.docx` — data collection
simulation using the public Kaggle Delhivery Logistics Dataset, identified data
quality issues, and a full cleaning/preprocessing methodology.

`week2/src/preprocessing_pipeline.py` — end-to-end runnable pipeline: format
standardization, deduplication, missing-value handling, IQR outlier treatment,
categorical encoding, scaling, and SMOTE class-imbalance correction.

## Week 3 — Advanced Data Analysis and Visualization
`week3/docs/Logistics_EDA_Visualization_Report.docx` — EDA and 7 visualizations
(distribution, boxplot, violin, scatter, heatmap, trend line, grouped bar) on a
simulated 3,000-order dataset, with insights and recommendations.

`week3/src/generate_dataset.py` — synthetic dataset generator (distance, cost,
cycle time, RTO, with realistic correlations and a June demand-spike event).
`week3/src/eda_visualization.py` — full EDA + chart generation script.

## Week 4 — Predictive Modeling and Optimization
`week4/docs/Logistics_Predictive_Modeling_Optimization_Report.docx` — cycle-time
forecasting (Linear Regression vs. Random Forest vs. Gradient Boosting, with
hyperparameter tuning and cross-validation) plus a linear-programming courier-
allocation optimization built on the model's feature-importance findings.

`week4/src/predictive_modeling.py` — model training, tuning, and evaluation.
`week4/src/optimization.py` — PuLP-based courier allocation cost optimization
(10.2% simulated cost reduction).

## KPIs tracked (from Week 1)
On-Time Delivery Rate, First-Attempt Delivery Success (FADS), Average Delivery
Cycle Time, Cost per Delivery, Return-to-Origin (RTO) Rate.

## Status
Weeks 1–4 complete. All scripts run against simulated data with realistic,
documented distributions and correlations; a real or public proxy dataset
(e.g. the Kaggle Delhivery dataset) can be substituted using the same schemas.

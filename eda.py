"""Exploratory analysis: delay distribution by courier and regional demand seasonality."""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_delay_by_courier(df: pd.DataFrame, out_path: str = "delay_by_courier.png") -> None:
    sns.boxplot(data=df, x="courier", y="cycle_time_days")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def weekly_demand_by_region(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby([pd.Grouper(key="order_date", freq="W"), "region"])["order_id"]
        .count()
        .reset_index(name="order_count")
    )


if __name__ == "__main__":
    df = pd.read_csv("analytical_table.csv", parse_dates=["order_date"])
    plot_delay_by_courier(df)
    demand = weekly_demand_by_region(df)
    demand.to_csv("weekly_demand.csv", index=False)

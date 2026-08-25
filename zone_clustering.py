"""Cluster delivery destinations into geographically coherent zones."""
import pandas as pd
from sklearn.cluster import KMeans


def cluster_delivery_zones(df: pd.DataFrame, n_clusters: int = 6) -> pd.DataFrame:
    coords = df[["dest_lat", "dest_lon"]].drop_duplicates()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    coords["zone"] = kmeans.fit_predict(coords[["dest_lat", "dest_lon"]])
    return coords


if __name__ == "__main__":
    df = pd.read_csv("analytical_table.csv")
    zones = cluster_delivery_zones(df)
    zones.to_csv("delivery_zones.csv", index=False)

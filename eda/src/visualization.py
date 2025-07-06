from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import col
import os

def create_visualizations(spark: SparkSession, df):
    """
    Creates and saves 4 plots showing the 2D embeddings.
    """
    # Collect data to Pandas DataFrame for plotting
    # Be cautious with very large datasets; consider sampling if memory is an issue
    plot_df = df.select(
        col("pca_features_2d")[0].alias("x"),
        col("pca_features_2d")[1].alias("y"),
        col("prediction").alias("cluster"),
        col("is_outlier").alias("outlier"),
        col("section_title").alias("section"),
        col("company_name").alias("company")
    ).toPandas()

    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)

    # Plot 1: Colored by cluster assignment
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="x", y="y", hue="cluster", data=plot_df, palette="viridis", s=50, alpha=0.7)
    plt.title("2D Embeddings Colored by Cluster Assignment")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.savefig("plots/clusters.jpeg")
    plt.close()

    # Plot 2: Colored by outlier flag (binary: outlier/normal)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="x", y="y", hue="outlier", data=plot_df, palette="coolwarm", s=50, alpha=0.7)
    plt.title("2D Embeddings Colored by Outlier Flag")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Outlier")
    plt.grid(True)
    plt.savefig("plots/outliers.jpeg")
    plt.close()

    # Plot 3: Colored by section number (assuming section_name can be mapped to a number or is categorical)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="x", y="y", hue="section", data=plot_df, palette="tab10", s=50, alpha=0.7)
    plt.title("2D Embeddings Colored by Section Name")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Section", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/sections.jpeg")
    plt.close()

    # Plot 4: Combined view with company labels
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x="x", y="y", hue="company", style="cluster", data=plot_df, palette="tab20", s=70, alpha=0.8)
    plt.title("2D Embeddings with Company Labels and Cluster Assignment")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Company / Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/combined.jpeg")
    plt.close()

    print("Visualizations saved to the 'plots/' directory.")

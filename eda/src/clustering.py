from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import FloatType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf, col
from scipy.spatial.distance import euclidean
import numpy as np

def process_and_cluster(spark: SparkSession, df):
    """
    Applies StandardScaler, PCA, KMeans clustering, and outlier detection.
    """
    # Convert embedding list to VectorUDT for MLlib
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
    df_with_vectors = df.withColumn("embedding_vector", list_to_vector_udf(col("embedding")))

    # 1. StandardScaler for embedding normalization
    scaler = StandardScaler(inputCol="embedding_vector", outputCol="scaled_embedding",
                            withStd=True, withMean=False)
    scaler_model = scaler.fit(df_with_vectors)
    scaled_df = scaler_model.transform(df_with_vectors)
    
    # Cache scaled data
    scaled_df.cache()
    scaled_df.count()

    # 2. PCA for initial dimensionality reduction (e.g., to 50 components)
    pca_50 = PCA(k=50, inputCol="scaled_embedding", outputCol="pca_features_50")
    pca_50_model = pca_50.fit(scaled_df)
    pca_50_df = pca_50_model.transform(scaled_df)
    
    # Cache PCA 50 data
    pca_50_df.cache()
    pca_50_df.count()

    # 3. Additional dimensionality reduction to 2D (using PCA)
    # FUTURE IMPROVEMENT: PCA is used for its speed and availability in PySpark. However, for visualization,
    # UMAP or t-SNE often produce more meaningful and well-separated clusters by better preserving
    # the local structure of the data. This would require converting a sample of the data to a Pandas
    # DataFrame and using a library like 'umap-learn' or 'scikit-learn'.
    pca_2 = PCA(k=2, inputCol="pca_features_50", outputCol="pca_features_2d")
    pca_2_model = pca_2.fit(pca_50_df)
    pca_2d_df = pca_2_model.transform(pca_50_df)
    
    # Cache PCA 2D data
    pca_2d_df.cache()
    pca_2d_df.count()

    # 4. KMeans clustering
    # FUTURE IMPROVEMENT: The number of clusters (k=7) is hardcoded. This is not ideal as the optimal
    # number of clusters can vary. A better approach is to use the Elbow Method or Silhouette Score
    # to programmatically determine the optimal 'k' by testing a range of values.
    kmeans = KMeans().setK(7).setSeed(1)
    kmeans_model = kmeans.fit(pca_2d_df)
    clustered_df = kmeans_model.transform(pca_2d_df)
    
    # Cache clustered data
    clustered_df.cache()
    clustered_df.count()

    # 5. Outlier detection using distance from cluster centers
    # Get cluster centers
    centers = kmeans_model.clusterCenters()

    @udf(FloatType())
    def get_distance_to_center_udf(features, cluster_id):
        center = centers[cluster_id]
        return float(euclidean(features.toArray(), center))

    clustered_df = clustered_df.withColumn("distance_to_center", 
                                           get_distance_to_center_udf(col("pca_features_2d"), col("prediction")))

    # Calculate mean and std dev of distances for each cluster in a distributed way
    agg_df = clustered_df.groupBy("prediction").agg(
        F.mean("distance_to_center").alias("mean_dist"),
        F.stddev("distance_to_center").alias("std_dist")
    )

    # Join the aggregated stats back to the main df
    clustered_df = clustered_df.join(agg_df, "prediction")

    # Determine outliers in a distributed way
    final_df = clustered_df.withColumn("is_outlier", 
                                      col("distance_to_center") > (col("mean_dist") + 2 * col("std_dist")))
    
    # Cache final data
    final_df.cache()
    final_df.count()

    return final_df

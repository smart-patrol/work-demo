import os
from pyspark.sql import SparkSession
from src.data_loader import load_and_filter_data
from src.chunking import chunk_sections
from src.embeddings import generate_embeddings
from src.clustering import process_and_cluster
from src.visualization import create_visualizations

# Set the python executable for pyspark
os.environ['PYSPARK_PYTHON'] = 'python'

# Initialize Spark
spark = SparkSession.builder     .appName("SEC_Filings_Analysis")     .config("spark.driver.memory", "8g")     .config("spark.executor.memory", "8g")     .config("spark.driver.host", "127.0.0.1")     .config("spark.network.timeout", "800s")     .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")     .config("spark.python.worker.faulthandler.enabled", "true")     .getOrCreate()

# Main pipeline
def main():
    print("Step 1: Loading and filtering data...")
    filtered_df = load_and_filter_data(spark)
    print(f"Filtered data count: {filtered_df.count()}")

    print("Step 2: Chunking documents...")
    chunked_df = chunk_sections(spark, filtered_df)
    print(f"Chunked data count: {chunked_df.count()}")

    print("Step 3: Generating embeddings...")
    embeddings_df = generate_embeddings(spark, chunked_df)
    print(f"Embeddings data count: {embeddings_df.count()}")

    print("Step 4: Processing and clustering...")
    clustered_df = process_and_cluster(spark, embeddings_df)
    print(f"Clustered data count: {clustered_df.count()}")

    print("Step 5: Creating visualizations...")
    create_visualizations(spark, clustered_df)
    print("Analysis complete. Visualizations saved.")

    spark.stop()

if __name__ == "__main__":
    main()

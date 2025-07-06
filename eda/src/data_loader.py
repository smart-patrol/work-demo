from pyspark.sql import SparkSession
from datasets import load_dataset
from pyspark.sql.functions import col

def load_and_filter_data(spark: SparkSession):
    """
    Loads the IEDGAR dataset from HuggingFace for a specific year, 
    filters it, and returns a PySpark DataFrame.
    """
    # Load the 'validation' split of the IEDGAR dataset for the year 2020 in streaming mode
    dataset = load_dataset("eloukas/edgar-corpus", "year_2020", split="validation", streaming=True)

    # Convert the dataset to a Spark DataFrame
    spark_df = spark.createDataFrame(dataset)

    # Get 5 unique companies
    # FUTURE IMPROVEMENT: The selection of 5 companies is random (`.limit(5)`), which means the
    # analysis will run on a different set of companies each time. For reproducible and
    # consistent analysis, it would be better to either:
    # 1.  Hardcode a specific list of company CIKs to filter on.
    # 2.  Sort the distinct companies by CIK before taking the top 5, ensuring the same
    #     companies are selected in every run.
    unique_companies = spark_df.select("cik").distinct().limit(5)
    
    # Filter the DataFrame to keep only the selected companies
    filtered_df = spark_df.join(unique_companies, "cik")
    
    # Cache the filtered data to avoid recomputation in subsequent actions
    filtered_df.cache()
    
    # Trigger an action to materialize the cache
    count = filtered_df.count()
    print(f"Successfully loaded and cached {count} records for 5 companies in 2020.")

    return filtered_df

if __name__ == '__main__':
    # Create a SparkSession
    spark = SparkSession.builder \
        .appName("IEDGAR 2020 Data Loader") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # Load and filter the data
    filtered_spark_df = load_and_filter_data(spark)

    # Show some of the resulting data
    print("\nSample of the filtered data:")
    filtered_spark_df.show(5)
    
    print("\nSchema of the final DataFrame:")
    filtered_spark_df.printSchema()

    # Stop the SparkSession
    spark.stop()

from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType
import pandas as pd
from sentence_transformers import SentenceTransformer

# Global variable to hold the model, loaded once per worker
_model = None

@pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)
def generate_embedding_udf(texts: pd.Series) -> pd.Series:
    """
    Pandas UDF to generate embeddings for a batch of texts using SentenceTransformer.
    The model is loaded once per worker.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Handle None values in the input series
    embeddings = texts.apply(lambda x: _model.encode(x).tolist() if x is not None else [])
    return embeddings

def generate_embeddings(spark: SparkSession, df):
    """
    Generates embeddings for each chunk in the DataFrame and stores them as dense vectors.
    """
    # Apply the Pandas UDF to create a new column 'embedding'
    embeddings_df = df.withColumn("embedding", generate_embedding_udf(df["chunk_text"]))
    
    # Cache the embeddings DataFrame
    embeddings_df.cache()
    embeddings_df.count() # Trigger caching

    return embeddings_df

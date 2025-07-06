from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, explode
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, IntegerType
import re

def chunk_document(text, chunk_size=500):
    """
    Splits a given text into chunks of approximately chunk_size words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def extract_sections(text):
    sections = {}
    # Example pattern for section headers (case-insensitive)
    # This pattern looks for "ITEM" followed by a number and optional letter, then a period,
    # and then captures the section title until a newline.
    # It also handles "PART" sections.
    pattern = r'(?im)^(ITEM\s+\d+[A-Z]?\.?|PART\s+[IVXLCDM]+)\s*\n(.*?)(?=(?:^ITEM\s+\d+[A-Z]?\.?|^PART\s+[IVXLCDM]+)\s*$)'
    
    matches = re.findall(pattern, text, re.DOTALL)
    
    for i, match in enumerate(matches):
        section_title = match[0].strip()
        section_content = match[1].strip()
        # Normalize title: remove trailing period and collapse whitespace
        section_title = re.sub(r'\s+', ' ', section_title).strip('.').upper()
        sections[section_title] = section_content
        
    return sections

def chunk_sections(spark: SparkSession, df):
    """
    Splits documents by section and then further splits long sections into ~500 word chunks.
    Preserves metadata (company, section number, chunk index).
    """
    # Define schema for the exploded DataFrame
    chunk_schema = ArrayType(ArrayType(StringType()))

    @udf(chunk_schema)
    def process_document_udf(text, cik, company_name, filing_type, year):
        all_chunks = []
        sections = extract_sections(text) # Extract sections from the raw text
        for section_title, section_text in sections.items():
            # Split sections into smaller chunks
            chunks = chunk_document(section_text)
            for i, chunk_text in enumerate(chunks):
                all_chunks.append([
                    cik,
                    company_name,
                    filing_type,
                    str(year),
                    section_title,
                    chunk_text,
                    str(i)
                ])
        return all_chunks

    # Apply the UDF to process each document and explode the results
    # Now using the 'text' column as input
    chunked_df = df.withColumn("processed_chunks", process_document_udf(
        df["text"], df["cik"], df["company_name"], df["filing_type"], df["year"]
    )).select(explode("processed_chunks").alias("chunk"))

    # Select and flatten the chunk data
    final_chunked_df = chunked_df.select(
        col("chunk")[0].alias("cik"),
        col("chunk")[1].alias("company_name"),
        col("chunk")[2].alias("filing_type"),
        col("chunk")[3].alias("year"),
        col("chunk")[4].alias("section_title"),
        col("chunk")[5].alias("chunk_text"),
        col("chunk")[6].alias("chunk_index")
    )
    
    # Cache the chunked data
    final_chunked_df.cache()
    final_chunked_df.count() # Trigger caching

    return final_chunked_df

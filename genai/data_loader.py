# Memory-Efficient SEC Data Loader
# Save as: load_filtered_sec_data.py

import pandas as pd
from datasets import load_dataset

def load_filtered_edgar_data():
    """Load and filter EDGAR data efficiently before converting to pandas"""
    print("Loading EDGAR dataset from HuggingFace...")
    
    # Load the dataset
    dataset = load_dataset("eloukas/edgar-corpus", "year_2020")
    
    print(f"Original dataset size: {len(dataset['train'])} rows")
    
    # Define our filters
    COMPANY_CIK = 1411906  # The biopharmaceutical company
    TARGET_YEARS = [2018, 2019, 2020]
    
    print(f"Filtering for CIK {COMPANY_CIK} and years {TARGET_YEARS}...")
    
    # Filter in HuggingFace format (memory efficient!)
    filtered_dataset = dataset['train']
    
    print(f"Filtered dataset size: {len(filtered_dataset)} rows")
    
    if len(filtered_dataset) == 0:
        print("No data found for the specified filters!")
        return None
    
    # Now convert to pandas (much smaller dataset)
    df = filtered_dataset.to_pandas()
    
    print(f"Successfully loaded filtered data!")
    print(f"Shape: {df.shape}")
    print(f"Years found: {sorted(df['year'].unique())}")
    print(f"CIK: {df['cik'].unique()}")
    
    return df

def explore_filtered_data(df):
    """Show what we got"""
    if df is None:
        return
    
    print("\n" + "=" * 50)
    print("FILTERED DATA SUMMARY")
    print("=" * 50)
    
    for idx, row in df.iterrows():
        print(f"\nFiling {idx + 1}:")
        print(f"   Filename: {row['filename']}")
        print(f"   Year: {row['year']}")
        print(f"   CIK: {row['cik']}")
        
        # Check which sections have data
        sections_with_data = []
        for col in df.columns:
            if col.startswith('section_') and pd.notna(row[col]) and str(row[col]).strip():
                section_length = len(str(row[col]))
                sections_with_data.append(f"{col} ({section_length:,} chars)")
        
        # print(f"   Sections with data: {len(sections_with_data)}")
        # for section in sections_with_data[:5]:  # Show first 5
        #     print(f"     • {section}")
        # if len(sections_with_data) > 5:
        #     print(f"     • ... and {len(sections_with_data) - 5} more")
        
        # # Sample from section_1 (Business description)
        # if pd.notna(row['section_1']) and str(row['section_1']).strip():
        #     sample = str(row['section_1'])[:300] + "..." if len(str(row['section_1'])) > 300 else str(row['section_1'])
        #     print(f"   Business Section Preview:")
        #     print(f"      {sample.encode('ascii', 'ignore').decode('ascii')}")

# Main execution
if __name__ == "__main__":
    print("Loading Filtered SEC EDGAR Data")
    print("=" * 50)
    
    # Load filtered data (memory efficient!)
    df = load_filtered_edgar_data()
    
    if df is not None:
        # Explore what we got
        explore_filtered_data(df)
        
        # Save to CSV
        output_file = "edgar_cik_1411906_filtered.csv"
        df.to_csv(output_file, index=False)
        print(f"\nFiltered data saved to: {output_file}")
        
        print(f"\nReady for analysis!")
        print(f"Use 'df' variable to access your filtered data")
        print(f"Available columns: {list(df.columns)}")
    else:
        print("No data found with current filters")
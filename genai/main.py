import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class DocumentProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the document processor with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None
        self.metadata = []
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load CSV data from file."""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} documents from {filepath}")
        return df
    
    def create_chunks(self, df: pd.DataFrame, chunk_size: int = 500) -> List[Dict]:
        """
        Convert documents to chunks with metadata.
        We'll focus on 5 attributes: section_1, section_1A, section_7, section_8, section_10
        """
        chunks = []
        selected_sections = ['section_1', 'section_1A', 'section_7', 'section_8', 'section_10']
        
        for idx, row in df.iterrows():
            year = row['year']
            cik = row['cik']
            filename = row['filename']
            
            for section in selected_sections:
                if pd.notna(row[section]) and row[section].strip():
                    text = str(row[section])
                    
                    # Split text into chunks
                    words = text.split()
                    for i in range(0, len(words), chunk_size):
                        chunk_text = ' '.join(words[i:i+chunk_size])
                        
                        chunk_data = {
                            'text': chunk_text,
                            'year': year,
                            'cik': cik,
                            'filename': filename,
                            'section': section,
                            'chunk_id': f"{filename}_{section}_{i//chunk_size}"
                        }
                        chunks.append(chunk_data)
        
        self.chunks = chunks
        self.metadata = [{k: v for k, v in chunk.items() if k != 'text'} for chunk in chunks]
        print(f"Created {len(chunks)} chunks from documents")
        return chunks
    
    def create_embeddings(self) -> np.ndarray:
        """Convert chunks to embeddings using sentence transformer."""
        if not self.chunks:
            raise ValueError("No chunks available. Please create chunks first.")
        
        texts = [chunk['text'] for chunk in self.chunks]
        print(f"Creating embeddings for {len(texts)} chunks...")
        
        # Create embeddings in batches to manage memory
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_tensor=True)
            all_embeddings.append(batch_embeddings)
        
        self.embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
        print(f"Created embeddings with shape: {self.embeddings.shape}")
        return self.embeddings
    
    def search_chunks(self, query: str, year: int, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks from a specific year."""
        if self.embeddings is None:
            raise ValueError("No embeddings available. Please create embeddings first.")
        
        # Encode the query
        query_embedding = self.model.encode([query], convert_to_tensor=True).cpu().numpy()
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Filter by year and get top results
        year_indices = [i for i, meta in enumerate(self.metadata) if meta['year'] == year]
        
        if not year_indices:
            print(f"No chunks found for year {year}")
            return []
        
        year_similarities = [(idx, similarities[idx]) for idx in year_indices]
        year_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k results
        results = []
        for idx, score in year_similarities[:top_k]:
            result = {
                'chunk': self.chunks[idx]['text'][:200] + "...",  # Preview
                'full_text': self.chunks[idx]['text'],
                'metadata': self.metadata[idx],
                'similarity_score': score
            }
            results.append(result)
        
        return results

def create_validation_dataset(df: pd.DataFrame, year: int) -> List[Dict]:
    """Create a validation dataset with 5 true values from chunks for a specific year."""
    validation_data = []
    year_data = df[df['year'] == year]
    
    if year_data.empty:
        print(f"No data found for year {year}")
        return validation_data
    
    # Extract specific facts from the data
    row = year_data.iloc[0]
    
    # Define validation queries and expected answers
    validation_items = [
        {
            'query': 'When was the company organized and in which state?',
            'expected_section': 'section_1',
            'expected_answer': 'organized under the laws of the State of Vermont in 1982'
        },
        {
            'query': 'What banking services does the company provide?',
            'expected_section': 'section_1',
            'expected_answer': 'Business Banking, Commercial Real Estate Lending, Residential Real Estate Lending'
        },
        {
            'query': 'When did the company acquire LyndonBank?',
            'expected_section': 'section_1',
            'expected_answer': 'December 31, 2007'
        },
        {
            'query': 'What is the original name of the bank?',
            'expected_section': 'section_1',
            'expected_answer': 'Peoples Bank'
        },
        {
            'query': 'How many branch offices does the company maintain?',
            'expected_section': 'section_1',
            'expected_answer': 'eleven branch offices'
        }
    ]
    
    for item in validation_items:
        if pd.notna(row[item['expected_section']]):
            item['year'] = year
            item['cik'] = row['cik']
            validation_data.append(item)
    
    return validation_data

def demonstrate_retrieval(processor: DocumentProcessor, validation_data: List[Dict], year: int):
    """Demonstrate that the LLM can retrieve correct chunks for the correct year."""
    print(f"\n{'='*80}")
    print(f"DEMONSTRATING RETRIEVAL FOR YEAR {year}")
    print(f"{'='*80}\n")
    
    for i, val_item in enumerate(validation_data, 1):
        print(f"\nValidation Item {i}:")
        print(f"Query: {val_item['query']}")
        print(f"Expected Answer: {val_item['expected_answer']}")
        print(f"\nRetrieving chunks...")
        
        # Search for relevant chunks
        results = processor.search_chunks(val_item['query'], year, top_k=3)
        
        if results:
            print(f"\nTop retrieved chunk (similarity: {results[0]['similarity_score']:.4f}):")
            print(f"Section: {results[0]['metadata']['section']}")
            print(f"Year: {results[0]['metadata']['year']}")
            print(f"Preview: {results[0]['chunk']}")
            
            # Check if expected answer is in the retrieved chunk
            if val_item['expected_answer'].lower() in results[0]['full_text'].lower():
                print(f"VALIDATION PASSED: Expected answer found in retrieved chunk!")
            else:
                print(f"VALIDATION FAILED: Expected answer not found in top chunk")
                # Check other chunks
                for j, result in enumerate(results[1:], 2):
                    if val_item['expected_answer'].lower() in result['full_text'].lower():
                        print(f"  Note: Expected answer found in chunk #{j}")
                        break
        else:
            print("No chunks retrieved!")
        
        print("-" * 80)

def main():
    # Initialize processor
    processor = DocumentProcessor()
    
    # Load data
    df = processor.load_data('edgar_cik_1411906_filtered.csv')
    
    # Choose a specific year for demonstration
    target_year = 2020
    
    # Step 1: Convert documents to chunks
    chunks = processor.create_chunks(df)
    
    # Step 2: Convert chunks to embeddings
    embeddings = processor.create_embeddings()
    
    # Step 3: Create validation dataset
    validation_data = create_validation_dataset(df, target_year)
    print(f"\nCreated {len(validation_data)} validation items for year {target_year}")
    
    # Step 4: Demonstrate retrieval
    demonstrate_retrieval(processor, validation_data, target_year)
    
    # Additional demonstration: Custom query
    print(f"\n{'='*80}")
    print("CUSTOM QUERY DEMONSTRATION")
    print(f"{'='*80}\n")
    
    custom_query = "What types of real estate lending does the company offer?"
    print(f"Custom Query: {custom_query}")
    results = processor.search_chunks(custom_query, target_year, top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (similarity: {result['similarity_score']:.4f}):")
        print(f"Section: {result['metadata']['section']}")
        print(f"Preview: {result['chunk']}")

if __name__ == "__main__":
    main()
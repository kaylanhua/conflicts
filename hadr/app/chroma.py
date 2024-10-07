import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

# Initialize the Chroma client
client = chromadb.Client()

# Use OpenAI's embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)

def create_or_get_collection(country_name: str):
    """Create or get a collection for a specific country."""
    collection_name = f"hadr-{country_name}"
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=openai_ef
    )

def insert_embedding(collection, year: int, month: int, summary: str, death_count: int):
    """Insert vector embedding for the monthly summary into the collection."""
    id = f"{year}_{month:02d}"
    collection.add(
        documents=[summary],
        metadatas=[{"year": year, "month": month, "death_count": death_count}],
        ids=[id]
    )
    print(f"Inserted embedding for {id}")

def get_similar_months(collection, current_year: int, current_month: int, current_summary: str, top_k: int = 3) -> List[Dict]:
    """Find similar past months based on the current month's summary."""
    results = collection.query(
        query_texts=[current_summary],
        n_results=top_k*2,
        where={"$or": [
            {"year": {"$lt": current_year}},
            {"$and": [
                {"year": current_year},
                {"month": {"$lt": current_month}}
            ]}
        ]}
    )
    
    similar_months = []
    for i, id in enumerate(results['ids'][0]):
        year, month = map(int, id.split('_'))
        similar_months.append({
            'id': id,
            'year': year,
            'month': month,
            'distance': results['distances'][0][i],
            'metadata': results['metadatas'][0][i]
        })
    
    return similar_months[:top_k]

# Example usage
if __name__ == "__main__":
    country_name = "example_country"
    collection = create_or_get_collection(country_name)
    
    # Example insertion
    insert_embedding(collection, 2023, 1, "This is a sample summary for January 2023.", 100)
    
    # Example similarity search
    similar = get_similar_months(collection, 2023, 2, "This is a query summary for February 2023.")
    print("Similar months:", similar)
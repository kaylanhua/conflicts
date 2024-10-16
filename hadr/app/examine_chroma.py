import chromadb
from chromadb.config import Settings
import pandas as pd

def examine_chroma_database(collection_name="hadr-all", persist_directory='vector_db'):
    chroma_client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_directory
    ))
    collection = chroma_client.get_collection(collection_name)

    # Get all items from the collection
    results = collection.get(include=['metadatas', 'documents', 'embeddings'])

    # Create a list to store all metadata
    all_metadata = []

    # Iterate through the results and extract metadata
    for i, metadata in enumerate(results['metadatas']):
        metadata['document'] = results['documents'][i]
        all_metadata.append(metadata)

    # Create a DataFrame from the metadata
    df = pd.DataFrame(all_metadata)

    return df

if __name__ == "__main__":
    # Example usage
    collection_name = "hadr-all"
    persist_directory = 'vector_db'
    df = examine_chroma_database(collection_name, persist_directory)

    print(df.head())
    print(df.info())

    # Save the DataFrame to a CSV file (optional)
    df.to_csv("chroma_database_entries.csv", index=False)
    print("DataFrame saved to chroma_database_entries.csv")


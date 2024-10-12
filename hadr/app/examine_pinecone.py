from pinecone import Pinecone, ServerlessSpec
import os 
from dotenv import load_dotenv

load_dotenv()


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY")) # , environment=os.getenv("PINECONE_ENVIRONMENT")

COUNTRY_NAME = "myanmar"

index_name = f"hadr-1536-{COUNTRY_NAME}" # myanmar
if index_name not in pc.list_indexes().names():
    print(f"Index {index_name} not found")

index = pc.Index(index_name)

# Get the total number of vectors in the index
index_stats = index.describe_index_stats()
total_vector_count = index_stats.total_vector_count
print(f"Total number of entries in the index: {total_vector_count}")

# Query the index to get all vector IDs
query_result = index.query(vector=[0]*1536, top_k=total_vector_count)
vector_ids = [match.id for match in query_result['matches']]

# Fetch all vectors using the retrieved IDs
all_vectors = index.fetch(ids=vector_ids)

# Extract dates from vector IDs
dates = [tuple(map(int, vector_id.split('_'))) for vector_id in all_vectors['vectors'].keys()]

# Sort dates
sorted_dates = sorted(dates)

if sorted_dates:
    print(f"Dates added to the index:")
    for year, month in sorted_dates:
        print(f"  - {year}-{month:02d}")
else:
    print("No entries found in the index.")
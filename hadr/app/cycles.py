import json
import pandas as pd
import numpy as np
from typing import List, Dict
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime, timedelta
from pulling_gdelt import get_gdelt_data, create_dataset
import os
from dotenv import load_dotenv
load_dotenv()

DATA_SOURCE = '../../data/views/sudan.csv'

# Initialize OpenAI and Pinecone clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY")) # , environment=os.getenv("PINECONE_ENVIRONMENT")

index_name = "hadr-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=2,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 

index = pc.Index(index_name)


def prepare_monthly_data(year: int, month: int, queries: List[str]):
    """
    Prepare and save monthly news data using GDELT API.
    """
    start_date = datetime(year, month, 1)
    end_date = start_date + timedelta(days=32)
    end_date = end_date.replace(day=1) - timedelta(days=1)
    
    urls, _ = get_gdelt_data(queries, start_date, end_date, max_records=50)
    news_data = create_dataset(urls)
    
    # Combine all article contents into a single string
    combined_news = "\n\n".join([f"Title: {article['title']}\n{article['body']}" for article in news_data])
    
    # Save news data to a text file
    with open(f'data/news_{year}_{month:02d}.txt', 'w', encoding='utf-8') as f:
        f.write(combined_news)
    
    print(f"Saved news data for {year}-{month:02d} to data/news_{year}_{month:02d}.txt")
    return combined_news

def summarize_monthly_news(year: int, month: int) -> str: 
    # TODO is this the best method
    """
    Summarize the monthly news data using OpenAI's GPT model.
    """
    with open(f'data/news_{year}_{month:02d}.txt', 'r') as f:
        news_text = f.read()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": "Summarize the key violent incidents, their locations, and estimated casualties from the following news articles. Limit your summary to about 500 words."},
            {"role": "user", "content": news_text}
        ]
    )
    
    summary = response.choices[0].message.content
    
    # Save summary to JSON file
    with open(f'data/summary_{year}_{month:02d}.json', 'w') as f:
        json.dump({'year': year, 'month': month, 'summary': summary}, f)
    
    return summary

def create_vector_embedding(year: int, month: int, summary: str, death_count: int):
    """
    Create and store vector embedding for the monthly summary.
    """
    response = client.embeddings.create(
        model="text-embedding-ada-002", # TODO try other embedding methods
        input=summary
    )
    embedding = response.data[0].embedding
    
    # Insert into Pinecone
    index.upsert([(f"{year}_{month:02d}", embedding, {"death_count": death_count})])

def get_similar_months(current_summary: str, top_k: int = 3) -> List[Dict]:
    """
    Find similar past months based on the current month's summary.
    """
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=current_summary
    )
    query_embedding = response.data[0].embedding
    
    similar_months = index.query(query_embedding, top_k=top_k, include_metadata=True)
    return similar_months['matches']

def load_month_key():
    """
    Load the month_key.csv file and return a dictionary mapping month_id to (year, month).
    """
    month_key_df = pd.read_csv('data/month_key.csv')
    return {row['month_id']: (row['year'], row['month']) for _, row in month_key_df.iterrows()}

def get_historical_death_counts(year: int, month: int, num_months: int = 3) -> List[Dict]:
    """
    Get historical death counts for the specified number of past months.
    """
    df = pd.read_csv(DATA_SOURCE)
    month_key = load_month_key()
    historical_counts = []
    
    # Find the month_id for the given year and month
    current_month_id = next(id for id, (y, m) in month_key.items() if y == year and m == month)
    
    for i in range(num_months):
        past_month_id = current_month_id - i - 1
        if past_month_id in month_key:
            past_year, past_month = month_key[past_month_id]
            past_count = df[df['month_id'] == past_month_id]['ged_sb'].values
            if len(past_count) > 0:
                historical_counts.append({
                    'year': past_year,
                    'month': past_month,
                    'death_count': int(past_count[0])
                })
    
    return historical_counts

def predict_next_month(year: int, month: int) -> int:
    """
    Predict the death count for the next month.
    """
    with open(f'data/summary_{year}_{month:02d}.json', 'r') as f:
        current_data = json.load(f)
    
    similar_months = get_similar_months(current_data['summary'])
    historical_counts = get_historical_death_counts(year, month)
    
    prompt = f"Current month summary: {current_data['summary']}\n\n"
    prompt += "Similar past months:\n"
    for match in similar_months:
        prompt += f"- Month: {match['id']}, Death count: {match['metadata']['death_count']}\n"
    
    prompt += f"\nHistorical death counts for the past 3 months:\n"
    for count in historical_counts:
        prompt += f"- Month: {count['year']}_{count['month']:02d}, Death count: {count['death_count']}\n"
    
    prompt += "\nBased on this information, predict the death count for the next month. Provide only the number."
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI trained to predict death counts in civil conflicts based on news summaries and historical data."},
            {"role": "user", "content": prompt}
        ]
    )
    
    prediction = response.choices[0].message.content
    return int(prediction)

def evaluate_predictions(year: int) -> Dict[str, float]:
    """
    Evaluate predictions for a given year and return error metrics.
    """
    df = pd.read_csv('data/historical_death_counts.csv')
    df_year = df[df['year'] == year]
    
    actual_counts = []
    predicted_counts = []
    
    for _, row in df_year.iterrows():
        month = row['month']
        actual = row['death_count']
        predicted = predict_next_month(year, month-1 if month > 1 else 12)  # Predict based on previous month
        
        actual_counts.append(actual)
        predicted_counts.append(predicted)
    
    mae = np.mean(np.abs(np.array(actual_counts) - np.array(predicted_counts)))
    rmse = np.sqrt(np.mean((np.array(actual_counts) - np.array(predicted_counts))**2))
    
    return {"MAE": mae, "RMSE": rmse}

def run_prediction_cycle(year: int, month: int, queries: List[str]) -> int:
    """
    Run a complete prediction cycle for a given month.
    """
    prepare_monthly_data(year, month, queries)
    summary = summarize_monthly_news(year, month)
    
    # Get the death count from the historical data
    month_key = load_month_key()
    month_id = next(id for id, (y, m) in month_key.items() if y == year and m == month)
    df = pd.read_csv(DATA_SOURCE)
    death_count = df[df['month_id'] == month_id]['ged_sb'].values[0]
    
    create_vector_embedding(year, month, summary, death_count)
    next_month_prediction = predict_next_month(year, month)
    return next_month_prediction

if __name__ == "__main__":
    # Example usage
    current_year = 2023
    current_month = 5
    queries = ["sudan", "rapid support force", "RSF", "janjaweed"]
    
    prediction = run_prediction_cycle(current_year, current_month, queries)
    print(f"Predicted death count for next month: {prediction}")
    
    # Evaluate predictions for the previous year
    evaluation_results = evaluate_predictions(current_year - 1)
    print(f"Evaluation results for {current_year - 1}:")
    print(f"Mean Absolute Error: {evaluation_results['MAE']}")
    print(f"Root Mean Square Error: {evaluation_results['RMSE']}")

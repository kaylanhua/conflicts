import json
import pandas as pd
import numpy as np
from typing import List, Dict
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
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

index_name = "hadr-index-1536"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
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
    """
    Summarize the monthly news data using OpenAI's GPT model.
    If a summary already exists for the given month, return it without regenerating.
    """
    summary_file = f'data/summary_{year}_{month:02d}.json'
    
    # Check if summary file already exists
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            existing_summary = json.load(f)
        return existing_summary['summary']
    
    # If no existing summary, generate a new one
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
    with open(summary_file, 'w') as f:
        json.dump({'year': year, 'month': month, 'summary': summary}, f)
    
    return summary

def create_vector_embedding(year: int, month: int, summary: str, death_count: int):
    """
    Create and store vector embedding for the monthly summary.
    """
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    embedding = embeddings.embed_documents([summary])[0]
    
    # Insert into Pinecone
    index.upsert([(f"{year}_{month:02d}", embedding, {"death_count": int(death_count)})])

def get_similar_months(current_summary: str, top_k: int = 3) -> List[Dict]:
    """
    Find similar past months based on the current month's summary.
    """
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=current_summary
    )
    query_embedding = response.data[0].embedding
    
    similar_months = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    print(f"****** similar months: {similar_months}\033[0m")
    return similar_months['matches']

def load_month_key():
    """
    Load the month_key.csv file and return a dictionary mapping month_id to (year, month).
    """
    month_key_df = pd.read_csv('../../data/views/month_key.csv')
    return {row['month_id']: (row['Year'], row['Month']) for _, row in month_key_df.iterrows()}

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
    
    print(f"****** historial death data: {historical_counts}")
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
    try:
        return int(float(prediction))
    except ValueError:
        print(f"Warning: Unable to convert prediction '{prediction}' to integer. Returning 0.")
        return 0

def prepare_and_insert_month(year: int, month: int, queries: List[str]) -> None:
    month_id = f"{year}_{month:02d}"
    
    if index.fetch([month_id]).vectors:
        print(f"Month {month_id} already exists in the vector database. Skipping.")
        return

    print(f"Preparing and inserting data for {month_id}")
    if not os.path.exists(f'data/news_{year}_{month:02d}.txt'):
        prepare_monthly_data(year, month, queries)
    
    summary = summarize_monthly_news(year, month)
    
    # Get the death count from the historical data
    month_key = load_month_key()
    month_id_num = next(id for id, (y, m) in month_key.items() if y == year and m == month)
    df = pd.read_csv(DATA_SOURCE)
    true_death_count = df[df['month_id'] == month_id_num]['ged_sb'].values[0]
    
    create_vector_embedding(year, month, summary, true_death_count)
    print(f"Month {month_id} has been prepared and inserted into the vector database.")

def evaluate_predictions(year: int, queries: List[str], forecast_months: int = 12) -> Dict[str, float]:
    print(f"Starting evaluation for year {year} with {forecast_months} forecast months")
    df = pd.read_csv(DATA_SOURCE)
    month_key_df = pd.read_csv('../../data/views/month_key.csv')
    
    actual_counts = []
    predicted_counts = []
    
    for month in range(1, forecast_months + 1):
        print(f"\033[92mPROCESSING MONTH {year}-{month:02d}\033[0m")
        month_id = month_key_df[(month_key_df['Year'] == year) & (month_key_df['Month'] == month)]['month_id'].values[0]
        actual = df[df['month_id'] == month_id]['ged_sb'].values[0]
        
        prev_month = 12 if month == 1 else month - 1
        prev_year = year - 1 if month == 1 else year
        
        # Prepare and insert the previous month if it doesn't exist
        prepare_and_insert_month(prev_year, prev_month, queries)
        
        predicted = predict_next_month(prev_year, prev_month)
        
        actual_counts.append(actual)
        predicted_counts.append(predicted)
        print(f"Actual: {actual}, Predicted: {predicted}")
    
    mae = np.mean(np.abs(np.array(actual_counts) - np.array(predicted_counts)))
    rmse = np.sqrt(np.mean((np.array(actual_counts) - np.array(predicted_counts))**2))
    
    print(f"Evaluation complete. MAE: {mae}, RMSE: {rmse}")
    return {"MAE": mae, "RMSE": rmse}

def run_prediction_cycle(year: int, month: int, queries: List[str]) -> int:
    print(f"Running prediction cycle for {year}-{month:02d}")
    if not os.path.exists(f'data/news_{year}_{month:02d}.txt'):
        prepare_monthly_data(year, month, queries)
    
    summary = summarize_monthly_news(year, month)
    
    # Get the death count from the historical data
    month_key = load_month_key()
    month_id = next(id for id, (y, m) in month_key.items() if y == year and m == month)
    df = pd.read_csv(DATA_SOURCE)
    death_count = df[df['month_id'] == month_id]['ged_sb'].values[0]
    print(f"Death count for {year}-{month:02d}: {death_count}")
    
    create_vector_embedding(year, month, summary, death_count)
    next_month_prediction = predict_next_month(year, month)
    
    # Get the true value for the next month
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    next_month_id = next(id for id, (y, m) in month_key.items() if y == next_year and m == next_month)
    true_value = df[df['month_id'] == next_month_id]['ged_sb'].values[0]
    print(f"Prediction for {next_year}-{next_month:02d}: {next_month_prediction}")
    print(f"True value for {next_year}-{next_month:02d}: {true_value}")
    
    return next_month_prediction

if __name__ == "__main__":
    # Example usage
    current_year = 2023
    current_month = 5
    queries = ["sudan", "rapid support force", "RSF", "janjaweed"]
    
    prediction = run_prediction_cycle(current_year, current_month, queries)
    
    print("-------------------EVALUATION---------------------")
    
    # Evaluate predictions for the previous year
    evaluation_results = evaluate_predictions(current_year - 1, queries, forecast_months=1)
    print(f"Evaluation results for {current_year - 1}:")
    print(f"Mean Absolute Error: {evaluation_results['MAE']}")
    print(f"Root Mean Square Error: {evaluation_results['RMSE']}")

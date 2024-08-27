import json
import pandas as pd
import numpy as np
from typing import List, Dict
from openai import OpenAI
import pinecone
from datetime import datetime, timedelta

# Initialize OpenAI and Pinecone clients
client = OpenAI()
pinecone.init(api_key="your-pinecone-api-key", environment="your-pinecone-environment")
index = pinecone.Index("your-index-name")

def prepare_monthly_data(year: int, month: int, news_data: str, death_count: int):
    """
    Prepare and save monthly news data and death count.
    """
    # Save news data to a text file
    with open(f'data/news_{year}_{month:02d}.txt', 'w') as f:
        f.write(news_data)
    
    # Update historical death counts CSV
    df = pd.read_csv('data/historical_death_counts.csv')
    new_row = pd.DataFrame({'year': [year], 'month': [month], 'death_count': [death_count]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv('data/historical_death_counts.csv', index=False)

def summarize_monthly_news(year: int, month: int) -> str:
    """
    Summarize the monthly news data using OpenAI's GPT model.
    """
    with open(f'data/news_{year}_{month:02d}.txt', 'r') as f:
        news_text = f.read()
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
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
        model="text-embedding-ada-002",
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

def get_historical_death_counts(year: int, month: int, num_months: int = 3) -> List[Dict]:
    """
    Get historical death counts for the specified number of past months.
    """
    df = pd.read_csv('data/historical_death_counts.csv')
    historical_counts = []
    
    for i in range(num_months):
        past_date = datetime(year, month, 1) - timedelta(days=(i+1)*30)
        past_year, past_month = past_date.year, past_date.month
        past_count = df[(df['year'] == past_year) & (df['month'] == past_month)]['death_count'].values
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
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI trained to predict death counts based on news summaries and historical data."},
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

def run_prediction_cycle(year: int, month: int, news_data: str, death_count: int) -> int:
    """
    Run a complete prediction cycle for a given month.
    """
    prepare_monthly_data(year, month, news_data, death_count)
    summary = summarize_monthly_news(year, month)
    create_vector_embedding(year, month, summary, death_count)
    next_month_prediction = predict_next_month(year, month)
    return next_month_prediction

if __name__ == "__main__":
    # Example usage
    current_year = 2023
    current_month = 5
    sample_news_data = "Sample news data for the month..."
    sample_death_count = 150
    
    prediction = run_prediction_cycle(current_year, current_month, sample_news_data, sample_death_count)
    print(f"Predicted death count for next month: {prediction}")
    
    # Evaluate predictions for the previous year
    evaluation_results = evaluate_predictions(current_year - 1)
    print(f"Evaluation results for {current_year - 1}:")
    print(f"Mean Absolute Error: {evaluation_results['MAE']}")
    print(f"Root Mean Square Error: {evaluation_results['RMSE']}")

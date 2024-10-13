import json
import pandas as pd
import numpy as np
from typing import List, Dict
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
from dateutil.relativedelta import relativedelta
import anthropic
import sys
import os
from dotenv import load_dotenv
from fuzzywuzzy import process
import chromadb
from chromadb.utils import embedding_functions

sys.path.append('../../')
from components.universal import get_country_id
from pulling_gdelt import get_gdelt_data, create_dataset

# Load environment variables
load_dotenv()

# Constants
MAX_RECORDS = 10
MODEL_CHOICE = "claude"  # "gpt" or "claude"
DATA_PERTURB = ""  # or "" for militia movement
SAMPLES = 3
USE_CHROMA = True  

# Check for country name argument
if len(sys.argv) < 2:
    print("Please provide a country name as an argument. Ex: python clean_cycles.py 'Sri Lanka'")
    sys.exit(1)

COUNTRY_NAME = sys.argv[1].lower()
DATA_SOURCE = f'/Users/kaylahuang/Desktop/conflicts/data/views/{COUNTRY_NAME}.csv'
COUNTRY_FOLDER = f"{COUNTRY_NAME}_data"

COUNTRY_ID = get_country_id(COUNTRY_NAME)
print(f"COUNTRY_ID: {COUNTRY_ID}")

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Initialize vector database
if USE_CHROMA:
    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.get_or_create_collection(
        name=f"hadr-all",
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
    )
else:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = f"hadr-1536-{COUNTRY_NAME}"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    index = pc.Index(index_name)

def prepare_monthly_data(year: int, month: int, queries: List[str]) -> str:
    start_date = datetime(year, month, 1)
    end_date = (start_date + relativedelta(months=1)) - relativedelta(days=1)
    
    urls, _ = get_gdelt_data(queries, start_date, end_date, max_records=MAX_RECORDS)
    news_data = create_dataset(urls)
    
    combined_news = "\n\n".join([f"Title: {article['title']}\n{article['body']}" for article in news_data])
    
    with open(f'{COUNTRY_FOLDER}/news_{year}_{month:02d}.txt', 'w', encoding='utf-8') as f:
        f.write(combined_news)
    
    print(f"Saved news data for {year}-{month:02d} to data/news_{year}_{month:02d}.txt")
    return combined_news

def summarize_monthly_news(year: int, month: int) -> str:
    summary_file = f'{COUNTRY_FOLDER}/summary_{year}_{month:02d}.json'
    
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            return json.load(f)['summary']
    
    with open(f'{COUNTRY_FOLDER}/news_{year}_{month:02d}.txt', 'r') as f:
        news_text = f.read()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": "Summarize the key violent incidents, their locations, and estimated casualties from the following news articles. Limit your summary to about 500 words."},
            {"role": "user", "content": news_text}
        ]
    )
    
    summary = response.choices[0].message.content
    
    with open(summary_file, 'w') as f:
        json.dump({'year': year, 'month': month, 'summary': summary}, f)
    
    return summary

def create_vector_embedding(year: int, month: int, summary: str, feature: str):
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    embedding = embeddings.embed_documents([summary])[0]
    
    if USE_CHROMA:
        chroma_collection.add(
            documents=[summary],
            embeddings=[embedding],
            metadatas=[{"year": year, "month": month, "country": COUNTRY_NAME, "feature": feature}],
            ids=[f"{year}_{month:02d}"]
        )
    else:
        index.upsert([(f"{year}_{month:02d}", embedding, {"type": "militia"})])

def get_similar_months(current_year: int, current_month: int, current_summary: str, top_k: int = 3) -> List[Dict]:
    """
    Find similar past months based on the current month's summary.
    Only returns months that are before or equal to the current month.
    """
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    query_embedding = embeddings.embed_documents([current_summary])[0]
    
    if USE_CHROMA:
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k*2,
            where={
                "$or": [
                    {"year": {"$lt": current_year}},
                    {"$and": [
                        {"year": current_year},
                        {"month": {"$lt": current_month}}
                    ]}
                ]
            }
        )
        filtered_months = [
            {
                "id": results["ids"][0][i],
                "score": results["distances"][0][i],
                "metadata": results["metadatas"][0][i]
            }
            for i in range(min(top_k, len(results["ids"][0])))
        ]
    else:
        similar_months = index.query(vector=query_embedding, top_k=top_k*3, include_metadata=True)
        filtered_months = [
            match for match in similar_months['matches']
            if (year := int(match['id'].split('_')[0])) < current_year or 
               (year == current_year and int(match['id'].split('_')[1]) < current_month)
        ][:top_k]
    
    print(f"****** similar months: {filtered_months}\033[0m")
    return filtered_months

def load_month_key():
    month_key_df = pd.read_csv('/Users/kaylahuang/Desktop/conflicts/data/views/month_key.csv')
    return {row['month_id']: (row['Year'], row['Month']) for _, row in month_key_df.iterrows()}

def get_historical_death_counts(year: int, month: int, num_months: int = 3) -> List[Dict]:
    df = pd.read_csv(DATA_SOURCE)
    month_key = load_month_key()
    current_month_id = next(id for id, (y, m) in month_key.items() if y == year and m == month)
    
    historical_counts = [
        {
            'year': month_key[past_month_id][0],
            'month': month_key[past_month_id][1],
            'death_count': int(df[df['month_id'] == past_month_id]['ged_sb'].values[0])
        }
        for past_month_id in range(current_month_id - num_months, current_month_id)
        if past_month_id in month_key
    ]
    
    print(f"****** historical death data: {historical_counts}")
    return historical_counts

def scrub_summary(summary: str, scrub_all: bool = False) -> str:
    print(f"****** scrubbing summary: {summary}")
    system_content = "You are an AI trained to scrub summaries of news articles. "
    if scrub_all:
        system_content += "Replace any identifiable real-world people, organizations, places, and dates with generic placeholders ('Person A' instead of 'John Doe', 'Organization A' instead of 'The Red Cross', 'Location A' instead of 'Tokyo, Japan', 'Date A' instead of 'January 1, 2023'). "
    else:
        system_content += "Replace only dates with the month (if applicable) and a year placeholder (e.g. 'January, Year 4' instead of 'January 1, 2023' or 'Year 1' instead of '2019'). Keep the distance between the years the same. "
    system_content += "Do not change any other text or context. Return only the scrubbed summary."

    print(f"****** system content: {system_content}")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": summary}
        ]
    )
        
    scrubbed_summary = response.choices[0].message.content
    return summary if not scrubbed_summary else scrubbed_summary

def predict_next_month(year: int, month: int, samples: int = 3, model: str = "gpt", prediction_type: str = "point") -> List[int]:
    with open(f'{COUNTRY_FOLDER}/summary_{year}_{month:02d}.json', 'r') as f:
        current_data = json.load(f)
    
    df = pd.read_csv(DATA_SOURCE)
    similar_months = get_similar_months(year, month, current_data['summary'])
    historical_counts = get_historical_death_counts(year, month)
    
    scrubbed_summary = scrub_summary(current_data['summary'], scrub_all=False)
    
    prompt = f"Current month summary: {scrubbed_summary}\n\n"
    prompt += "Similar past months:\n"
    similarity_labels = ["Most similar", "Second most similar", "Third most similar"]
    
    month_key = load_month_key()
    for i, match in enumerate(similar_months[:3]):
        match_year, match_month = map(int, match['id'].split('_'))
        month_id = next(id for id, (y, m) in month_key.items() if y == match_year and m == match_month)
        next_month_id = month_id + 1
        next_month_actual = df[df['month_id'] == next_month_id]['ged_sb'].values[0]
        prompt += f"- Month: {similarity_labels[i]}, Death count for the next month: {next_month_actual}\n"
    
    prompt += f"\nHistorical death counts for the past 3 months:\n"
    for i, count in enumerate(historical_counts, start=1):
        prompt += f"- Month: {i} month{'s' if i > 1 else ''} before, Death count: {count['death_count']}\n"
    
    if prediction_type == "point":
        prompt += f"\n Based on this information, predict the death count for the next month. Provide only the number."
    elif prediction_type == "distribution":
        prompt += f"\n Based on this information, predict a distribution for the death count for the next month. Provide the distribution in the form of an array of three numbers [low, median, high] where low and high are the bounds of the prediction interval. ONLY return the array with no other text."

    print(f"****** THE PREDICTION PROMPT: {prompt}")
    predictions = []
    
    repeat = samples if prediction_type == "point" else 1
        
    if model == "gpt":
        for _ in range(repeat):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI trained to predict death counts in civil conflicts based on news summaries and historical data."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            prediction = response.choices[0].message.content
            try:
                predictions.append(int(float(prediction)))
            except ValueError:
                print(f"Warning: Unable to convert prediction '{prediction}' to integer. Skipping this sample.")
        
    elif model == "claude":
        if prediction_type == "point":
            prompt += "\n\nNo matter what, provide only a number as your response."
            
        for _ in range(repeat):
            message = claude_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            print(message.content)
            
            prediction = message.content[0].text
            try:
                if prediction_type == "point":
                    predictions.append(int(float(prediction)))
                elif prediction_type == "distribution":
                    print(f"****** THE DISTRIBUTION PREDICTION: {prediction}")
                    predictions = [int(x.strip()) for x in prediction.strip('[]').split(',')]
                    print(f"****** THE split: {predictions}")
            except ValueError:
                print(f"Warning: Unable to convert prediction '{prediction}' to integer. Skipping this sample.")
            
    if not predictions:
        print(f"Warning: No valid predictions were made. Returning 0.")
        return [0]
    
    print(f"****** THE PREDICTIONS: {predictions}")
    return predictions # this is for the NEXT MONTH!

def prepare_and_insert_month(year: int, month: int, queries: List[str], feature: str) -> None:
    if not os.path.exists(COUNTRY_FOLDER):
        os.makedirs(COUNTRY_FOLDER)
        print(f"Created folder: {COUNTRY_FOLDER}")
    
    month_id = f"{year}_{month:02d}"
    
    if USE_CHROMA:
        if chroma_collection.get(ids=[month_id])['ids']:
            print(f"Month {month_id} already exists in the vector database. Skipping.")
            return
    else:
        if index.fetch([month_id]).vectors:
            print(f"Month {month_id} already exists in the vector database. Skipping.")
            return

    print(f"Preparing and inserting data for {month_id}")
    if not os.path.exists(f'{COUNTRY_FOLDER}/news_{year}_{month:02d}.txt'):
        prepare_monthly_data(year, month, queries)
    
    summary = summarize_monthly_news(year, month)
    
    create_vector_embedding(year, month, summary, feature)
    print(f"Month {month_id} has been prepared and inserted into the vector database.")

def evaluate_predictions(year: int, queries: List[str], feature: str, forecast_months: int = 12, samples: int = 3, model: str = "gpt", prediction_type: str = "point") -> Dict[str, float]:
    print(f"Starting evaluation for year {year} with {forecast_months} forecast months")
    df = pd.read_csv(DATA_SOURCE)
    month_key_df = pd.read_csv('/Users/kaylahuang/Desktop/conflicts/data/views/month_key.csv')
    
    predictions_data = []
    
    for month in range(1, forecast_months + 1):
        print(f"\033[92mPROCESSING MONTH {year}-{month:02d}\033[0m")
        month_id = month_key_df[(month_key_df['Year'] == year) & (month_key_df['Month'] == month)]['month_id'].values[0]
        actual = df[df['month_id'] == month_id]['ged_sb'].values[0]
        
        prev_month = 12 if month == 1 else month - 1
        prev_year = year - 1 if month == 1 else year
        
        prepare_and_insert_month(prev_year, prev_month, queries, feature)
        
        predicted = predict_next_month(prev_year, prev_month, samples, model, prediction_type)
        
        for i, pred in enumerate(predicted):
            predictions_data.append({
                'month_id': month_id,
                'country_id': COUNTRY_ID, 
                'draw': i,
                'outcome': pred
            })
        
        print(f"\033[91mActual: {actual}, Predicted: {predicted}\033[0m")
    
    predictions_df = pd.DataFrame(predictions_data)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'/Users/kaylahuang/Desktop/conflicts/hadr/results/{COUNTRY_NAME}/{year}/{COUNTRY_NAME}_RAG_{year}_{DATA_PERTURB}{current_time}.csv'
    predictions_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    actual_counts = df[df['month_id'].isin(predictions_df['month_id'].unique())]['ged_sb'].values
    mean_predictions = predictions_df.groupby('month_id')['outcome'].mean().values
    
    mae = np.mean(np.abs(actual_counts - mean_predictions))
    rmse = np.sqrt(np.mean((actual_counts - mean_predictions)**2))
    
    return {"MAE": mae, "RMSE": rmse}

def run_prediction_cycle(year: int, month: int, queries: List[str], samples: int = 3, model: str = "gpt", prediction_type: str = "point") -> List[int]:
    print(f"Running prediction cycle for {year}-{month:02d}")
    
    prepare_and_insert_month(year, month, queries)
    
    month_key = load_month_key()
    month_id = next(id for id, (y, m) in month_key.items() if y == year and m == month)
    df = pd.read_csv(DATA_SOURCE)
    death_count = df[df['month_id'] == month_id]['ged_sb'].values[0]
    print(f"Death count for {year}-{month:02d}: {death_count}")
    
    next_month_predictions = predict_next_month(year, month, samples, model, prediction_type)
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    next_month_id = next(id for id, (y, m) in month_key.items() if y == next_year and m == next_month)
    true_value = df[df['month_id'] == next_month_id]['ged_sb'].values[0]
    print(f"Predictions for {next_year}-{next_month:02d}: {next_month_predictions}")
    print(f"True value for {next_year}-{next_month:02d}: {true_value}")
    
    return next_month_predictions

def prepare_and_insert_range(start_year: int, start_month: int, n_months: int, queries: List[str], feature: str) -> None:
    start_date = datetime(start_year, start_month, 1)
    
    for i in range(n_months):
        current_date = start_date + relativedelta(months=i)
        current_year, current_month = current_date.year, current_date.month
        
        print(f"\033[94mPreparing and inserting data for {current_year}-{current_month:02d} ({i+1}/{n_months})\033[0m")
        prepare_and_insert_month(current_year, current_month, queries, feature)

if __name__ == "__main__":
    run_one_test = False
    run_insertion = False
    run_evaluation = True
    
    current_year = 2017
    current_month = 1
    
    query_lists = {
        "drc": (["drc", "M23", "ADF", "FDLR"], "militia activity"),
        "myanmar": (["myanmar", "ULA", "Arakan", "TNLA", "shan state", "KIA"], "militia activity"),
        "afghanistan": (["afghanistan", "taliban", "ISKP", "ISIS", "ANSF", "Haqqani Network"], "militia activity"),
        "somalia": (["somalia", "Al-Shabaab", "AMISOM", "SNA", "ISS"], "militia activity"),
        "syria": (["syria", "refugee", "displaced", "asylum seeker", "humanitarian crisis"], "refugee movements"),
    }
    
    queries, feature = query_lists.get(COUNTRY_NAME.lower(), ([], None))
    if not queries:
        raise ValueError(f"No queries found for country: {COUNTRY_NAME}")
    
    evaluation_year = 2019
    evaluation_month = 1
    
    if run_one_test: 
        print("-------------------TEST PREDICTION---------------------")
        print(run_prediction_cycle(current_year, current_month, queries, samples=SAMPLES, model=MODEL_CHOICE, feature=feature))
    
    if run_insertion: 
        print("-------------------INSERTION---------------------")
        prepare_and_insert_range(current_year, start_month=1, n_months=36, queries=queries, feature=feature)
    
    if run_evaluation: 
        print("-------------------EVALUATION---------------------")
        evaluation_results = evaluate_predictions(evaluation_year, queries, forecast_months=12, samples=SAMPLES, model=MODEL_CHOICE, prediction_type="distribution", feature=feature)
        print(f"Evaluation results for {COUNTRY_NAME} in {evaluation_year}:")
        print(f"Mean Absolute Error: {evaluation_results['MAE']}")
        print(f"Root Mean Square Error: {evaluation_results['RMSE']}")
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from trafilatura import extract_metadata

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Chroma client
chroma_client = chromadb.Client()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)

def get_gdelt_data(country, queries, start_date, end_date, max_records=50):
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    lang = "sourcelang:english"
    urls = []
    
    # combined_query = f"{country} AND ({' OR '.join(queries)}) {lang}"
    # params = {
    #     "query": combined_query,
    #     "mode": "artlist",
    #     "format": "json",
    #     "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
    #     "enddatetime": end_date.strftime("%Y%m%d%H%M%S"),
    #     "maxrecords": max_records,
    # }
    for query in queries[1:]: 
        lang_query = country + f" AND {query}" + " " + lang
        params = {
            "query": lang_query,
            "mode": "artlist",
            "format": "json",
            "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end_date.strftime("%Y%m%d%H%M%S"),
            "maxrecords": max_records,
        }
        
        request_url = f"{base_url}?{'&'.join([f'{key}={value}' for key, value in params.items()])}" 
        print(f"Request URL: {request_url}")
        response = requests.get(base_url, params=params).json()
        urls.extend([article["url"] for article in response.get("articles", [])])
    
    return urls
    
    # response = requests.get(base_url, params=params).json()
    # return [article["url"] for article in response.get("articles", [])]

def scrape_article(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        metadata = extract_metadata(response.content)
        title = soup.title.string if soup.title else "No title found"
        description = metadata.description if metadata and hasattr(metadata, 'description') else "No description found"
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        content = "\n".join(paragraphs)
        
        return {
            "url": url,
            "title": title,
            "body": content,
            "description": description,
        }
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None

def save_news_text(country, feature, year, month, articles):
    folder_name = f"{country}_data"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    file_name = f"{folder_name}/news_{year}_{month:02d}.txt"
    
    with open(file_name, 'w', encoding='utf-8') as f:
        for article in articles:
            if article:
                f.write(f"Title: {article['title']}\n")
                f.write(f"{article['body']}\n\n")
    
    print(f"Saved news data for {year}-{month:02d} to {file_name}")

def summarize_articles(articles, feature):
    combined_text = "\n\n".join([f"Title: {article['title']}\n{article['body']}" for article in articles if article])
    
    prompt = f"""Summarize the following news articles, focusing on {feature} in the context of the country mentioned. 
    Highlight key events, trends, and developments related to {feature}. Keep the summary concise, around 500 words.

    Articles:
    {combined_text}

    Summary:"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant specializing in summarizing news articles about specific features in different countries."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content.strip()

def store_summary(collection, country, feature, year, month, summary):
    id = f"{country}_{feature}_{year}_{month:02d}"
    collection.add(
        documents=[summary],
        metadatas=[{"country": country, "feature": feature, "year": year, "month": month}],
        ids=[id]
    )
    print(f"Stored summary for {id}")

def analyze_feature(country, feature, queries, start_date, end_date):
    # Get or create Chroma collection
    collection = chroma_client.get_or_create_collection(
        name="feature_summaries",
        embedding_function=openai_ef
    )

    # Fetch articles from GDELT
    urls = get_gdelt_data(country, queries, start_date, end_date)
    
    # Scrape articles
    articles = [scrape_article(url) for url in tqdm(urls, desc="Scraping articles")]
    articles = [article for article in articles if article]  # Remove None values
    
    # Save news text
    save_news_text(country, feature, start_date.year, start_date.month, articles)
    
    # Summarize articles
    summary = summarize_articles(articles, feature)
    print(summary)
    
    # Store summary in Chroma
    # store_summary(collection, country, feature, start_date.year, start_date.month, summary)

    return summary

if __name__ == "__main__":
    # Example usage for Syria
    country = "Syria"
    feature = "refugee movements"
    queries = ["refugee", "displaced", "asylum seeker", "humanitarian crisis"]
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2018, 1, 31)

    summary = analyze_feature(country, feature, queries, start_date, end_date)
    print(f"Summary for {feature} in {country}:")
    print(summary)
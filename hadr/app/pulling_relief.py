## BASIC IMPORTS
import requests
from datetime import datetime
import os
from tqdm import tqdm
from bs4 import BeautifulSoup

# Import shared functions from pulling_gdelt.py
from pulling_gdelt import (
    scrape, split_documents, process_articles, query_llm,
    OpenAI, PromptTemplate, RunnableSequence, Chroma, OpenAIEmbeddings,
    CharacterTextSplitter, TextLoader
)

## API KEYS
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
print("\033[92mOPENAI API KEY DETECTED\033[0m" if openai.api_key else "\033[91mNO API KEY DETECTED\033[0m")

def get_reliefweb_data(query, country, start_date, end_date, max_records=5):
    base_url = "https://api.reliefweb.int/v1/reports"
    params = {
        "appname": "kaylahuang.com",
        "query[value]": query,
        "query[operator]": "AND",
        "filter[operator]": "AND",
        "filter[conditions][0][field]": "country",
        "filter[conditions][0][value]": country,
        "filter[conditions][1][field]": "date.created",
        "filter[conditions][1][value][from]": start_date.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "filter[conditions][1][value][to]": end_date.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "limit": max_records
    }
    
    response = requests.get(base_url, params=params)
    return response.json()

def reliefweb_timeline(query, country, start_date, end_date):
    base_url = "https://api.reliefweb.int/v1/reports"
    params = {
        "appname": "kaylahuang.com",
        "query[value]": query,
        "query[operator]": "AND",
        "filter[operator]": "AND",
        "filter[conditions][0][field]": "country",
        "filter[conditions][0][value]": country,
        "filter[conditions][1][field]": "date.created",
        "filter[conditions][1][value][from]": start_date.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "filter[conditions][1][value][to]": end_date.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "facets[0][field]": "date.created",
        "facets[0][interval]": "day"
    }
    
    response = requests.get(base_url, params=params)
    return response.json()

def create_dataset(data):
    processed_data = []
    for item in tqdm(data['data'], desc="Processing reports"):
        try:
            url = item['href']
            title = item['fields']['title']
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            body = "\n".join(paragraphs)
            
            processed_data.append({
                "url": url,
                "title": title,
                "body": body,
            })
            
        except Exception as e:
            print(f"Error processing report: {str(e)}")
    
    return processed_data

def main():
    country = "somalia"
    queries = ["Al-Shabaab", "AMISOM", "SNA", "ISS"]
    start_date = datetime(2015, 6, 1)
    end_date = datetime(2015, 6, 30)
    max_records = 10

    all_data = []

    for query in queries:
        print(f"Querying ReliefWeb for: {query} in {country} from {start_date} to {end_date}")
        
        data = get_reliefweb_data(query, country, start_date, end_date, max_records)
        print(f"Fetched {len(data['data'])} reports for query: {query}")
        
        all_data.extend(data['data'])

    print(f"Total reports fetched: {len(all_data)}")
    print(all_data)

    # Process articles and create vector store
    processed_data = create_dataset({'data': all_data})
    
    # scrape(processed_data)
    # db = process_articles(processed_data)

    # # Query the LLM
    # user_query = "What are the main humanitarian challenges in Myanmar?"
    # context = db.similarity_search(user_query, k=3)
    # response = query_llm(user_query, context)
    # print(response)

    # # Generate timeline data (but don't plot it)
    # timeline_data = reliefweb_timeline(country, country, start_date, end_date)
    # print(timeline_data)

if __name__ == "__main__":
    main()
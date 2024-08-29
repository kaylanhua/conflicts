## BASIC IMPORTS
import requests
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import os
import plotly.express as px

from tqdm import tqdm
from trafilatura.sitemaps import sitemap_search
from trafilatura import extract_metadata

# LANGCHAIN
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# LLAMA
from llama_index.core import ServiceContext, PromptHelper, VectorStoreIndex, SimpleDirectoryReader, set_global_service_context 
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.node_parser import SimpleNodeParser

## API KEYS
import openai
# openai.organization = "org-raWgaVqCbuR9YlP1CIjclYHk" # Harvard
openai.api_key = os.getenv("OPENAI_API_KEY")
print("\033[92mOPENAI API KEY DETECTED\033[0m" if openai.api_key else "\033[91mNO API KEY DETECTED\033[0m")


# '''
# GDELT PROCESSING SECTION
# '''
def get_gdelt_data(queries, start_date, end_date, max_records=5):
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    print(f"queries: {queries}")
    
    if len(queries) > 1:
        combined_query = " OR ".join(queries)
        lang_query = f"({combined_query} sourcelang:english)" 
        # TODO consider removing language
    else:
        lang_query = f"{queries[0]} sourcelang:english"
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
    urls = [article["url"] for article in response.get("articles", [])]
    return urls, response

def gdelt_timeline(queries, start_date, end_date, timelinesmooth=5):
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    print(f"timeline queries: {queries}")
    
    if len(queries) > 1:
        combined_query = " OR ".join(queries)
        lang_query = f"{combined_query}"
    else: 
        lang_query = f"{queries[0]}"
    params = {
        "query": lang_query,
        "mode": "timelinevolinfo",
        "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_date.strftime("%Y%m%d%H%M%S"),
        "format": "json",
        "TIMELINESMOOTH": timelinesmooth,
    }
    
    request_url = f"{base_url}?{'&'.join([f'{key}={value}' for key, value in params.items()])}" 
    print("Request URL for timeline:", request_url)
    response = requests.get(base_url, params=params)
    return response


def plot_gdelt_timeline(timeline_data):
    # Extract the data from the timeline
    dates = []
    values = []
    for item in timeline_data['timeline']:
        for data_point in item['data']:
            dates.append(datetime.strptime(data_point['date'], '%Y%m%dT%H%M%SZ'))
            values.append(data_point['value'])
    
    # Create a DataFrame
    df = pd.DataFrame({'Date': dates, 'Volume Intensity': values})
    
    # Create the plot
    fig = px.line(df, x='Date', y='Volume Intensity', 
                  title='GDELT Timeline: Volume Intensity Over Time',
                  labels={'Volume Intensity': 'Volume Intensity', 'Date': 'Date'},
                  line_shape='linear')
    
    # Customize the layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Volume Intensity',
        font=dict(size=12),
        hovermode='x unified'
    )
    
    return fig


def create_dataset(list_of_websites: list) :
    """
    scrapes the data from the list of websites
    """
    data = []
    print(list_of_websites)
    for url in tqdm(list_of_websites, desc="urls"):
        try:
            # Send HTTP request to the URL with a timeout of 8 seconds
            response = requests.get(url, timeout=5)
            response.raise_for_status()  # Check for successful response
            # Parse HTML content
            soup = BeautifulSoup(response.content, "html.parser")
            
            metadata = extract_metadata(response.content)
            title = soup.title.string if soup.title else "No title found"
            description = metadata.description if metadata and hasattr(metadata, 'description') else "No description found"
            # Extract text from each paragraph
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            content = "\n".join(paragraphs)
            d = {
                "url": url,
                "title": title,
                "body": content,
                "description": description,
            }
            data.append(d)
        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error: {errh}")
        except requests.exceptions.ConnectionError as errc:
            print(f"Error Connecting: {errc}")
        except requests.exceptions.Timeout as errt:
            print(f"Timeout Error: {errt}")
        except requests.RequestException as err:
            print(f"Error during requests to {url}: {str(err)}")
    return data

def scrape(list_of_websites: list) -> None:
    data = create_dataset(list_of_websites)

    current_time = datetime.now().strftime("%d%H%M%S")
    dataset_filename = f"./data/dataset_{current_time}.txt"

    with open(dataset_filename, "w", encoding="utf-8") as file:
        for paragraph in data:
            file.write("\n" + paragraph["title"] + "\n")
            file.write(paragraph["body"]+"\n\n")
            

## FRAGMENTING DOCUMENTS
def split_documents():
    """Load the most recent file from the data folder, split it into chunks, embed each chunk and load it into the vector store."""
    data_folder = "./data"
    files = os.listdir(data_folder)
    latest_file = max([os.path.join(data_folder, f) for f in files], key=os.path.getctime)
    raw_documents = TextLoader(latest_file).load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(raw_documents)


def process_articles(urls):
    # generate documents in /data folder
    scrape(urls)
    # read from data folder
    documents = split_documents()
    # create vector store 
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(documents, embeddings)
    return db


def query_llm(query, context):
    llm = OpenAI()
    template = """
    You are an AI assistant specializing in armed conflicts and international relations.
    Use the following context to answer the question. If you can't answer based on the context, say "I don't have enough information to answer that."

    Context: {context}

    Human: {human_input}
    AI Assistant: """
    
    prompt = PromptTemplate(template=template, input_variables=["context", "human_input"])
    chain = RunnableSequence(prompt | llm)
    response = chain.invoke({"context": context, "human_input": query})
    return response

def main():
    # queries = ["rapid support forces", "militia", "RSF", "Paramilitary", "hemedti", "janjaweed"]
    queries = ["rsf"]
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2018, 3, 31)
    # urls, _ = get_gdelt_data(queries, start_date, end_date, max_records=5)
    # print(urls)
    
    timeline_response = gdelt_timeline(queries, start_date, end_date, timelinesmooth=5)
    print(timeline_response)
    if timeline_response.status_code == 200:
        timeline_data = timeline_response.json()
        figure = plot_gdelt_timeline(timeline_data)
        figure.show()

if __name__ == "__main__":
    main()
    

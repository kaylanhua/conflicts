## BASIC IMPORTS
import streamlit as st
from streamlit_timeline import st_timeline

import requests
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import os
import json
import plotly.express as px
import io
import base64


# '''
# GDELT PROCESSING SECTION
# '''
def get_gdelt_data(queries, start_date, end_date, max_records=5):
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    print(queries)
    
    if len(queries) > 1:
        combined_query = " OR ".join(queries)
        lang_query = f"({combined_query} sourcelang:english)"
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
    print("Request URL:", request_url)
    response = requests.get(base_url, params=params).json()
    urls = [article["url"] for article in response.get("articles", [])]
    return urls, response


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
            title = soup.title.string
            description = metadata.description
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
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run(context=context, human_input=query)
    return response

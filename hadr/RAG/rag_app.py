import streamlit as st
import requests
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import os
import json

from llama_index.core import ServiceContext, PromptHelper, VectorStoreIndex, SimpleDirectoryReader, set_global_service_context 
from llama_index.llms.openai import OpenAI
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.node_parser import SimpleNodeParser

from tqdm import tqdm
from trafilatura.sitemaps import sitemap_search
from trafilatura import extract_metadata

## API KEYS
import openai
from openai import OpenAI
openai.organization = "org-raWgaVqCbuR9YlP1CIjclYHk" # Harvard
openai.api_key = os.getenv("OPENAI_API_KEY")
print("\033[92mOPENAI API KEY DETECTED\033[0m" if openai.api_key else "\033[91mNO API KEY DETECTED\033[0m")

# Your existing functions (create_dataset, scrape) remain unchanged

def get_gdelt_data(query, start_date, end_date):
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_date.strftime("%Y%m%d%H%M%S"),
        "maxrecords": 100,
    }
    response = requests.get(base_url, params=params).json()
    urls = [article["url"] for article in response.get("articles", [])]
    return urls

def create_dataset(list_of_websites: list) :
    """
    scrapes the data from the list of websites
    """
    data = []
    for url in tqdm(list_of_websites, desc="urls"):
        try:
            # Send HTTP request to the URL
            response = requests.get(url)
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

def process_articles(articles):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents([Document(page_content=article["body"]) for article in articles])
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings)
    return vectorstore

def query_llm(query, context):
    llm = OpenAI(temperature=0.7)
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

def create_timeline(events, vectorstore):
    query = "Create a timeline of important events based on the given information."
    docs = vectorstore.similarity_search(query)
    context = "\n".join([doc.page_content for doc in docs])
    timeline = query_llm(query, context)
    return timeline

def create_network_graph(vectorstore):
    query = "Identify the main actors involved in this conflict and their relationships."
    docs = vectorstore.similarity_search(query)
    context = "\n".join([doc.page_content for doc in docs])
    actors_info = query_llm(query, context)
    
    # Parse the LLM response to create a network graph
    # This is a simplified version; you might need more sophisticated parsing
    G = nx.Graph()
    for line in actors_info.split('\n'):
        if '-' in line:
            actor1, actor2 = line.split('-')
            G.add_edge(actor1.strip(), actor2.strip())
    
    return G, actors_info

def main():
    st.title("War News Tracker")

    war_name = st.text_input("Enter the name of a war:")
    if war_name:
        with st.spinner("Fetching and processing data..."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Last 30 days
            news_data = get_gdelt_data(war_name, start_date, end_date) # returns list of urls
            vectorstore = process_articles(news_data.get("articles", []))

        st.subheader("Timeline of Important Events")
        timeline = create_timeline(news_data.get("articles", []), vectorstore)
        st.write(timeline)

        st.subheader("Network of Actors")
        G, actors_info = create_network_graph(vectorstore)
        fig, ax = plt.subplots()
        nx.draw(G, with_labels=True, ax=ax)
        st.pyplot(fig)
        st.write(actors_info)

        st.subheader("Recent News Articles")
        for article in news_data.get("articles", []):
            st.write(f"**{article['title']}**")
            st.write(article['url'])
            st.write(article['seendate'])
            
            # Add a button to query the LLM about this specific article
            if st.button(f"Analyze this article"):
                query = f"Summarize the key points of this article about {war_name}"
                docs = vectorstore.similarity_search(query)
                context = "\n".join([doc.page_content for doc in docs])
                analysis = query_llm(query, context)
                st.write(analysis)
            
            st.write("---")

if __name__ == "__main__":
    main()
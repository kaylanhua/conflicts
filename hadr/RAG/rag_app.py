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

# LANGCHAIN
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
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

from tqdm import tqdm
from trafilatura.sitemaps import sitemap_search
from trafilatura import extract_metadata

## API KEYS
import openai
openai.organization = "org-raWgaVqCbuR9YlP1CIjclYHk" # Harvard
openai.api_key = os.getenv("OPENAI_API_KEY")
print("\033[92mOPENAI API KEY DETECTED\033[0m" if openai.api_key else "\033[91mNO API KEY DETECTED\033[0m")


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

# '''
# TIMELINE SECTION
# '''

conflict_schema = ResponseSchema(name="conflicts", description="List of conflicts")
conflict_parser = StructuredOutputParser.from_response_schemas([conflict_schema])

event_schema = ResponseSchema(name="events", description="List of timeline events")
event_parser = StructuredOutputParser.from_response_schemas([event_schema])

actor_schema = ResponseSchema(name="actors", description="List of main actors in the conflict")
actor_parser = StructuredOutputParser.from_response_schemas([actor_schema])


def query_llm_for_conflicts(country):
    llm = OpenAI(temperature=0.7)
    template = """
    You are an AI assistant specializing in armed conflicts and international relations.
    Please provide a list of major armed conflicts that have occurred in {country}.
    Format your response as a JSON array of objects, where each object has the following fields:
    - name: the name of the conflict
    - start_year: the year the conflict started (integer)
    - end_year: the year the conflict ended, or "ongoing" if it's still active (integer or string)

    {format_instructions}

    Human: List conflicts in {country}
    AI Assistant:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["country"],
        partial_variables={"format_instructions": conflict_parser.get_format_instructions()}
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run(country=country)
    return conflict_parser.parse(response)

def query_llm_for_timeline(conflict_name, start_year, end_year):
    llm = OpenAI(temperature=0.7)
    template = """
    You are an AI assistant specializing in armed conflicts and international relations.
    Please provide a timeline of important events for the {conflict_name}, which occurred from {start_year} to {end_year}.
    Format your response as a JSON array of objects, where each object has the following fields:
    - date: the date of the event in "YYYY-MM-DD" format (use an approximate date if the exact date is unknown)
    - description: a brief description of the event

    {format_instructions}

    Human: Create a timeline for {conflict_name}
    AI Assistant:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["conflict_name", "start_year", "end_year"],
        partial_variables={"format_instructions": event_parser.get_format_instructions()}
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run(conflict_name=conflict_name, start_year=start_year, end_year=end_year)
    return event_parser.parse(response)

def query_llm_for_actors(conflict_name):
    llm = OpenAI(temperature=0.7)
    template = """
    You are an AI assistant specializing in armed conflicts and international relations.
    Please provide a list of the main actors (individuals, groups, or countries) involved in the {conflict_name}.
    For each actor, provide a brief description of their role or involvement.
    Format your response as a JSON array of objects, where each object has the following fields:
    - name: the name of the actor
    - description: a brief description of their role or involvement in the conflict
    - type: the type of actor (e.g., "individual", "group", "country")

    {format_instructions}

    Human: List main actors in the {conflict_name}
    AI Assistant:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["conflict_name"],
        partial_variables={"format_instructions": actor_parser.get_format_instructions()}
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run(conflict_name=conflict_name)
    return actor_parser.parse(response)


def show_timeline(events):
    items = [{"id": idx + 1, "content": event["description"], "start": event["date"]} for idx, event in enumerate(events)]
    print("\033[94mitems: ", items, "\033[0m") 
    
    timeline = st_timeline(items, groups=[], options={}, height="300px")
    st.subheader("Selected item")
    st.write(timeline)

if 'conflicts' not in st.session_state:
    st.session_state.conflicts = {}

def get_conflicts(country_name):
    if country_name not in st.session_state.conflicts:
        with st.spinner("Fetching conflicts..."):
            conflicts_data = query_llm_for_conflicts(country_name)
            st.session_state.conflicts[country_name] = conflicts_data.get("conflicts", [])
    return st.session_state.conflicts[country_name]

def create_network_graph(actors):
    G = nx.Graph()
    for actor in actors:
        G.add_node(actor['name'], description=actor['description'], type=actor.get('type', 'None'))
    
    # Add edges between all actors (you might want to refine this logic)
    for i, actor1 in enumerate(actors):
        for actor2 in actors[i+1:]:
            G.add_edge(actor1['name'], actor2['name'])
    
    return G

def plot_network_graph(G):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=8, font_weight='bold')
    
    # Add node labels
    labels = nx.get_node_attributes(G, 'description')
    pos_labels = {k: (v[0], v[1]+0.1) for k, v in pos.items()}  # Adjust label positions
    nx.draw_networkx_labels(G, pos_labels, labels, font_size=6)
    
    # Convert plot to image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return graph_url

def create_timeline(country_name):
    conflicts = get_conflicts(country_name)
    
    if conflicts:
        st.subheader(f"Conflicts in {country_name}")
        for conflict in conflicts:
            end_year = conflict['end_year'] if conflict['end_year'] != "ongoing" else "Present"
            
            if 'timeline_events' not in st.session_state:
                st.session_state.timeline_events = {}
            if 'conflict_actors' not in st.session_state:
                st.session_state.conflict_actors = {}
            
            conflict_key = f"{country_name}_{conflict['name']}"
            if st.button(f"{conflict['name']}: {conflict['start_year']} – {end_year}", key=f"analyze_{conflict['name'].replace(' ', '_')}"):
                with st.spinner("Generating timeline and network graph..."):
                    if conflict_key not in st.session_state.timeline_events:
                        timeline_data = query_llm_for_timeline(conflict['name'], conflict['start_year'], end_year)
                        st.session_state.timeline_events[conflict_key] = timeline_data.get("events", [])
                    
                    if conflict_key not in st.session_state.conflict_actors:
                        actors_data = query_llm_for_actors(conflict['name'])
                        print("\033[93mactors data: ", actors_data, "\033[0m")
                        st.session_state.conflict_actors[conflict_key] = actors_data.get("actors", [])
            
            if conflict_key in st.session_state.timeline_events:
                timeline_events = st.session_state.timeline_events[conflict_key]
                if timeline_events:
                    timeline_events = [event for event in timeline_events if event.get('date') and event.get('description')]
                    show_timeline(timeline_events)
                else:
                    st.write("No timeline data retrieved.")
            
            if conflict_key in st.session_state.conflict_actors:
                actors = st.session_state.conflict_actors[conflict_key]
                if actors:
                    st.subheader("Network of Main Actors")
                    G = create_network_graph(actors)
                    graph_url = plot_network_graph(G)
                    st.image(f"data:image/png;base64,{graph_url}")
                    
                    st.subheader("Actor Information")
                    for actor in actors:
                        st.write(f"**{actor['name']}** ({actor.get('type', 'None')}): {actor['description']}")
                else:
                    st.write("No actor data retrieved.")
    else:
        st.write("No conflicts found for the specified country.")
    
    return conflicts


# '''
# APP SECTION
# '''
def main():
    st.title("conflict tracker")

    war_name = st.text_input("Enter the name of a region:")
    if war_name:
        st.subheader("Timeline of Important Events")
        create_timeline(war_name)

if __name__ == "__main__":
    main()
    
    
# inspo from the bottom left here: https://tree.nathanfriend.io/?s
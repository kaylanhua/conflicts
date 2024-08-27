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
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from tqdm import tqdm
from trafilatura.sitemaps import sitemap_search
from trafilatura import extract_metadata

## API KEYS
import openai
openai.organization = "org-raWgaVqCbuR9YlP1CIjclYHk" # Harvard
openai.api_key = os.getenv("OPENAI_API_KEY")
print("\033[92mOPENAI API KEY DETECTED\033[0m" if openai.api_key else "\033[91mNO API KEY DETECTED\033[0m")

st.set_page_config(layout="wide")


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
    chain = RunnableSequence(prompt | llm)
    response = chain.invoke({"country": country})
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
    chain = RunnableSequence(prompt | llm)
    response = chain.invoke({"conflict_name": conflict_name, "start_year": start_year, "end_year": end_year})
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
    chain = RunnableSequence(prompt | llm)
    response = chain.invoke({"conflict_name": conflict_name})
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

def plot_country_data(country_name, n=None):
    file_name = f"../../data/views/{country_name.replace(' ', '_')}.csv"
    try:
        df = pd.read_csv(file_name)
        # Read the month_key.csv file
        month_key_df = pd.read_csv('../../data/views/month_key.csv')
        
        # Merge the original dataframe with month_key_df based on month_id
        df = pd.merge(df, month_key_df[['month_id', 'Date']], on='month_id', how='left')
        
        # Convert the 'Date' column to datetime
        dates = pd.to_datetime(df['Date'], format='%b-%y')
        target = df['ged_sb']

        plt.figure(figsize=(12, 6))
        plt.plot(dates, target, label='UCDP Estimate', alpha=0.7)
        
        if n is not None:
            rolling_avg = target.rolling(window=n).mean()
            plt.plot(dates, rolling_avg, label=f'{n}-Month Rolling Average', color='red', linewidth=2)
        
        plt.xlabel('Date')
        plt.ylabel('Fatalities')
        plt.gca().spines['top'].set_color('#F9F5F1')
        plt.gca().spines['right'].set_color('#F9F5F1')

        plt.title(f'Fatalities Over Time for {country_name}')
        plt.legend()
        st.pyplot(plt)
    except FileNotFoundError:
        st.write(f"Data for {country_name} not found.")

def create_timeline(country_name):
    conflicts = get_conflicts(country_name)
    
    if conflicts:
        st.subheader(f"Conflicts in {country_name}")
        for conflict in conflicts:
            end_year = "Present" if conflict.get('end_year') in ["ongoing", None] else conflict.get('end_year', "Unknown")
            
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
    st.title("LLM conflict tracker")
    st.write("This app uses LLMs to track militia movement and visualize data on fatalities over time.")

    country_name = st.text_input("Enter the name of a region:")
    if country_name:
        # Add a slider for selecting the rolling average period
        n_months = st.slider("Select rolling average period (months)", min_value=1, max_value=10, value=3)
        
        # Plot country data with the selected rolling average period
        plot_country_data(country_name, n=n_months)
        
        st.subheader("Timeline of Important Events")
        create_timeline(country_name)

if __name__ == "__main__":
    main()
    
    
# inspo from the bottom left here: https://tree.nathanfriend.io/?s

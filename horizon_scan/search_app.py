import streamlit as st
from pyvis.network import Network
import networkx as nx
import json
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

def set_bg_color(hex_color):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {hex_color};
        }}
        div.stButton > button:first-child {{
            margin-top: 1.5em;
            height: 2.6em;   
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_color('#F9F5F1')



# Read the JSON data from file
filename = '/mount/src/conflicts/horizon_scan/graph72.json'
with open(filename, 'r') as file:
    papers = json.load(file)

len(papers)




# Function to generate and display the network
def generate_network(search_keyword=None):
    G = nx.Graph()  # or however you construct your graph
    # Nodes
    for id, info in papers.items():
        G.add_node(id, **info[0])

    # Edges
    for id, info in papers.items():
        for cited_id in info[1]:
            if G.has_node(cited_id):
                G.add_edge(id, cited_id)

    # Assuming 'G' is your NetworkX graph
    net = Network(notebook=False, height="750px", width="100%", bgcolor="#F9F5F1", font_color="black")

    ## searching for keywords and adding nodes
    for node, node_attrs in G.nodes(data=True):
        color = '#5C4033' if search_keyword and search_keyword.lower() in node_attrs['abstract'].lower() else '#C8A482'
        net.add_node(node, label=node_attrs['title'], title=node_attrs['abstract'], color=color)

    for edge in G.edges():
        net.add_edge(edge[0], edge[1], arrows='to')

    # Customize the visualization as needed
    net.repulsion(node_distance=420, central_gravity=0.33,
                spring_length=110, spring_strength=0.10,
                damping=0.95)

    # Save and show the graph as an HTML file
    net.show('str_graph.html', notebook=False)

    # Load and display the graph
    with open('str_graph.html', 'r', encoding='utf-8') as html_file:
        source_code = html_file.read()
        components.html(source_code, height=750)


# ----------------------------------------------

# Title and introduction
st.title("kayla's network analysis")
st.write('an exploration of machine learning and conflict modeling')

# Keyword search functionality
col1, col2 = st.columns([3, 1]) 

with col1:
    search_keyword = st.text_input("search for mentions of keywords (e.g., 'random forest' or 'Syria') in paper abstracts:")
with col2:
    # Button to generate the network
    search_button = st.button('search network', key='search')

if search_button:
    generate_network(search_keyword)



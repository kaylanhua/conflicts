import streamlit as st
from pyvis.network import Network
import networkx as nx
import json
import streamlit.components.v1 as components
import os
import openai
from openai import OpenAI

## needs to be the first line of code in the app
st.set_page_config(layout="wide")

## OpenAI API requirements
openai.organization = "org-raWgaVqCbuR9YlP1CIjclYHk" # Harvard
openai.api_key = os.getenv("OPENAI_API_KEY")

## css 
def styling(hex_color):
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

styling('#F9F5F1')

## ----------------------------------------------
## Setting Up Data
## ----------------------------------------------

# Read the JSON data from file
local = False
filename = 'rf_jsons/rf_edited.json' ## on local
if not local:
    filename = '/mount/src/conflicts/archive/horizon_scan/' + filename ## on server
with open(filename, 'r') as file:
    papers = json.load(file)

len(papers)

# ----------------------------------------------
# FUNCTIONS
# ----------------------------------------------

# Function to generate and display the network
## consider caching with st.cache to make this faster
def generate_network(search_keyword=None):
    # st.spinner('searching the network...')
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
        if not node_attrs['abstract']:
            node_attrs['abstract'] = 'NA'
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

def generate_review(directions):
    basics = {}

    for id, paper in papers.items():
        paper_info = paper[0]
        basics[paper_info['paperId']] = {
            'title': paper_info['title'],
            'abstract': paper_info['abstract'],
            'year': paper_info['year'],
            'author': paper_info['authors'][0]['name'] if paper_info['authors'][0] else None
        }
        
    paper_abstracts = ""

    for id, entry in basics.items():
        if not entry['abstract']:
            continue
        paper_abstracts += "Title: " + entry["title"] + '\n' + "Abstract: " + entry["abstract"] + '\n' + "Author and year: (" + entry["author"] + ', ' + str(entry["year"]) + ')\n\n'
    
    client = OpenAI() 

    completion = client.chat.completions.create(
        # turbo (1106-preview) has 128k context window, about 300 pages of text
        model="gpt-4-1106-preview", # test with: gpt-3.5-turbo, run final: gpt-4-1106-preview
        messages=[
            {"role": "system", "content": 
                """You are a writing a literature review about the use of tree-based algorithms 
                in modeling armed civil conflicts. I will provide you with multiple existing research 
                papers on the subject with the paper title, abstract, authors, and year of publication.
                Please provide a detailed literature review based on these papers, focusing on the 
                methodologies used, the evolution of these techniques over time, and their effectiveness 
                in conflict analysis. Please include references in the text in an (author, year) format."""},
            {"role": "user", "content": f"""Here is a list of the papers and the relevant information: 
                [start of papers] {paper_abstracts} [end of papers]. 
                Please also consider the following additional context provided by the user: {directions}"""}
        ]
    )

    gpt_summary = completion.choices[0].message.content
    return gpt_summary



# ----------------------------------------------

# Title and introduction
st.title("mapping research networks")
st.write('an exploration of machine learning and conflict modeling')

# Keyword search functionality
col1, col2 = st.columns([3, 1]) 

with col1:
    search_keyword = st.text_input("search for mentions of keywords (e.g., 'random forest' or 'Africa') in paper abstracts:")
with col2:
    # Button to generate the network
    search_button = st.button('search network', key='search')

if search_button:
    generate_network(search_keyword)

st.markdown('---')
st.markdown('## generate review of literature')
st.markdown('disclaimer: i don\'t have infinite money in my openai account, so please do not click this button an excessive number of times :)')

# Keyword search functionality
rl_col1, rl_col2 = st.columns([3, 1]) 

with rl_col1:
    generation_directions = st.text_input("i\'ve already done some prompting on the backend, but is there anything else you would like gpt to consider when generating the review? (review generation takes a few moments, please be patient!)")
with rl_col2:
    # Button to generate the network
    generate_button = st.button('generate review of literature', key='create_rl')
    
if generate_button:
    st.markdown("#### results of your input")
    st.write(generate_review(generation_directions))
else:
    st.markdown("#### results of the example input")
    st.markdown("**input:** please create a review that is about 1-2 pages in length and written in the style of a published journal")
    st.markdown("""**result:** Tree-based algorithms have gained traction over the last two decades as a powerful set of tools for modeling and predicting armed civil conflicts. The use of these algorithms reflects an ongoing methodological shift in the field of conflict analysis—from classic, theory-driven approaches towards more data-driven, machine learning techniques. This literature review synthesizes research papers exploring the use of tree-based algorithms to forecast armed conflicts, focusing on their methodologies, evolution, and effectiveness.

Early explorations into machine learning for conflict prediction, like the work of Weidmann (2008), recognized that traditional methods such as logistic regression were often unsuited for the "rare events" nature of armed conflicts. Weidmann's use of bagging with decision trees introduced ensembles to overcome class imbalances—a significant step that would inform future research.

Muchlinski (2016) compared Random Forests with logistic regression variants and found the former to be superior in out-of-sample predictions of civil war onset. By harnessing the algorithmic power to manage high-dimensional data without assuming linearity, Random Forests allowed for capturing complex interactions within conflict data. The pursuit of improved forecasts continued with scholarly efforts, including Henrickson (2020), who used machine learning techniques, notably Random Forests, to gauge the expected costs of war, thus expanding the potential of the tree-based algorithms beyond conflict onset to the assessment of conflict intensity.

Alongside methodology advancements, Ettensperger (2021) introduced an ensemble averaging framework to predict conflict intensity changes, combining multiple tree-based algorithms models and diverse datasets. This ensemble approach, recognizing the interplay between conflict dynamics and socio-economic structures, offered nuanced insights into the predictive capacity of tree-based algorithms.

Fascination with the algorithms' predictive power was furthered by Musumba (2021), who directly compared the performance of various supervised classification machine learning algorithms, including gradient tree boosting, against more conventional statistical approaches. SMOTE, a synthetic over-sampling technique, was highlighted for its efficacy in resolving the class imbalance issue, illustrating the continuous methodological enhancements being made within the field.

Schellens (2020) contributed by delving into the contested role of natural resources in violent conflict. By applying tree-based models, she unearthed the intricate interconnections with socio-economic variables, advancing the interpretive capacity of machine learning over logistic regression models. This work shed light on the nuanced role that features like access to resources could play in conflict prediction models.

McAlexander (2020) further advocated for the use of nonparametric methods like Random Forests. His work demonstrated the potential to uncover subtle nonlinear relationships, affording a fresh perspective on determinants of civil war onset, challenging the preference for traditional linear models.

Trends within the literature shifted to incorporate external dimensions, as illustrated by Toukan (2019), who leveraged logistic regressions and Random Forests to study the international context of civil conflicts. The innovative framing of interstate rivalries suggested that geographic proximity to rivalries increased a state's risk of civil war, illustrating how tree-based models can enrich understanding of complex geopolitical factors in conflict dynamics.

Kaufman (2019), pushing the boundaries of political science prediction, applied AdaBoosted decision trees to several political phenomena, including civil war onset prediction. The success of these boosted decision trees called attention to their wide-ranging applicability and led to further considerations of their policy relevance.

Ward (2005), one of the earlier studies in this collection, provided a critical reflection on quantitative models of conflict, underscoring the challenges faced by statistical models in accurately predicting civil conflict onsets. This retrospective view highlights the progress made and the need for continual refinement in modeling strategies.

As demonstrated by Hegre (2022), prediction competitions have proven fruitful in fostering methodological innovation, highlighting the collaborative spirit prevalent in the field. The plethora of contributions from international teams underscored the value of diverse perspectives and the need for new metrics to appreciate the advancements in predictive modeling of conflict.

Finally, Floros (2008) and Lessing (2012) represent endeavors to integrate demographic factors and the logic of cartel-state conflict into the predictive models, respectively, broadening the scope of tree-based algorithm applications.

In summary, the use of tree-based algorithms in modeling armed civil conflicts has demonstrated a marked evolution from preliminary methods to sophisticated, multi-model ensembles. The accumulated evidence points to their superiority over traditional statistical approaches, particularly in handling non-linearities, complex interactions, and high-dimensional data. While methodological challenges, such as balancing the trade-off between predictive power and model interpretability, persist, tree-based algorithms have proven themselves indispensable in seeking to understand and forecast the multifaceted phenomena of armed conflicts. Further research is expected to refine these models, expand their applicability, and continue the exploration of their predictive potential.""")
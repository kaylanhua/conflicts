import requests
import json
import time
from typing import Dict, List, Tuple

# Constants
MAX_DEPTH = 3
SAVE_INTERVAL = 5
COUNTRY = "drc"  # Change this to the country you're interested in

# Keywords related to civil conflict causes
CONFLICT_KEYWORDS = [
    "civil war", "civil conflict", "insurgency", "rebellion", "uprising",
    "ethnic conflict", "sectarian violence", "political instability",
    "economic inequality", "resource scarcity", "religious tensions",
    "ethnic tensions", "government repression", "social grievances",
    "foreign intervention", "proxy war", "failed state", "corruption",
    "poverty", "unemployment", "human rights violations"
]

def create_url(paper_id: str, is_doi: bool = False, check_citations: bool = True) -> str:
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/"
    fields = "?fields=title,abstract,year,authors,references,citations"
    if is_doi:
        return f"{endpoint}DOI:{paper_id}{fields}"
    return f"{endpoint}{paper_id}{fields}"

def try_api_call(api_endpoint: str) -> Dict:
    try:
        response = requests.get(api_endpoint)
        if response.status_code == 429:
            print("Rate limit reached. Waiting before retrying...")
            time.sleep(20)
            return try_api_call(api_endpoint)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {api_endpoint}")
        return None
    except Exception as err:
        print(f"An error occurred: {err} - {api_endpoint}")
        return None

def is_relevant_paper(paper_info: Dict) -> bool:
    text = f"{paper_info.get('title', '')} {paper_info.get('abstract', '')}".lower()
    return COUNTRY.lower() in text and any(keyword.lower() in text for keyword in CONFLICT_KEYWORDS)

def filter_papers(papers: List[Dict]) -> List[Dict]:
    return [paper for paper in papers if paper.get('title') and COUNTRY.lower() in paper['title'].lower()]

def save_graph_to_file(graph_data: Dict, file_name: str):
    with open(file_name, 'w') as file:
        json.dump(graph_data, file, indent=4, default=str)

def generate_reference_network(start_paper_id: str, is_doi: bool = False) -> Dict:
    graph = {}
    queue = [(start_paper_id, 0)]
    visited = set()
    processed_papers = 0

    while queue:
        current_paper_id, current_depth = queue.pop(0)
        
        if current_paper_id in visited or current_depth > MAX_DEPTH:
            continue

        paper_info = try_api_call(create_url(current_paper_id, is_doi))
        
        if not paper_info or not is_relevant_paper(paper_info):
            continue

        visited.add(current_paper_id)
        print(f"Processing paper: {paper_info.get('title')} at depth {current_depth}")

        references = filter_papers(paper_info.get('references', []))
        citations = filter_papers(paper_info.get('citations', []))

        relevant_papers = []
        for related_paper in references + citations:
            related_id = related_paper['paperId']
            if related_id not in visited:
                queue.append((related_id, current_depth + 1))
                relevant_papers.append(related_id)

        graph[current_paper_id] = (paper_info, relevant_papers, current_depth)

        processed_papers += 1
        if processed_papers % SAVE_INTERVAL == 0:
            save_graph_to_file(graph, f"{COUNTRY.lower()}_civil_conflict_graph.json")
            print(f"Saved backup of graph at {processed_papers} papers.")

    return graph

if __name__ == "__main__":
    # Replace this with the paper ID or DOI of your starting paper
    start_paper_id = "10.1080/13698249.2016.1144496"  # Example: "The Syrian conflict: A case study of the challenges and acute need for medical humanitarian operations for women and children internally displaced persons"
    is_doi = True

    conflict_network = generate_reference_network(start_paper_id, is_doi)
    save_graph_to_file(conflict_network, f"{COUNTRY.lower()}_civil_conflict_graph.json")
    print(f"Finished scraping. Total papers found: {len(conflict_network)}")
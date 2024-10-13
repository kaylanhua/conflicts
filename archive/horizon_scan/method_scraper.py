import requests
import time
import json
import argparse

def semantic_scholar_bulk_search(query, fields, limit=300, token=None):
    url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    params = {
        "query": query,
        "limit": limit,
        "fields": fields
    }
    if token:
        params["token"] = token
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None
    
def is_conflict_forecasting(title, abstract):
    forecasting_keywords = [' forecast', ' model']
    conflict_keywords = [' war ', ' wars ', 'civil ']
    
    text = (title + " " + (abstract or "")).lower()
    
    has_forecasting = any(keyword in text for keyword in forecasting_keywords)
    has_conflict = any(keyword in text for keyword in conflict_keywords)
    
    return has_forecasting and has_conflict

def save_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def main(method):
    query = f'"{method}" AND conflict'
    fields = "title,abstract,url,year,citationCount"
    all_results = []
    token = None

    while True:
        print(f"Fetching next batch of results...")
        response = semantic_scholar_bulk_search(query, fields, token=token)
        
        if not response or 'data' not in response:
            break
        
        papers = response['data']
        if not papers:
            break
        
        for paper in papers:
            if is_conflict_forecasting(paper['title'], paper.get('abstract', '')):
                all_results.append(paper)
        
        print(f"Found {len(all_results)} relevant papers so far.")
        
        token = response.get('next')
        if not token:
            break
        
        time.sleep(1)  # Be nice to the API

    # Sort results by citation count
    all_results.sort(key=lambda x: x.get('citationCount', 0), reverse=True)
    
    filename = f'{method.replace(" ", "_")}_papers.json'
    save_results(all_results, filename)
    print(f"Saved {len(all_results)} papers to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search for conflict forecasting papers using a specific method.')
    parser.add_argument('method', type=str, help='The method to search for (e.g., "random forest", "transformers", "linear models")')
    args = parser.parse_args()
    
    main(args.method)
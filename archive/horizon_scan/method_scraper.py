import requests
import time
import json

def create_url(id, is_doi=False, full_details=True):
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/"
    # fields = '?fields=title'
    fields = '?fields=title,abstract,url,year,authors,referenceCount,citationCount'
    if full_details:
        fields += ',citations,references'
    
    if is_doi:
        return endpoint + "DOI:" + id + fields
    else: 
        return endpoint + id + fields

def try_api_call(api_endpoint):
    try:
        response = requests.get(api_endpoint)
        print(f"req is {api_endpoint}")
        if response.status_code == 429:
            print("Failed to reach endpoint. Waiting before retrying...")
            time.sleep(20)  # Sleep for 20 seconds
            return try_api_call(api_endpoint)  # Retry the request
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {api_endpoint}")
        return None
    except Exception as err:
        print(f"An error occurred: {err} - {api_endpoint}")
        return None

def save_dict_to_file(graph_data, file_name):
    with open(file_name, 'w') as file:
        json.dump(graph_data, file, indent=4, default=str)

def save_queue_to_file(queue, file_name):
    with open(file_name, 'w') as file:
        listed = [list(tup) for tup in queue]
        json.dump(listed, file, indent=4, default=str)

def method_keyword_check(string, method):
    string = string.lower()
    method = method.lower()
    return method in string

def conflict_keyword_check(string):
    conflict_keywords = ['conflict', 'war', 'violence', 'civil war', 'insurgency', 'terrorism']
    string = string.lower()
    return any(keyword in string for keyword in conflict_keywords)

def generate_method_network(start_paper_id, method, max_depth=3, res_name='method_papers.json'):
    seminal = {}
    
    graph = {}
    start_blob = try_api_call(create_url(start_paper_id))
    print(f"start_blob is {start_blob}")
    queue = [(start_paper_id, start_blob, 0)]
    seminal[start_paper_id] = {
        "url": start_blob["url"],
        "title": start_blob["title"],
        "year": start_blob["year"],
        "citationCount": start_blob["citationCount"]
    }
    
    visited = set()
    save_interval = 10
    processed_papers = 0

    while queue:
        current_paper_id, paper_info, current_depth = queue.pop(0)
        if not paper_info:
            url = create_url(current_paper_id)
            paper_info = try_api_call(url)
        if paper_info and current_paper_id not in visited and current_depth <= max_depth:
            print(f"Processing paper {current_paper_id} at depth {current_depth}")
            visited.add(current_paper_id)
            
            if paper_info['citationCount'] > 50 and method_keyword_check(paper_info['title'] + ' ' + paper_info['abstract'], method) and conflict_keyword_check(paper_info['title'] + ' ' + paper_info['abstract']):
                seminal[current_paper_id] = {
                    "url": paper_info["url"],
                    "title": paper_info["title"],
                    "year": paper_info["year"],
                    "citationCount": paper_info["citationCount"]
                }
                print(f"Seminal paper found: {paper_info['title']} with {paper_info['citationCount']} citations")
    
            children = paper_info.get('references', []) + paper_info.get('citations', [])
            if len(children) > 200:
                children = children[:200]
            
            print(f"------- the num of children are: {len(children)}")
            
            relevant_refs = []
            for child in children:
                child_id = child['paperId']
                if child_id:
                    url = create_url(child_id)
                    res = try_api_call(url)
                    
                    if not res:
                        queue.append((child_id, None, current_depth + 1))
                    elif res and res['abstract'] and method_keyword_check(res['title'] + ' ' + res['abstract'], method) and conflict_keyword_check(res['title'] + ' ' + res['abstract']):
                        relevant_refs.append(child_id) 
                        queue.append((child_id, res, current_depth + 1))
                    
            graph[current_paper_id] = (paper_info, relevant_refs, current_depth)
            print(f"------- the rel references are: {relevant_refs}")

            processed_papers += 1
            
            if processed_papers % save_interval == 0:
                save_dict_to_file(graph, res_name)
                save_queue_to_file(queue, 'method_queue.json')
                save_dict_to_file(seminal, 'seminal_method_papers.json')
                print(f"**** Saved backup of graph + queue at {processed_papers} papers.")
                
    return graph, seminal

# Example usage
start_paper_id = "a4f7c2c5e521257680f19ed1fea7285895327bea"  # Replace with a relevant starting paper ID
method = "random forest"  # Replace with the method you're interested in

graph, seminal_papers = generate_method_network(start_paper_id, method, max_depth=3, res_name='random_forest_papers.json')

print("Scraping complete. Results saved in 'random_forest_papers.json' and 'seminal_method_papers.json'")

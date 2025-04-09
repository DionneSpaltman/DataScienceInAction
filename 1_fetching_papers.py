# Fetching papers

# Import statement
import requests
import pandas as pd
import time
import json
import os

# Define where to save the results
folder_path = "/Users/dionnespaltman/Desktop/Luiss/Data Science in Action/Project/"
os.makedirs(folder_path, exist_ok=True)

# Open Alex URL
BASE_URL = "https://api.openalex.org/works"

# Define query 
query = """
(
  "artificial intelligence" OR AI OR "machine learning" OR ML OR "deep learning" OR DL OR 
  "natural language processing" OR NLP OR "predictive modeling" OR "data mining" OR 
  "data science" OR "neural networks" OR "transformer models" OR "language models" OR 
  "recommendation systems" OR "generative AI" OR "unsupervised learning" OR "supervised learning"
)
AND
(
  pricing OR promotion OR discount OR "price optimization" OR "dynamic pricing" OR 
  "sales prediction" OR "revenue management" OR "price elasticity" OR "demand forecasting" OR 
  marketing OR "campaign optimization" OR "consumer behavior" OR "targeting" OR "personalization"
)
AND
(
  retail OR supermarket OR "large-scale distribution" OR GDO OR "grocery stores" OR 
  "e-commerce" OR ecommerce OR "supply chain" OR "consumer goods" OR "FMCG" OR 
  "wholesale" OR "shopping behavior" OR "omnichannel" OR "online retail" OR "brick and mortar"
)
"""

# Set the parameters
params = {
    "search": query,
    "filter": "from_publication_date:2014-01-01,to_publication_date:2024-12-31",
    "per_page": 200,
    "cursor": "*"
}

# Results will be stored in a list 
all_results = []
max_pages = 5  

# Loop through the pages of the API response
for i in range(max_pages):
    print(f"Fetching page {i + 1}")

    # Make a GET request to the OpenAlex API with the current parameters
    response = requests.get(BASE_URL, params=params)

    # Check if the request was successful (HTTP 200 OK). If not, print the error and stop.
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        break

    # Parse the response content (JSON) into a Python dictionary
    # and append the results to the all_results list
    data = response.json()
    all_results.extend(data["results"])

    # Get the cursor for the next page of results
    next_cursor = data.get("meta", {}).get("next_cursor")

    # If there is no next cursor, we’ve reached the end — stop fetching
    if not next_cursor:
        break
    
    # Update the cursor parameter to fetch the next page in the next iteration
    params["cursor"] = next_cursor

    # Pause for 1 second to avoid being rate-limited by the API
    time.sleep(1) 

print(f"✅ Total papers fetched: {len(all_results)}")

# Function to reconstruct an abstract from OpenAlex's "abstract_inverted_index" format
def reconstruct_abstract(indexed):
    # First, check if the input is a dictionary (expected format). If not, return None.
    if not isinstance(indexed, dict):
        return None

    # Create an empty dictionary to store word positions and the corresponding words
    positions = {}

    # Loop through each word and its list of positions (indices) in the abstract
    for word, indices in indexed.items():
        for i in indices:
            # Map each index to its corresponding word
            positions[i] = word

    # Reconstruct the abstract by sorting the positions and joining the words in order
    return ' '.join(positions[i] for i in sorted(positions))


# All the columns we're not interested in
columns_to_delete = [
    "abstract_inverted_index", "abstract_inverted_index_v3", "fulltext_origin", 
    "is_authors_truncated", "mesh", "grants", "datasets", "versions", 
    "institution_assertions", "countries_distinct_count", "institutions_distinct_count", 
    "apc_list", "is_paratext"
]

# Add the abstract to each entry and remove unnecessary columns
for entry in all_results:
    entry["abstract"] = reconstruct_abstract(entry.get("abstract_inverted_index"))
    for col in columns_to_delete:
        entry.pop(col, None)

# Save results
df_clean = pd.DataFrame(all_results)
df_clean.to_csv(os.path.join(folder_path, "/Users/dionnespaltman/Downloads/openalex_results_clean.csv"), index=False)
print("✅ Clean CSV saved as openalex_results_clean.csv")

with open(os.path.join(folder_path, "/Users/dionnespaltman/Downloads/openalex_results_clean.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print("✅ Clean JSON saved as openalex_results_clean.json")

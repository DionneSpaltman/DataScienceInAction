from pyalex import Works, Authors, Source, Institutions, Topics
import pandas as pd
import requests
import time 
import json
from IPython import display

folder_path = '/Users/sanduandrei/Desktop/Luiss 1st Year/Data Science in Action/Project Scripts/Open Alex Results'
BASE_URL = "https://api.openalex.org/works"


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


params = {
    "search": query,
    "filter": "from_publication_date:2014-01-01,to_publication_date:2024-12-31",
    "per_page": 200,
    "cursor": "*"
}
all_results = []
max_pages = 5  # Safety limit: remove this if you want everything

for i in range(max_pages):
    print(f"Fetching page {i+1}")
    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        break

    data = response.json()
    all_results.extend(data["results"])

    next_cursor = data.get("meta", {}).get("next_cursor")
    if not next_cursor:
        break  # No more pages
    params["cursor"] = next_cursor
    time.sleep(1)  # Optional: Avoid rate-limiting (max 10 requests/sec)

print(f"Total papers fetched: {len(all_results)}")

# Check the first paper
first_paper = all_results[0]
print(json.dumps(first_paper, indent=2))

# Make sure you have at least one result
if all_results:
    first_paper = all_results[0]
    pub_year = first_paper.get("publication_year", "N/A")
    print(f"Publication year of the first paper: {pub_year}")
else:
    print("No papers found.")

file_path_json = "/Users/sanduandrei/Desktop/Luiss 1st Year/Data Science in Action/Project Scripts/Open Alex Results.json"
file_path_csv = "//Users/sanduandrei/Desktop/Luiss 1st Year/Data Science in Action/Project Scripts/Open Alex Results.csv"

# Save data in both CSV and JSON formats

# Convert to DataFrame for flat/tabular structure (good for CSV export)
df = pd.DataFrame(all_results)

# Save as CSV
# CSV is ideal for quick inspection, Excel use, or working with pandas.
# However, it will flatten the structure and drop nested fields like 'authorships'.
df.to_csv(file_path_csv, index=False)
print("Data saved to openalex_results.csv")

# Save as JSON
# JSON retains the full nested structure of each entry (e.g., authorship, abstract index, locations),
# which is essential for any downstream processing like recommendation systems or NLP tasks.
with open(file_path_json, "w") as f:
    json.dump(all_results, f, indent=2)
print("Data saved to openalex_results.json")


display(df.head())
display(df.info())
print(df.isnull().sum())

def reconstruct_abstract(indexed):
    if not isinstance(indexed, dict):
        return None
    positions = {}
    for word, indices in indexed.items():
        for i in indices:
            positions[i] = word
    return ' '.join(positions[i] for i in sorted(positions))

# Columns to remove
columns_to_delete = [
    "abstract_inverted_index",
    "abstract_inverted_index_v3",
    "fulltext_origin",
    "is_authors_truncated"
]

# Step 2: Clean each record
for entry in all_results:
    # Reconstruct abstract
    entry["abstract"] = reconstruct_abstract(entry.get("abstract_inverted_index"))
    
    for col in columns_to_delete:
        entry.pop(col, None)

print(df_clean.columns)

# CSV Display  
df_clean = pd.DataFrame(all_results)
display(df_clean)


with open(folder_path + "openalex_results_clean.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("Clean JSON saved as openalex_results_clean.json")

# Clean CSV
df_clean = pd.DataFrame(all_results)
df_clean.to_csv(folder_path + "openalex_results_clean.csv", index=False)
print("Clean CSV saved as openalex_results_clean.csv")

# CSV into DF
csv_path = os.path.join(folder_path, "openalex_results_clean.csv")
df_loaded = pd.read_csv(csv_path)
print("✅ CSV loaded. Shape:", df_loaded.shape)

# JSON to list of dict
json_path = os.path.join(folder_path, "openalex_results_clean.json")
with open(json_path, "r") as f:
    json_loaded = json.load(f)
print("✅ JSON loaded. Total entries:", len(json_loaded))
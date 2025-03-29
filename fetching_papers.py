from pyalex import Works, Authors, Source, Institutions, Topics
import pandas as pd
import requests
import time 

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
max_pages = 50  # Safety limit: remove this if you want everything

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

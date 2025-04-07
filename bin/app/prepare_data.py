import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("openalex_results_clean.json", 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df_clean = df[df['abstract'].notna()].copy()
docs = ("Title: " + df_clean['title'] + " Abstract: " + df_clean['abstract']).tolist()

embeddings = model.encode(docs, show_progress_bar=True)
df_clean['embedding'] = list(embeddings)

np.save("paper_embeddings.npy", np.vstack(embeddings))
df_clean.to_pickle("papers_df.pkl")

print("✅ Preprocessing done.")

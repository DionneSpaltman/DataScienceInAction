import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import json

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def load_data():
    with open("openalex_results_clean.json", 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df_clean = df[df['abstract'].notna()].copy()
    docs = ("Title: " + df_clean['title'] + " Abstract: " + df_clean['abstract']).tolist()
    embeddings = model.encode(docs, show_progress_bar=False)
    df_clean['embedding'] = list(embeddings)
    return df_clean, np.vstack(embeddings)

model = load_model()
papers_df, paper_embeddings = load_data()

st.title("🔍 Research Paper Recommender")
query = st.text_input("Enter your research query:", "Reinforcement learning for pricing in e-commerce")

top_k = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

if st.button("Recommend"):
    query_embedding = model.encode(query, convert_to_tensor=True).cpu()
    cosine_scores = util.pytorch_cos_sim(query_embedding, paper_embeddings)[0].numpy()
    top_indices = np.argsort(-cosine_scores)[:top_k]

    st.success(f"Top {top_k} recommended papers for: *{query}*")

    for idx in top_indices:
        paper = papers_df.iloc[idx]
        st.markdown(f"### {paper['title']}")
        st.markdown(f"**Score:** {cosine_scores[idx]:.4f}")
        st.markdown(f"**Topic:** {paper.get('topic_label', 'Unknown')}")
        with st.expander("Show abstract"):
            st.markdown(paper['abstract'])
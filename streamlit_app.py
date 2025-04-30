# To run this app, paste in your terminal: streamlit run streamlit_app.py

import streamlit as st  
import pandas as pd     
import numpy as np     
from sentence_transformers import SentenceTransformer, util
import json

# --- Load and cache the model ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- Load and embed data ---
@st.cache_data
def load_data():
    with open("/Users/dionnespaltman/Desktop/Luiss/Data Science in Action/Project/openalex_results_summaries_full.json", 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    df_clean = df[df['abstract'].notna()].copy()

    docs = ("Title: " + df_clean['title'] + " Abstract: " + df_clean['abstract']).tolist()
    embeddings = model.encode(docs, show_progress_bar=False)
    df_clean['embedding'] = list(embeddings)

    return df_clean, np.vstack(embeddings)

# --- Initialize model and data ---
model = load_model()
papers_df, paper_embeddings = load_data()

# --- UI ---
st.title("🔍 Research Paper Recommender")

query = st.text_input("Enter your research query:", "Reinforcement learning for pricing in e-commerce")
top_k = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

# --- Recommend button ---
if st.button("Recommend"):
    with st.spinner("Generating recommendations..."):
        query_embedding = model.encode(query, convert_to_tensor=True).cpu()
        cosine_scores = util.pytorch_cos_sim(query_embedding, paper_embeddings)[0].numpy()
        top_indices = np.argsort(-cosine_scores)

        st.session_state.results = {
            "scores": cosine_scores,
            "indices": top_indices
        }

# --- Display results if available ---
if "results" in st.session_state:
    scores = st.session_state.results["scores"]
    indices = st.session_state.results["indices"][:top_k]

    st.success(f"Top {top_k} recommended papers for: *{query}*")

    for idx in indices:
        paper = papers_df.iloc[idx]

        st.markdown(f"### {paper['title']}")
        st.markdown(f"**Score:** {scores[idx]:.4f}")
        st.markdown(f"**DOI:** {paper.get('doi', 'Not available')}")

        with st.expander("Show abstract"):
            st.markdown(paper.get('abstract', '_No abstract available._'))

        if paper.get("summary"):
            st.markdown(f"**📝 Summary:** {paper['summary']}")
        else:
            st.markdown("_No summary available._")

        st.markdown("---")


# To run this app, paste in your terminal: streamlit run new_app.py

# Import necessary libraries
import streamlit as st  
import pandas as pd     
import numpy as np     
from sentence_transformers import SentenceTransformer, util  # For semantic similarity
import json       

# Load and cache the sentence transformer model
@st.cache_resource
def load_model():
    # This loads a lightweight, general-purpose model for embedding sentences
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load data and compute embeddings
def load_data():
    # Load the dataset from the saved JSON file
    with open("openalex_results_clean.json", 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Filter out entries that have no abstract
    df_clean = df[df['abstract'].notna()].copy()

    # Create combined strings of title + abstract for embedding
    docs = ("Title: " + df_clean['title'] + " Abstract: " + df_clean['abstract']).tolist()

    # Compute embeddings for each paper
    embeddings = model.encode(docs, show_progress_bar=False)
    df_clean['embedding'] = list(embeddings)  # Store embeddings in the DataFrame

    return df_clean, np.vstack(embeddings)  # Return both the DataFrame and the embedding matrix

# Initialize model and data
model = load_model()
papers_df, paper_embeddings = load_data()

# Streamlit app user interface
st.title("🔍 Research Paper Recommender")

# Input field for user's query
query = st.text_input("Enter your research query:", "Reinforcement learning for pricing in e-commerce")

# Slider for number of recommendations to display
top_k = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

# When the "Recommend" button is clicked
if st.button("Recommend"):
    # Embed the user's query
    query_embedding = model.encode(query, convert_to_tensor=True).cpu()

    # Compute cosine similarity between query and each paper's embedding
    cosine_scores = util.pytorch_cos_sim(query_embedding, paper_embeddings)[0].numpy()

    # Get the indices of the top_k most similar papers
    top_indices = np.argsort(-cosine_scores)[:top_k]

    st.success(f"Top {top_k} recommended papers for: *{query}*")

    # Display the top recommended papers
    for idx in top_indices:
        paper = papers_df.iloc[idx]
        st.markdown(f"### {paper['title']}")  # Paper title
        st.markdown(f"**Score:** {cosine_scores[idx]:.4f}")  # Similarity score
        # st.markdown(f"**Topic:** {paper.get('topic_label', 'Unknown')}")  # Optional topic label

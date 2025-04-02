# Import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Path to my json file with the openalex results (1000 papers)
json_path = "/Users/sanduandrei/Desktop/Luiss 1st Year/Data Science in Action/Project/openalex_results_clean.json"

# Open and load the JSON data
with open(json_path, 'r') as f:
    data = json.load(f)

# Convert to DataFrame 
df = pd.DataFrame(data)
df_clean = df[df['abstract'].notna()].copy()
docs = ("Title: " + df_clean['title'] + " Abstract: " + df_clean['abstract']).tolist()

# Topic modeling
topic_model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")
topics, probs = topic_model.transform(docs)

df_clean['topic_id'] = topics
df_clean['topic_label'] = df_clean['topic_id'].apply(
    lambda x: topic_model.topic_labels_[x] if x != -1 and x < len(topic_model.topic_labels_) else "Unknown"
)

# Embedding
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(docs, show_progress_bar=True)
df_clean['embedding'] = list(embeddings)

# Finalize main paper DataFrame
papers_df = df_clean.copy()
paper_embeddings = np.vstack(papers_df['embedding'].values)

'''
def recommend_similar_papers_from_query(query_text, top_k=5, boost_topic=True):
    # Get query embedding 
    query_embedding = embedding_model.encode(query_text, convert_to_tensor=True).cpu()

    # Get the cosine scores 
    cosine_scores = util.pytorch_cos_sim(query_embedding, paper_embeddings)[0].cpu().numpy()

    # Get the predicted topic for the query
    if boost_topic:
        query_topic, _ = topic_model.transform([query_text])
        topic_boost_mask = (papers_df['topic_id'] == query_topic[0]).values.astype(float)
        cosine_scores += 0.05 * topic_boost_mask  # Boost same-topic papers slightly
    
    # Get the top results 
    top_results = np.argsort(-cosine_scores)[:top_k]

    # And print them 
    for idx in top_results:
        print(f"Title: {papers_df.iloc[idx]['title']}")
        print(f"Score: {cosine_scores[idx]:.4f}")
        print(f"Topic: {papers_df.iloc[idx]['topic_label']}")
        print(f"Abstract: {papers_df.iloc[idx]['abstract']}\n")

# Define your query
QUERY = "Reinforcement learning for automated pricing in e-commerce"
recommend_similar_papers_from_query(query_text=QUERY, top_k=5)

QUERY = "Effectiveness of personalized promotion strategies using customer data"
recommend_similar_papers_from_query(query_text=QUERY, top_k=5)


QUERY = "Natural language processing for sentiment-driven pricing strategies"
recommend_similar_papers_from_query(query_text=QUERY, top_k=5)
'''



'''
def explain_recommendations(query_text, top_k=5, boost_topic=True):
    print(f"\n🔍 Query: {query_text}\n")

    # 1. Encode the query
    query_embedding = embedding_model.encode(query_text, convert_to_tensor=True).cpu()

    # 2. Compute raw cosine similarity
    cosine_sim_raw = util.pytorch_cos_sim(query_embedding, paper_embeddings)[0].numpy()

    # 3. Compute topic boost (if enabled)
    if boost_topic:
        query_topic, _ = topic_model.transform([query_text])
        topic_boost_mask = (papers_df['topic_id'] == query_topic[0]).values.astype(float)
        topic_boost = 0.05 * topic_boost_mask
    else:
        topic_boost = np.zeros_like(cosine_sim_raw)

    # 4. Final score = similarity + boost
    final_scores = cosine_sim_raw + topic_boost

    # 5. Get top results
    top_indices = np.argsort(-final_scores)[:top_k]

    # 6. Print detailed breakdown
    for idx in top_indices:
        title = papers_df.iloc[idx]['title']
        topic = papers_df.iloc[idx]['topic_label']
        abstract = papers_df.iloc[idx]['abstract'][:300] + "..."  # trim abstract for readability
        boost_applied = topic_boost[idx] > 0

        print(f"📄 Title: {title}")
        print(f"    - Topic: {topic}")
        print(f"    - Raw Cosine Similarity: {cosine_sim_raw[idx]:.4f}")
        print(f"    - Topic Boost Applied: {'Yes (+0.05)' if boost_applied else 'No'}")
        print(f"    - Final Score: {final_scores[idx]:.4f}")
        print(f"    - Abstract: {abstract}\n")

explain_recommendations("Reinforcement learning for automated pricing in e-commerce", top_k=5)
'''


def debug_cosine_similarity(query_text, paper_idx):
    query_embedding = embedding_model.encode(query_text, convert_to_tensor=True).cpu().numpy()
    paper_vector = paper_embeddings[paper_idx]

    # Compute components
    dot_product = np.dot(query_embedding, paper_vector)
    norm_query = np.linalg.norm(query_embedding)
    norm_paper = np.linalg.norm(paper_vector)
    cosine_sim = dot_product / (norm_query * norm_paper)

    print(f"\n📄 Comparing to Paper #{paper_idx}: '{papers_df.iloc[paper_idx]['title']}'")
    print(f"🔹 Cosine Similarity: {cosine_sim:.4f}")
    print(f"🔹 Dot Product: {dot_product:.4f}")
    print(f"🔹 Query Norm: {norm_query:.4f}")
    print(f"🔹 Paper Norm: {norm_paper:.4f}")

    # Show contribution of first 10 dimensions
    contributions = query_embedding * paper_vector
    print("\nTop 10 dimensions contributing to the dot product:")
    for i in range(10):
        print(f"Dim {i:3d}: Q={query_embedding[i]:+.4f}, P={paper_vector[i]:+.4f}, Q*P={contributions[i]:+.4f}")

    # Optionally: visualize all 384 contributions
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.plot(contributions)
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f"Per-Dimension Contributions to Dot Product (Paper #{paper_idx})")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Q[i] * P[i]")
    plt.show()

query = "Reinforcement learning for pricing"
top_paper_idx = np.argsort(-util.pytorch_cos_sim(
    embedding_model.encode(query, convert_to_tensor=True).cpu(), paper_embeddings
)[0].numpy())[0]

debug_cosine_similarity(query, top_paper_idx)


def abstract_word_importance_top_k(paper_idx, query_text, top_k=20):
    abstract = papers_df.iloc[paper_idx]['abstract']
    title = papers_df.iloc[paper_idx]['title']
    base_input = f"Title: {title} Abstract: {abstract}"

    base_score = util.pytorch_cos_sim(
        embedding_model.encode(query_text, convert_to_tensor=True).cpu(),
        embedding_model.encode(base_input, convert_to_tensor=True).cpu().reshape(1, -1)
    )[0].item()

    words = abstract.split()
    drops = []

    for i, word in enumerate(words):
        new_abstract = " ".join(words[:i] + words[i+1:])
        new_input = f"Title: {title} Abstract: {new_abstract}"
        new_embedding = embedding_model.encode(new_input, convert_to_tensor=True).cpu()
        new_score = util.pytorch_cos_sim(
            embedding_model.encode(query_text, convert_to_tensor=True).cpu(),
            new_embedding.reshape(1, -1)
        )[0].item()

        drop = base_score - new_score  # How much the score dropped when removing this word
        drops.append((word, drop))

    # Sort by importance: biggest drop = most important
    drops.sort(key=lambda x: -x[1])

    print(f"\nTop {top_k} most influential words in abstract for paper: {title}")
    print(f"Query: {query_text}")
    print(f"Base similarity score: {base_score:.4f}\n")

    for word, drop in drops[:top_k]:
        print(f"{word:>15s}: ↓ {drop:.4f}")


query = "Reinforcement learning for dynamic pricing in e-commerce"
paper_idx = 369  # or one of your top papers
abstract_word_importance_top_k(paper_idx, query, top_k=20)


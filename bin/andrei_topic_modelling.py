import pandas as pd 
from pyalex import Works, Authors, Sources, Institutions, Topics
import requests
import time 
import json
from IPython import display
import ast
from sentence_transformers import SentenceTransformer
import numpy as np


csv_file = pd.read_csv('/Users/sanduandrei/Desktop/Luiss 1st Year/Data Science in Action/Project Scripts/Open Alex Results.csv')
print(f'Column Names of CSV_file: {csv_file.columns.tolist()}')


# Removing irrelevant columns from CSV file 
csv_file_clean = csv_file.drop(columns=['id', 'display_name', 'ids', 'language', 'primary_location',
                                  'indexed_in', 'open_access', 'institution_assertions', 'countries_distinct_count',
                                  'institutions_distinct_count', 'corresponding_author_ids', 'corresponding_institution_ids', 
                                  'apc_list', 'apc_paid', 'fwci', 'has_fulltext', 'citation_normalized_percentile', 'cited_by_percentile_year', 'biblio', 'primary_topic', 
                                  'topics', 'mesh', 'locations_count', 'locations', 'best_oa_location', 'sustainable_development_goals', 
                                  'grants', 'datasets', 'versions', 'referenced_works', 'counts_by_year', 'updated_date', 
                                  'created_date', 'fulltext_origin', 'abstract_inverted_index_v3'])


# ---------------
# Reconstructing abstracts
#----------------

def reconstruct_abstract(inverted_index_str):
    try:
        index_dict = ast.literal_eval(inverted_index_str)
        word_list = [None] * (max(i for pos in index_dict.values() for i in pos) + 1)
        for word, positions in index_dict.items():
            for pos in positions:
                word_list[pos] = word
        return ' '.join(word_list)
    except Exception as e:
        return ''
    
csv_file_clean['abstract'] = csv_file_clean['abstract_inverted_index'].apply(reconstruct_abstract)

# Save new CSV file in project folder
csv_file_clean.to_csv('/Users/sanduandrei/Desktop/Luiss 1st Year/Data Science in Action/Project Scripts/Open Alex Results/cleaned_openalex.csv', index=False)
print(f'Column Names of clean CSV file: {csv_file_clean.columns.tolist()}')

# ---------------
# BERT Embeddings
# ---------------

model = SentenceTransformer('all-MiniLM-L6-v2')
abstracts = csv_file_clean['abstract'].tolist()
embeddings = model.encode(abstracts, show_progress_bar=True)

emb_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
csv_with_embeddings = pd.concat([csv_file_clean.reset_index(drop=True), emb_df], axis=1)
csv_with_embeddings.to_csv('/Users/sanduandrei/Desktop/Luiss 1st Year/Data Science in Action/Project Scripts/Open Alex Results/with_bert_embeddings.csv', index=False)

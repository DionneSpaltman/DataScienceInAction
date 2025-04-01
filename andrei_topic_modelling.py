import pandas as pd 
from pyalex import Works, Authors, Sources, Institutions, Topics
import requests
import time 
import json
from IPython import display


csv_file = pd.read_csv('/Users/sanduandrei/Desktop/Luiss 1st Year/Data Science in Action/Project Scripts/Open Alex Results.csv')
print(f'Column Names of CSV_file: {csv_file.columns.tolist()}')


# Removing irrelevant columns from CSV file 
csv_file_clean = csv_file.drop(columns=['id', 'display_name', 'ids', 'language', 'primary_location',
                                  'indexed_in', 'open_access', 'institution_assertions', 'countries_distinct_count',
                                  'institutions_distinct_count', 'corresponding_author_ids', 'corresponding_institution_ids', 
                                  'apc_list', 'apc_paid', 'fwci', 'has_fulltext', 'citation_normalized_percentile', 'cited_by_percentile_year', 'biblio', 'primary_topic', 
                                  'topics', 'mesh', 'locations_count', 'locations', 'best_oa_location', 'sustainable_development_goals', 
                                  'grants', 'datasets', 'versions', 'referenced_works', 'counts_by_year', 'updated_date', 
                                  'created_date', 'fulltext_origin'])

# Save new CSV file in project folder
csv_file_clean.to_csv('/Users/sanduandrei/Desktop/Luiss 1st Year/Data Science in Action/Project Scripts/Open Alex Results/cleaned_openalex.csv', index=False)
print(f'Column Names of clean CSV file: {csv_file_clean.columns.tolist()}')
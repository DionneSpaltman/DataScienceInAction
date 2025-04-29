import json
import os

# --- Path to the JSON file with summaries ---
summary_path = "/Users/dionnespaltman/Desktop/Luiss /Data Science in Action/Project/openalex_results_test_summaries.json"

# --- Load the summarized entries ---
with open(summary_path, 'r') as f:
    summarized_data = json.load(f)

# --- Print abstracts and summaries ---
for i, entry in enumerate(summarized_data, 1):
    print(f"\n📄 Paper {i}")
    print("-" * 60)
    print(f"🔍 Abstract:\n{entry.get('abstract', 'No abstract available.')}\n")
    print(f"📝 Summary:\n{entry.get('summary', 'No summary available.')}")
    print("=" * 60)

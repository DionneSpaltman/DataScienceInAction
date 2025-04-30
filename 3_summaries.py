import os
import json
from tqdm import tqdm
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch

# Paths
json_path = os.path.join("/Users/dionnespaltman/Desktop/Luiss/Data Science in Action/Project/openalex_results_clean.json")
output_path = os.path.join("/Users/dionnespaltman/Desktop/Luiss/Data Science in Action/Project/openalex_results_summaries_full.json")

# Load JSON 
with open(json_path, 'r') as f:
    data = json.load(f)

# Filter to entries with abstracts and no existing summary
full_batch = [entry for entry in data if entry.get("abstract")]

# test_batch = [entry for entry in data if entry.get("abstract")]
# test_batch = test_batch[:5]  # ✅ Only take first 5 entries for testing

# Load Pegasus model 
# model_name = "google/pegasus-xsum"
model_name = "google/pegasus-cnn_dailymail"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Summary 
def generate_summary(text, max_length=100):
    prompt = (
        "Summarize this abstract focusing on the research question, method, and key results:\n\n"
        + text
    )
    inputs = tokenizer(prompt, truncation=True, padding="longest", return_tensors="pt").to(device)
    summary_ids = model.generate(inputs["input_ids"], max_length=max_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary.replace("<n>", " ")  # Clean up any newline artifacts


# Run summarization on full batch
for entry in tqdm(full_batch, desc="Generating summaries"):
    try:
        entry["summary"] = generate_summary(entry["abstract"])
    except Exception as e:
        print(f"Error summarizing abstract: {e}")
        entry["summary"] = None

# --- Save results ---
with open(output_path, 'w') as f:
    json.dump(full_batch, f, indent=2)

print(f"✅ Full dataset summaries saved to: {output_path}")


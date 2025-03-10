import json
import os
import torch
from transformers import BertTokenizer, BertForMaskedLM
import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_lg")

# Paths
SCENE_GRAPH_FILE = "scene_graph_chunks/scene_graphs_1.json"
OUTPUT_DIR = "query_type_2"
QUERY_OUTPUT_FILE = "hallucination_queries.json"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the scene graph JSON
with open(SCENE_GRAPH_FILE, "r") as f:
    scene_graphs = json.load(f)

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()

# Function to extract verb and preposition
def extract_verb_prep(predicate):
    doc = nlp(predicate.replace("_", " "))
    verb = None
    for token in doc:
        if token.pos_ == "VERB":
            verb = token.text
            break
    return verb

# Function to get alternative verbs using BERT
def get_alternative_verbs(subject, correct_verb, object_, top_k=5):
    masked_sentence = f"The {subject} [MASK] the {object_}."  # Removed "is" and preposition
    inputs = tokenizer(masked_sentence, return_tensors="pt")
    mask_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = outputs.logits[0, mask_index, :].topk(top_k)
    predicted_tokens = [tokenizer.decode([idx]) for idx in predictions.indices[0]]
    
    # Filter out non-verbs and the original correct verb
    filtered_verbs = []
    for token in predicted_tokens:
        doc = nlp(token)
        if len(doc) == 1 and doc[0].pos_ == "VERB" and token != correct_verb:
            filtered_verbs.append(token)
    
    return filtered_verbs

# Generate hallucination queries
structured_queries = []
chunk_size = 20

for i in range(0, len(scene_graphs), chunk_size):
    chunk = scene_graphs[i:i + chunk_size]
    output_file = os.path.join(OUTPUT_DIR, f"scene_graphs_{i//chunk_size + 1}.json")
    
    with open(output_file, "w") as f:
        json.dump(chunk, f, indent=2)
    
    print(f"Saved {len(chunk)} scene graphs to {output_file}")
    
    for graph in chunk:
        image_queries = []
        for relation in graph.get("relationships", []):
            subject_obj = next((obj for obj in graph["objects"] if obj["object_id"] == relation["subject_id"]), None)
            object_obj = next((obj for obj in graph["objects"] if obj["object_id"] == relation["object_id"]), None)
            
            if not subject_obj or not object_obj:
                continue
            
            subject = subject_obj["names"][0]
            object_ = object_obj["names"][0]
            predicate = relation["predicate"]
            
            verb = extract_verb_prep(predicate)
            if not verb:
                continue  # Skip if no verb found
            
            alt_verbs = get_alternative_verbs(subject, verb, object_)
            for alt_verb in alt_verbs:
                query = {
                    "question": f"Is the {subject} {alt_verb} the {object_}?",
                    "answer": "no"
                }
                image_queries.append(query)
        
        if image_queries:
            structured_queries.append({
                "image_id": graph["image_id"],
                "queries": image_queries
            })

# Save hallucination queries in the new structure
with open(QUERY_OUTPUT_FILE, "w") as f:
    json.dump(structured_queries, f, indent=2)

print(f"Generated queries for {len(structured_queries)} images in {QUERY_OUTPUT_FILE}")

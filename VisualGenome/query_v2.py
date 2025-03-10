import json
import random
import spacy

# Load spaCy's English model
nlp = spacy.load("en_core_web_md")

def get_pos_tag(word):
    """Use spaCy to get the POS tag of a word."""
    doc = nlp(word)
    return doc[0].pos_  # Return POS tag of the first token

# Load the scene graph JSON file
with open("scene_graph_chunks/scene_graphs_1.json", "r") as f:
    data_list = json.load(f)

# Store queries grouped by image_id
structured_queries = []

for data in data_list:    
    image_id = data["image_id"]  # Extract the image ID
    
    # Build a mapping of object_id to object name
    object_map = {obj["object_id"]: obj["names"][0] if obj["names"] else f"object_{obj['object_id']}" for obj in data["objects"]}

    # Select 3 random relationships
    relationships = data["relationships"]
    if len(relationships) > 3:
        relationships = random.sample(relationships, 3)  # Pick random 3

    queries = []
    for relation in relationships:
        subject_id = relation["subject_id"]
        object_id = relation["object_id"]
        predicate = relation["predicate"].lower()

        # Lookup names from the mapping
        subject_name = object_map.get(subject_id, f"object_{subject_id}")
        object_name = object_map.get(object_id, f"object_{object_id}")

        # Determine POS type dynamically
        pos_tag = get_pos_tag(predicate)

        if pos_tag == "ADP":  # If preposition (like "on", "in", "has")
            question = f"Where is the {subject_name} in relation to the {object_name}?"
        elif pos_tag == "VERB":  # If verb (like "touching", "holding")
            question = f"What is the {subject_name} doing to the {object_name}?"
        else:
            question = f"What one word describes the relationship between {subject_name} and {object_name}?"

        queries.append({
            "question": question,
            "expected_answer": predicate,
            "subject_id": subject_id,
            "object_id": object_id,
            "relationship_id": relation["relationship_id"]
        })

    # Append to the structured output
    structured_queries.append({
        "image_id": image_id,
        "queries": queries
    })

# Save the structured queries to a JSON file
with open("queries_v2.json", "w") as f:
    json.dump(structured_queries, f, indent=4)

print("Queries saved to queries_v2.json")

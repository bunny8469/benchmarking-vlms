import json
import random

# Load scene graph data
with open("scene_graph_chunks/scene_graphs_1.json", "r") as f:
    data_list = json.load(f)

all_queries = []

for data in data_list:
    image_id = data["image_id"]

    # Build a mapping of object_id to object name
    object_map = {obj["object_id"]: obj["names"][0] if obj["names"] else f"object_{obj['object_id']}" for obj in data["objects"]}

    # Select 3 random relationships
    relationships = random.sample(data["relationships"], min(3, len(data["relationships"])))

    queries = []
    for relation in relationships:
        subject_id = relation["subject_id"]
        object_id = relation["object_id"]

        subject_name = object_map.get(subject_id, f"object_{subject_id}")
        object_name = object_map.get(object_id, f"object_{object_id}")

        # Create a query with a blank for the relationship
        question = f"Fill the blank with one verb or one preposition: {subject_name} _____ {object_name}. Answer only one word strictly"

        queries.append({
            "question": question,
            "expected_answer": relation["predicate"],  # This is the answer to be filled
            "subject_id": subject_id,
            "object_id": object_id,
            "relationship_id": relation["relationship_id"]
        })

    all_queries.append({"image_id": image_id, "queries": queries})

# Save queries to a file
with open("queries_blank.json", "w") as f:
    json.dump(all_queries, f, indent=4)

print("Queries saved to queries_blank.json!")

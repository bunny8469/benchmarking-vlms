import json
import os
from collections import defaultdict

# Define specific attribute vocabularies
known_colors = {
    "red", "blue", "green", "white", "black", "yellow", "brown",
    "orange", "pink", "purple", "gray", "grey"
}

known_shapes = {
    "round", "square", "triangular", "circular", "rectangular"
}

known_materials = {
    "wooden", "metal", "plastic", "glass", "leather", "fabric"
}

known_conditions = {
    "broken", "dirty", "clean", "wet", "dry", "old", "new"
}

wearable_parts = {
    "shirt", "t-shirt", "jacket", "hat", "pants", "shoes", "scarf",
    "dress", "jeans", "sneakers", "boots"
}

def get_position_descriptor(obj, objects):
    obj_type = obj['names'][0]
    same_type = [o for o in objects if o['names'][0] == obj_type]
    if len(same_type) <= 1:
        return ""
    x_center = obj['x'] + obj['w'] / 2
    sorted_centers = sorted([o['x'] + o['w'] / 2 for o in same_type])
    rank = sorted_centers.index(x_center)
    if rank == 0:
        return " on the left"
    elif rank == len(sorted_centers) - 1:
        return " on the right"
    else:
        return " in the center"

def classify_attribute(attr):
    attr = attr.lower()
    if attr in known_colors:
        return "color"
    if attr in known_shapes:
        return "shape"
    if attr in known_materials:
        return "material"
    if attr in known_conditions:
        return "condition"
    return None

def attribute_query(obj_name, attr, attr_type, position_desc):
    if attr_type == "color":
        return f"What color is the {obj_name}{position_desc}?"
    elif attr_type == "shape":
        return f"What is the shape of the {obj_name}{position_desc}?"
    elif attr_type == "material":
        return f"What material is the {obj_name}{position_desc} made of?"
    elif attr_type == "condition":
        return f"What is the condition of the {obj_name}{position_desc}?"
    return None

def generate_attribute_hallucination_queries(scene_graphs_path, image_ids, output_path):
    with open(scene_graphs_path, 'r') as f:
        scene_graphs = json.load(f)

    queries_by_type = defaultdict(list)

    for graph in scene_graphs:
        if graph['image_id'] not in image_ids:
            continue

        objects = graph['objects']
        for obj in objects:
            if 'names' not in obj or not obj['names']:
                continue

            obj_name = obj['names'][0].lower()
            attributes = obj.get('attributes', [])
            if not attributes:
                continue

            position_desc = get_position_descriptor(obj, objects)

            for attr in attributes:
                attr_type = classify_attribute(attr)
                if not attr_type:
                    continue

                query = attribute_query(obj_name, attr, attr_type, position_desc)
                if query:
                    queries_by_type[attr_type].append({
                        "image_id": graph['image_id'],
                        "object": obj_name,
                        "attribute": attr,
                        "query": query,
                        "answer": attr
                    })

    with open(output_path, 'w') as f:
        json.dump(queries_by_type, f, indent=2)
    print(f"Saved {sum(len(v) for v in queries_by_type.values())} queries with answers to {output_path}")

# Example usage
image_ids = [1, 3, 7, 8, 10, 18, 19, 20, 21, 22, 23, 24, 25]
scene_graphs_path = "./VisualGenome/scene_graphs.json"
output_path = "attribute_hallucination_queries_with_answers.json"

generate_attribute_hallucination_queries(scene_graphs_path, image_ids, output_path)

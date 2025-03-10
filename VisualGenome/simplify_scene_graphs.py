import json

def simplify_scene_graphs():
    # Load the scene graph JSON
    with open("scene_graph_chunks/scene_graphs_1.json", "r") as f:
        scene_graphs = json.load(f)

    simplified_graphs = []
    
    for graph in scene_graphs:
        simplified_relations = {}
        
        for relation in graph.get("relationships", []):
            # Get subject and object details
            subject_obj = next((obj for obj in graph["objects"] 
                              if obj["object_id"] == relation["subject_id"]), None)
            object_obj = next((obj for obj in graph["objects"]
                             if obj["object_id"] == relation["object_id"]), None)
            
            if not (subject_obj and object_obj):
                continue
                
            # Create simplified relation
            subject = subject_obj["names"][0]
            if subject not in simplified_relations:
                simplified_relations[subject] = {
                    "action": relation["predicate"],
                    "object": object_obj["names"][0]
                }
        
        if simplified_relations:
            simplified_graphs.append({
                "image_id": graph["image_id"],
                "relations": simplified_relations
            })

    # Save simplified format
    with open("simplified_scene_graphs.json", "w") as f:
        json.dump(simplified_graphs, f, indent=2)
        
    print(f"Processed {len(simplified_graphs)} images into simplified format")

if __name__ == "__main__":
    simplify_scene_graphs()
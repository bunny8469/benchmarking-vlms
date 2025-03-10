from sentence_transformers import SentenceTransformer, util

# Load the SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight & efficient

def compute_similarity(word1, word2):
    """Compute semantic similarity using SBERT embeddings."""
    emb1, emb2 = model.encode([word1, word2])
    return util.pytorch_cos_sim(emb1, emb2).item()

def evaluate_prediction(scene_graph, predicted_word, actual_word, threshold=0.5):
    """Determine if the model's prediction is a misclassification or hallucination."""
    
    # Compute semantic similarity
    similarity = compute_similarity(predicted_word, actual_word)
    
    # Check if the actual word exists in the scene graph
    objects = set(scene_graph["relations"].keys())
    relations = {v["object"] for v in scene_graph["relations"].values()}
    actions = {v["action"] for v in scene_graph["relations"].values()}
    
    is_in_scene = actual_word in objects or actual_word in relations or actual_word in actions
    print("Is in scene? ", is_in_scene)
    
    print(f"Similarity({predicted_word}, {actual_word}) = {similarity:.4f}")

    if is_in_scene and similarity < threshold:
        return "Misclassification"
    elif not is_in_scene:
        return "Hallucination"
    else:
        return "Correct Prediction"

# Example Usage
scene_graph = {
    "image_id": 1, 
    "relations": { 
        "man": { "action": "running", "object": "sneakers" }
    }
}

print(evaluate_prediction(scene_graph, "running", "sprinting"))  # Should be "Correct Prediction"
print(evaluate_prediction(scene_graph, "jumping", "running"))  # Likely "Misclassification"
print(evaluate_prediction(scene_graph, "walking", "running"))  # Likely "Hallucination"

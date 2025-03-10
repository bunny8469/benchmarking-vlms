import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_lg")

def compute_similarity(word1, word2):
    """Compute semantic similarity between two words."""
    vec1, vec2 = nlp(word1).vector, nlp(word2).vector
    if np.any(vec1) and np.any(vec2):  # Ensure vectors exist
        return cosine_similarity([vec1], [vec2])[0][0]
    return 0  # Return zero similarity if either vector is empty

def evaluate_prediction(scene_graph, predicted_word, actual_word, threshold=0.7):
    """Determine if the model's prediction is a misclassification or hallucination."""
    
    # Compute semantic similarity
    similarity = compute_similarity(predicted_word, actual_word)
    
    # Check if the predicted word exists in the scene graph
    objects = set(scene_graph["relations"].keys())
    relations = {v["object"] for v in scene_graph["relations"].values()}
    actions = {v["action"] for v in scene_graph["relations"].values()}
    
    is_in_scene = actual_word in objects or actual_word in relations or actual_word in actions
    
    print("Similarity", similarity)
    if is_in_scene and similarity < threshold:
        return "Misclassification"
    elif not is_in_scene:
        return "Hallucination"
    else:
        return "Correct Prediction"

# Example Usage
scene_graph = { "image_id": 1, "relations": { "man": { "action": "running", "object": "sneakers" }}} 
print(evaluate_prediction(scene_graph, "sprinting", "running"))  # Example test

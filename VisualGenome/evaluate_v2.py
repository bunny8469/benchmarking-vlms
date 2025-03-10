import json
from sentence_transformers import SentenceTransformer, util
import string

# Load the SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight & efficient

correct_count = 0
hallucination_count = 0

def compute_similarity(word1, word2):
    """Compute semantic similarity using SBERT embeddings."""
    emb1, emb2 = model.encode([word1, word2])
    return util.pytorch_cos_sim(emb1, emb2).item()

def evaluate_prediction(scene_graph, predicted_word, actual_word, threshold=0.5):
    """Determine if the model's prediction is a misclassification or hallucination."""
    
    def remove_punctuation(text):
        return text.translate(str.maketrans("", "", string.punctuation))

    predicted_word = remove_punctuation(predicted_word).lower()
    actual_word = actual_word.lower()

    # Compute semantic similarity
    similarity = compute_similarity(predicted_word, actual_word)
    
    # Check if the actual word exists in the scene graph
    objects = set(scene_graph["relations"].keys())
    relations = {v["object"] for v in scene_graph["relations"].values()}
    actions = {v["action"] for v in scene_graph["relations"].values()}
    
    is_in_scene = actual_word in objects or actual_word in relations or actual_word in actions
    print(f"Is in scene? {is_in_scene}")

    print(f"Similarity({predicted_word}, {actual_word}) = {similarity:.4f}")

    global correct_count, hallucination_count
    if similarity < threshold:
        hallucination_count += 1
        return "Hallucination"
    else:
        correct_count += 1
        return "Correct Prediction"

# Load JSON file with questions and answers
with open("model_responses.json", "r") as f:
    question_data = json.load(f)

# Example scene graph (modify based on actual data structure)
scene_graphs = {
    1: {  # Example scene graph for image_id = 7
        "relations": {
            "woman": {"action": "wear", "object": "sweater"},
            "woman2": {"action": "have", "object": "blue jeans"}
        }
    }
}

# Evaluate predictions from JSON
for entry in question_data:
    image_id = entry["image_id"]
    predicted_answer = entry["model_answer"].lower().strip(".")
    actual_answer = entry["expected_answer"].lower().strip()
    
    scene_graph = scene_graphs.get(image_id, {"relations": {}})  # Default empty scene graph
    evaluation = evaluate_prediction(scene_graph, predicted_answer, actual_answer)
    
    print(f"Question: {entry['question']}")
    print(f"Expected: {actual_answer}, Model: {predicted_answer}")
    print(f"Evaluation: {evaluation}\n")

print("\nEvaluation Summary:")
print(f"✅ Correct Predictions: {correct_count}")
print(f"❌ Hallucinations: {hallucination_count}")

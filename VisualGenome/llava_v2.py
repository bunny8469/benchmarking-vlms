import os
import json
import base64
from collections import Counter
from groq import Groq
from sentence_transformers import SentenceTransformer, util
import string

# Initialize Groq API Client
os.environ["GROQ_API_KEY"] = "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"  # Replace with your API key
client = Groq()
model = "llama-3.2-90b-vision-preview"

# Load sentence transformer model for similarity computation
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight & efficient

# Function to compute similarity
def compute_similarity(word1, word2):
    """Compute semantic similarity using SBERT embeddings."""
    emb1, emb2 = sbert_model.encode([word1, word2])
    return util.pytorch_cos_sim(emb1, emb2).item()

def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

# Function to evaluate predictions
def evaluate_prediction(scene_graph, predicted_word, actual_word, threshold=0.4):
    """Determine if the model's prediction is a misclassification or hallucination."""

    predicted_word = remove_punctuation(predicted_word)

    similarity = compute_similarity(predicted_word, actual_word)
    
    # Check if the actual word exists in the scene graph
    objects = set(scene_graph["relations"].keys())
    relations = {v["object"] for v in scene_graph["relations"].values()}
    actions = {v["action"] for v in scene_graph["relations"].values()}
    
    is_in_scene = actual_word in objects or actual_word in relations or actual_word in actions
    
    print(f"Similarity({predicted_word}, {actual_word}) = {similarity:.4f}")
    print(f"Is '{actual_word}' in scene? {is_in_scene}")

    # if is_in_scene and similarity < threshold:
    #     return "Misclassification"
    # elif not is_in_scene:
    #     return "Hallucination"
    # else:
    #     return "Correct Prediction"
    if similarity < threshold:
        return "Hallucination"
    else:
        return "Correct Prediction"

# Load dataset
with open("queries_v2.json", "r") as f:
    dataset = json.load(f)

# Function to convert image to Base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Define image directory
image_dir = "images/"  # Change this to your actual image folder

# System prompt for hallucination testing
system_prompt = {
    "role": "system",
    "content": "You are an AI assistant designed for factual visual reasoning. Answer only based on the given image. If uncertain, respond with 'I am unsure based on the image provided.' Avoid making assumptions."
}

# Process each entry in dataset
results = []
eval_counts = Counter({"Correct Prediction": 0, "Misclassification": 0, "Hallucination": 0})

for entry in dataset:
    image_id = entry["image_id"]
    image_path = os.path.join(image_dir, f"{image_id}.jpg")  # Assuming images are named as <image_id>.jpg
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_path} not found. Skipping...")
        continue

    # Convert image to Base64
    base64_image = encode_image_to_base64(image_path)

    # Process each query for the image
    for query in entry["queries"]:
        user_prompt = query["question"]
        expected_answer = query["expected_answer"]

        # Send request to Groq API
        response = client.chat.completions.create(
            messages=[
                # system_prompt, 
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
            model=model
        )

        # Extract model response
        model_response = response.choices[0].message.content.strip()

        # Print Q/A comparison
        print(f"\n📌 **Image ID**: {image_id}")
        print(f"**Question**: {user_prompt}")
        print(f"✅ **Expected Answer**: {expected_answer}")
        print(f"🔎 **Model Answer**: {model_response}\n")

        # Evaluate prediction
        scene_graph = {"image_id": image_id, "relations": {}}  # Placeholder for scene graph, replace as needed
        evaluation = evaluate_prediction(scene_graph, model_response.lower(), expected_answer.lower())
        
        eval_counts[evaluation] += 1  # Track evaluation stats

        # Store results
        results.append({
            "image_id": image_id,
            "question": user_prompt,
            "expected_answer": expected_answer,
            "model_answer": model_response,
            "evaluation": evaluation
        })

# Save results to JSON
with open("model_responses.json", "w") as f:
    json.dump(results, f, indent=4)

# Print summary statistics
total_queries = sum(eval_counts.values())
if total_queries > 0:
    print("\n📊 **Final Evaluation Summary**:")
    for category, count in eval_counts.items():
        percentage = (count / total_queries) * 100
        print(f"🔹 {category}: {count} ({percentage:.2f}%)")

print("\n✅ Processing complete! Results saved to model_responses.json")

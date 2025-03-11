import os
import json
import base64
from groq import Groq
import nltk
nltk.download('punkt_tab')
from collections import Counter
from sentence_transformers import SentenceTransformer, util

# Set the correct API key environment variable
os.environ["GROQ_API_KEY"] = "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"  # Replace with your Groq API key

# Initialize Groq client
client = Groq()
model = "llama-3.2-11b-vision-preview"
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")  # Sentence embedding model

# Load dataset
with open("hallucination_queries_llama_updated.json", "r") as f:
    dataset = json.load(f)

# Function to convert image to Base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to calculate F1 Score
def f1_score(prediction, ground_truth):
    pred_tokens = nltk.word_tokenize(prediction.lower())
    truth_tokens = nltk.word_tokenize(ground_truth.lower())
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1

# Function to calculate cosine similarity
def cosine_similarity(prediction, ground_truth):
    pred_embedding = sbert_model.encode(prediction, convert_to_tensor=True)
    truth_embedding = sbert_model.encode(ground_truth, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(pred_embedding, truth_embedding).item()
    return similarity

# Define image directory
image_dir = "images/"  # Change this to your actual image folder

# Process each entry in dataset
results = []
for image_id, objects in dataset.items():
    image_path = os.path.join(image_dir, f"{image_id}.jpg")  # Assuming images are named as <image_id>.jpg
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_path} not found. Skipping...")
        continue

    # Convert image to Base64
    base64_image = encode_image_to_base64(image_path)

    for object_name, queries in objects.items():
        for query_type, query_info in queries.items():
            question = query_info["question"]
            expected_answer = query_info["answer"]
            
            # Send request to Groq API
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            {"type": "text", "text": question},
                        ],
                    }
                ],
                model=model
            )
            
            # Extract model response
            model_response = response.choices[0].message.content
            print(f"Image {image_id} | Object: {object_name} | Query Type: {query_type}\nQ: {question}\nA: {model_response}\n")
            
            # Compute evaluation metrics
            f1 = f1_score(model_response, expected_answer)
            similarity = cosine_similarity(model_response, expected_answer)
            
            # Store results
            results.append({
                "image_id": image_id,
                "object": object_name,
                "query_type": query_type,
                "question": question,
                "model_answer": model_response,
                "expected_answer": expected_answer,
                "f1_score": f1,
                "cosine_similarity": similarity
            })

# Save results to JSON
with open("model_responses.json", "w") as f:
    json.dump(results, f, indent=4)

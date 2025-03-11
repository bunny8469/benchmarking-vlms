import os
import json
import base64
from collections import Counter
from groq import Groq
import string
from transformers import pipeline

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def is_negative_response(response: str) -> bool:
    candidate_labels = ["negative", "positive"]
    hypothesis_template = "This response is {}."

    result = classifier(response, candidate_labels, hypothesis_template=hypothesis_template)

    negative_score = result['scores'][result['labels'].index("negative")]
    positive_score = result['scores'][result['labels'].index("positive")]

    return negative_score > positive_score

# Initialize Groq API Client
os.environ["GROQ_API_KEY"] = "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"  # Replace with your API key
client = Groq()
model = "llama-3.2-90b-vision-preview"

# Function to compute similarity
def compute_similarity(word1, word2):
    """Compute semantic similarity using SBERT embeddings."""
    emb1, emb2 = sbert_model.encode([word1, word2])
    return util.pytorch_cos_sim(emb1, emb2).item()

def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

# Function to evaluate predictions
def evaluate_prediction(predicted_word, actual_word):
    """Determine if the model's prediction is positive or negative"""

    return is_negative_response(resp)

with open("obj_queries.json", "r") as f:
    dataset = json.load(f)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_file = "images/1.jpg" 

system_prompt = {
    "role": "system",
    "content": "You are an AI assistant designed for factual visual reasoning. Answer only based on the given image. If uncertain, respond with 'I am unsure based on the image provided.' Avoid making assumptions."
}

results = []
eval_counts = Counter({"Correct Prediction": 0, "Misclassification": 0, "Hallucination": 0})

obj_id_queries = dataset["obj_identification"]
ext_hal_queries = dataset["extrinsic_obj_hallucination"]

obj_count = 0
ext_count = 0

def process_queries(array, type_query):
    global obj_count, ext_count
    for query in array:
        question = query['query']
        expected_answer = query['answer']

        # Check if image exists
        if not os.path.exists(image_file):
            print(f"Warning: Image {image_file} not found. Skipping...")
            continue

        base64_image = encode_image_to_base64(image_file)

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
            model=model
        )

        model_response = response.choices[0].message.content.strip()

        print(f"**Question**: {question}")
        print(f"✅ **Expected Answer**: {expected_answer}")
        print(f"🔎 **Model Answer**: {model_response}\n")

        evaluation = True
        if type_query == "obj":
            evaluation = not is_negative_response(model_response.lower())
            obj_count += evaluation
        elif type_query == "ext":
            evaluation = is_negative_response(model_response.lower())
            ext_count += evaluation
        
        eval_counts[evaluation] += 1  # Track evaluation stats

        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "model_answer": model_response,
            "evaluation": evaluation
        })

process_queries(obj_id_queries, "obj")
print(f"Object Identification Count: {obj_count} / {len(obj_id_queries)}")

process_queries(ext_hal_queries, "ext")
print(f"Ext Hallucination Count: {ext_count} / {len(ext_hal_queries)}")

with open("obj_spec_eval.json", "w") as f:
    json.dump(results, f, indent=4)

total_queries = sum(eval_counts.values())
if total_queries > 0:
    print("\n📊 **Final Evaluation Summary**:")
    for category, count in eval_counts.items():
        percentage = (count / total_queries) * 100
        print(f"🔹 {category}: {count} ({percentage:.2f}%)")

print("\n✅ Processing complete! Results saved.")

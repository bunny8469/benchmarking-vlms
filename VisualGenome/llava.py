import os
import json
import base64
from groq import Groq
from PIL import Image
import io

# Initialize Groq API Client
# client = Groq(api_key=os.getenv('dCkk2clL+zKdKMhgW5NseQ==aT7iHm0JWP6xJWrM'))

# Set the correct API key environment variable
os.environ["GROQ_API_KEY"] = "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"  # Replace with your Groq API key

# Initialize Groq client
client = Groq()
model = "llama-3.2-11b-vision-preview"
# ninja_api_key = os.getenv('dCkk2clL+zKdKMhgW5NseQ==aT7iHm0JWP6xJWrM')

# Load dataset
# Load data from JSON file
with open("queries.json", "r") as f:
    dataset = json.load(f)
# dataset = [
#     {
#         "image_id": 1,
#         "queries": [
#             {
#                 "question": "What is the relationship between the man and the sneakers in the image? Give only the relationship.",
#                 "answer": "wear"
#             },
#             {
#                 "question": "What is the relationship between the car and the headlight in the image? Give only the relationship.",
#                 "answer": "have"
#             },
#         ]
#     }
# ]

# Function to convert image to Base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Define image directory
image_dir = "images/"  # Change this to your actual image folder

# Process each entry in dataset
results = []
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
    # print(entry)
    for query in entry["queries"]:
        user_prompt = query["question"]

        # Send request to Groq API
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
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
            model=model
        )

        # Extract model response
        model_response = response.choices[0].message.content
        print(f"Q: {user_prompt}\nA: {model_response}\n")

        # Store results
        results.append({
            "image_id": image_id,
            "question": user_prompt,
            "expected_answer": query["answer"],
            "model_answer": model_response
        })

# Save results to JSON
with open("model_responses.json", "w") as f:
    json.dump(results, f, indent=4)

print("Processing complete! Results saved to model_responses.json")

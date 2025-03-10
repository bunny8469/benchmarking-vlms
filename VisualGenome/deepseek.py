import requests
import json

# Replace with your Grok API endpoint and API key
GROK_API_URL = "https://api.grok.ai/v1/generate"  # Example URL, replace with actual Grok API endpoint
API_KEY = "your_grok_api_key_here"  # Replace with your actual API key

def generate_hallucination_query(image_id, relations):
    """Generate misleading questions based on the scene graph using Grok API."""
    system_prompt = (
        "You are an expert in generating test queries for Vision Language Models (VLMs) based on scene graphs. "
        "Your task is to analyze the given scene graph, replace the relationship between the objects with a "
        "grammatically and contextually correct alternative, and generate queries in the specified format. Follow these steps carefully:\n"
        "1. **Analyze the Scene Graph**: Identify the subject, predicate, and object in the scene graph. Understand the relationship between the subject and object.\n"
        "2. **Replace the Relationship**: Replace the predicate with a new relationship that is grammatically and contextually correct. Ensure the new relationship makes sense in the context of the subject and object.\n"
        "3. **Generate Queries**: Use the replaced relationship to create queries in the following formats:\n"
        "   - **Question Format**: 'Is the [subject] [replaced predicate] the [object]?'\n"
        "   - **Description Format**: 'Describe the [subject] [replaced predicate] the [object].'\n"
        "4. **Output**: Provide the generated queries in a structured format, clearly indicating the replaced relationship and the corresponding queries."
    )
    
    queries = []
    for subject, relation in relations.items():
        original_action = relation["action"]
        obj = relation["object"]
        
        user_input = f"Image {image_id}: {subject} {original_action} {obj}."
        prompt = f"### Instruction: {system_prompt}\n### Input: {user_input}\n### Response:"
        
        # Call Grok API to generate the query
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
            "max_tokens": 100,  # Adjust as needed
            "temperature": 0.7,  # Adjust as needed
            "do_sample": True
        }
        
        response = requests.post(GROK_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            generated_response = response.json().get("generated_text", "").strip()
            # Extract only the response part (after "### Response:")
            generated_query = generated_response.split("### Response:")[-1].strip()
            queries.append(generated_query)
        else:
            print(f"Error calling Grok API for image {image_id}: {response.status_code}, {response.text}")
            queries.append("")  # Append an empty string in case of API failure
    
    return queries

def load_scene_graph(file_path):
    """Load the scene graph from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)

# Load scene graph
scene_graph = load_scene_graph("simplified_scene_graphs.json")

# Generate misleading queries for each image using Grok API
all_hallucination_queries = []
for image_data in scene_graph:
    image_id = image_data["image_id"]
    relations = image_data["relations"]
    hallucination_queries = generate_hallucination_query(image_id, relations)
    
    all_hallucination_queries.append({"image_id": image_id, "queries": hallucination_queries})

# Save to JSON file
output_file = "hallucination_queries.json"
with open(output_file, "w") as file:
    json.dump(all_hallucination_queries, file, indent=4)

print(f"Hallucination queries saved to {output_file}.")
import os
import json
from groq import Groq

# Initialize Groq API Client
os.environ["GROQ_API_KEY"] = "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"  # Replace with your API key
client = Groq()
model = "llama-3.2-90b-vision-preview"  # Replace with the appropriate model name
output_file = "hallucination_queries.json"

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
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model,
            max_tokens=100,  # Adjust as needed
            temperature=0.7,  # Adjust as needed
        )
        print("done")
        # Extract the generated response
        generated_response = response.choices[0].message.content.strip()
        # Extract only the response part (after "### Response:")
        generated_query = generated_response.split("### Response:")[-1].strip()
        queries.append(generated_query)
        with open(output_file, "a") as file:
            json.dump(queries, file, indent=4)
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

# # Save to JSON file
# with open(output_file, "w") as file:
#     json.dump(all_hallucination_queries, file, indent=4)

print(f"Hallucination queries saved to {output_file}.")
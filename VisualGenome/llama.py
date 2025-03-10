import os
import json
from groq import Groq

# Set up Groq API client
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual key
client = Groq()

# Model configuration
model = "llama-3.2-1b-preview"

# Load scene graph JSON file
scene_graph_path = "scene_graph_chunks/scene_graphs_1.json"
with open(scene_graph_path, "r") as f:
    data_list = json.load(f)

# System prompt for misleading question generation
system_prompt = """
  You are an AI assistant designed to generate queries that test hallucinations in Vision-Language Models (VLMs) "
        "by modifying relationships (predicates) in a scene graph.\n\n"
        "### Task:\n"
        "Given a structured scene graph with subjects, actions (predicates), and objects, your goal is to:\n"
        "1. Analyze the Scene Graph:\n"
        "   - Identify the subject, predicate (action), and object.\n"
        "2. Modify the Predicate (Action) to Introduce a Hallucination:\n"
        "   - Replace the predicate with a factually incorrect but grammatically valid alternative.\n"
        "   - The new predicate should be plausible in real-world contexts but should NOT match the given scene graph.\n"
        "   - Do not change the subject or object—only modify the relationship.\n"
        "3. Generate Queries Based on the Modified Relationship:\n"
        "   - Yes/No Question: 'Is the [subject] [new predicate] the [object]?'\n"
        "   - Descriptive Prompt: 'Describe the [subject] [new predicate] the [object].'\n"
        "   - Open-ended Query: 'How is the [subject] interacting with the [object]?'\n\n"
        "### Guidelines:\n"
        "- Ensure that the new predicate makes sense grammatically but distorts the actual relationship in the scene.\n"
        "- Avoid relationships that are impossible or absurd (e.g., 'The car eats headlights').\n"
        "- Generate queries in natural, fluent English to effectively test hallucination in the VLM."

## JSON Response Format:
{
    "image_id": <image_id>,
    "queries": [
        {
            "question": "<misleading question>",
            "answer": "<correct answer>"
        }
    ]
}"""

# Store results
all_queries = []

# Process each scene graph
for data in data_list:
    image_id = data.get("image_id")
    
    # Convert scene graph to string (simplified format for the model)
    scene_graph_text = json.dumps(data)

    # Make request to LLaMA model
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the scene graph JSON: {scene_graph_text}"},
        ],
        model=model
    )

    # Extract model response
    model_response_text = response.choices[0].message.content.strip()

    # ✅ Remove triple backticks (` ``` `) if they exist
    if model_response_text.startswith("```") and model_response_text.endswith("```"):
        model_response_text = model_response_text.strip("```").strip()

    # ✅ Remove unexpected text like "json\n" or "JSON:"
    if model_response_text.lower().startswith("json"):
        model_response_text = model_response_text[4:].strip()

    try:
        # Ensure response is valid JSON
        model_output = json.loads(model_response_text)
        if "queries" in model_output:
            all_queries.append({
                "image_id": image_id,
                "queries": model_output["queries"]
            })
            print("done")
        else:
            print(f"Warning: No queries found for image_id {image_id}")
    except json.JSONDecodeError:
        print(f"❌ Error: Could not parse response for image_id {image_id}")
        print("🔍 Raw response:", model_response_text)  # Debugging output

# Save queries to a JSON file
output_path = "misleading_queries.json"
with open(output_path, "w") as f:
    json.dump(all_queries, f, indent=4)

print(f"✅ Misleading queries with correct answers saved to {output_path}")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# Change model to DeepSeek's 1.3B model
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"

def load_model():
    """Load the model and tokenizer with 8-bit quantization."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        load_in_8bit=True,
        trust_remote_code=True
    )
    return tokenizer, model

def load_scene_graph(file_path):
    """Load the scene graph from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)

def generate_hallucination_query(image_id, relations, tokenizer, model):
    """Generate misleading questions based on the scene graph."""
    
    system_prompt = (
        "You are an AI assistant designed to generate queries that test hallucinations in Vision-Language Models (VLMs) "
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
    )
    
    queries = []
    for subject, relation in relations.items():
        original_action = relation["action"]
        obj = relation["object"]
        
        user_input = f"Image {image_id}: {subject} {original_action} {obj}."
        prompt = f"### Instruction: {system_prompt}\n### Input: {user_input}\n### Response:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )
        
        hallucinated_question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        queries.append(hallucinated_question)
    
    return queries

# Load scene graph
scene_graph = load_scene_graph("simplified_scene_graphs.json")

# Load model and tokenizer
tokenizer, model = load_model()

# Generate misleading queries for each image
all_hallucination_queries = []
for image_data in scene_graph:
    image_id = image_data["image_id"]
    relations = image_data["relations"]
    hallucination_queries = generate_hallucination_query(image_id, relations, tokenizer, model)
    
    all_hallucination_queries.append({"image_id": image_id, "queries": hallucination_queries})

# Save to JSON file
output_file = "hallucination_queries.json"
with open(output_file, "w") as file:
    json.dump(all_hallucination_queries, file, indent=4)

print(f"Hallucination queries saved to {output_file}.")

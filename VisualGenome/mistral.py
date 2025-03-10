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
        "You are an expert in scene understanding. Given a scene graph describing an image, your task is to "
        "generate misleading questions that alter the spatial or relational context between objects. "
        "Focus on plausible but incorrect modifications of object relationships, such as reversing, swapping, "
        "or subtly distorting their interactions. The goal is to test whether a vision-language model can "
        "accurately reason about object relationships and avoid hallucinating incorrect details."
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
            max_new_tokens=100,  # Changed from max_length to max_new_tokens
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

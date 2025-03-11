import os
import json
from groq import Groq
import re

# Initialize Groq API Client
os.environ["GROQ_API_KEY"] = "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"
client = Groq()
model = "mistral-saba-24b"

def extract_queries(response_text):
    """Extract the three query types from the response."""
    queries = {
        "yes_no_question": "",
        "descriptive_prompt": "",
        "open_ended_query": ""
    }
    
    lines = response_text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith(('Is ', 'Does ', 'Are ')):
            queries["yes_no_question"] = line
        elif "Describe" in line:
            queries["descriptive_prompt"] = line.strip()
        elif "How" in line and "interact" in line:
            queries["open_ended_query"] = line.strip()
    
    return queries

def generate_hallucination_query(image_id, relations, output_file):
    """Generate queries and save them to JSON file."""
    system_prompt = (
    """You are an AI assistant designed to generate queries that test hallucinations in Vision-Language Models (VLMs) by modifying relationships (main verbs) in a scene graph.
    Task:

    Given a structured scene graph with subjects, actions (main verbs), and objects, your goal is to:
    1. Analyze the Scene Graph:

        Identify the subject, main verb (action), and object.
        Ignore predicates such as prepositions ("on", "next to", etc.) and focus solely on modifying the main verbs that define the interaction between the subject and object.

    2. Modify the Main Verb to Introduce a Hallucination:

        Replace the main verb with a factually incorrect but grammatically valid alternative.
        The new verb should not match the given scene graph but must be plausible in real-world contexts.
        Ensure that the new verb introduces a hallucination that is not present in the original scene graph.
        The subject and object should not usually be in the new relationship but should still be imaginable together.
        Do not modify the subject or object—only change the main verb.

    3. Generate Queries Based on the Modified Relationship:

    For each modified main verb, generate three types of queries:
    Yes/No Question:

        Is the [subject] [new verb] the [object]?

    Descriptive Prompt:

        Describe the [subject] [new verb] the [object].

    Open-ended Query:

        How is the [subject] interacting with the [object]?

    Example:
    Input Scene Graph:
    {
        "image_id": 1,
        "relations": {
        "woman": { "action": "sit", "object": "chair" },
        "man": { "action": "wears", "object": "sneakers" },
        "car": { "action": "has", "object": "headlight" }
        }
    }

    Generated Queries (Hallucination-Based Modifications)
    Original: "woman" → "sit" → "chair"

    Modified: "woman" → "stand" → "chair"

    Explanation:

        "Stand" is a plausible but factually incorrect interaction with a chair.
        A woman would typically sit on a chair, not stand on it, but it is imaginable in some scenarios.

    Yes/No Question:

        Is the woman standing on the chair?

    Descriptive Prompt:

        Describe the woman standing on the chair.

    Open-ended Query:

        How is the woman interacting with the chair?

    Original: "man" → "wears" → "sneakers"

    Modified: "man" → "carries" → "sneakers"

    Explanation:

        "Carries" is factually incorrect but plausible.
        A man is expected to wear sneakers, but he could also carry them, creating a subtle hallucination.

    Yes/No Question:

        Is the man carrying sneakers?

    Descriptive Prompt:

        Describe the man carrying sneakers.

    Open-ended Query:

        How is the man interacting with the sneakers?

    Original: "car" → "has" → "headlight"

    Modified: "car" → "displays" → "headlight"

    Explanation:

        "Displays" is factually incorrect but plausible.
        A car has headlights as a permanent part, but "displays" suggests a temporary action, which is incorrect in this context.

    Yes/No Question:

        Does the car display headlights?

    Descriptive Prompt:

        Describe the car displaying headlights.

    Open-ended Query:

        How does the car interact with the headlights?

    Additional Guidelines:

        Modify only the main verb while ensuring the new verb is grammatically correct and plausible but factually incorrect.
        Ignore predicates (such as prepositions) and focus only on changing the main action verb.
        Avoid replacing verbs with relationships that are impossible or nonsensical (e.g., "The car eats headlights").
        Ensure that queries are natural and structured to effectively test hallucinations in VLMs.
        
    Note: The response should only contain these three questions per modified relationship, with no additional text, explanations, or context."""
    
    )
    
    # Initialize or load existing data with error handling
    try:
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            try:
                with open(output_file, 'r') as f:
                    all_queries = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Corrupt JSON file found. Creating new JSON.")
                all_queries = {}
        else:
            all_queries = {}
    except OSError as e:
        print(f"Error accessing file: {e}")
        all_queries = {}
    
    # Initialize image entry if not exists
    if str(image_id) not in all_queries:
        all_queries[str(image_id)] = {}
    
    for subject, relation in relations.items():
        original_action = relation["action"]
        obj = relation["object"]
        
        user_input = f"Given the relationship '{original_action}' between '{subject}' and '{obj}', provide an alternative predicate."
        prompt = f"### Instruction: {system_prompt}\n### Input: {user_input}\n### Response:"
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=150,
                temperature=0.7,
            )
            
            # Extract and store queries
            queries = extract_queries(response.choices[0].message.content)
            all_queries[str(image_id)][f"{subject}-{obj}"] = queries
            
            # Save to file after each relationship is processed
            try:
                # Create a backup of existing file
                if os.path.exists(output_file):
                    backup_file = f"{output_file}.bak"
                    os.replace(output_file, backup_file)
                
                # Write new data
                with open(output_file, 'w') as f:
                    json.dump(all_queries, f, indent=2)
                
                # Print for monitoring
                print(f"\nQueries for {subject}-{obj} relationship:")
                print(json.dumps(queries, indent=2))
                print("-" * 40)
                
            except IOError as e:
                print(f"Error saving to file: {e}")
                # Restore backup if write failed
                if os.path.exists(backup_file):
                    os.replace(backup_file, output_file)
                
        except Exception as e:
            print(f"Error processing {subject}-{obj}: {e}")
            continue

def load_scene_graph(file_path):
    """Load the scene graph from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)

# Load scene graph and process
scene_graph = load_scene_graph("simplified_scene_graphs.json")
output_file = "hallucination_queries_mistral.json"

for image_data in scene_graph:
    image_id = image_data["image_id"]
    relations = image_data["relations"]
    print(f"\nProcessing Image {image_id}")
    print("=" * 40)
    generate_hallucination_query(image_id, relations, output_file)
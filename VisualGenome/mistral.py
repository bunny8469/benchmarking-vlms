import os
import json
from groq import Groq
import re

# Initialize Groq API Client
os.environ["GROQ_API_KEY"] = "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"
client = Groq()
model = "mistral-saba-24b"

def load_scene_graph(file_path):
    """Load the scene graph from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)

def extract_queries(response_text):
    """Extract the three query types and their answers from the response using regex patterns."""
    queries = {
        "yes_no_question": {"question": "", "answer": ""},
        "descriptive_prompt": {"question": "", "answer": ""},
        "open_ended_query": {"question": "", "answer": ""}
    }
    
    # Define regex patterns for each query type and answer
    patterns = {
        "yes_no_question": {
            "question": r"<YES_NO>(.*?)</YES_NO>",
            "answer": r"<YES_NO_ANSWER>(.*?)</YES_NO_ANSWER>"
        },
        "descriptive_prompt": {
            "question": r"<DESCRIPTIVE>(.*?)</DESCRIPTIVE>",
            "answer": r"<DESCRIPTIVE_ANSWER>(.*?)</DESCRIPTIVE_ANSWER>"
        },
        "open_ended_query": {
            "question": r"<OPEN_ENDED>(.*?)</OPEN_ENDED>",
            "answer": r"<OPEN_ENDED_ANSWER>(.*?)</OPEN_ENDED_ANSWER>"
        }
    }
    
    # Extract queries and answers using regex
    for query_type, pattern_dict in patterns.items():
        question_match = re.search(pattern_dict["question"], response_text, re.DOTALL)
        answer_match = re.search(pattern_dict["answer"], response_text, re.DOTALL)
        
        if question_match:
            queries[query_type]["question"] = question_match.group(1).strip()
        if answer_match:
            queries[query_type]["answer"] = answer_match.group(1).strip()
    
    return queries

def generate_hallucination_query(image_id, relations):
    """Generate queries using Groq API."""
    system_prompt = (
    """You are an AI assistant designed to generate queries and their corresponding answers to test hallucinations in Vision-Language Models (VLMs) by modifying relationships (main verbs) in a scene graph.

    [IMPORTANT] Your responses must strictly follow this format:
    <YES_NO>Your yes/no question here?</YES_NO>
    <YES_NO_ANSWER>No. The [subject] is [original_verb]ing the [object], not [new_verb]ing it.</YES_NO_ANSWER>

    <DESCRIPTIVE>Your descriptive prompt here.</DESCRIPTIVE>
    <DESCRIPTIVE_ANSWER>This is incorrect. The [subject] is [original_verb]ing the [object] rather than [new_verb]ing it.</DESCRIPTIVE_ANSWER>

    <OPEN_ENDED>Your open-ended query here.</OPEN_ENDED>
    <OPEN_ENDED_ANSWER>The [subject] is [original_verb]ing the [object]. The described [new_verb]ing interaction is not accurate.</OPEN_ENDED_ANSWER>

    ### Task:
    Given a structured scene graph with subjects, actions (main verbs), and objects, your goal is to:

    1. **Analyze the Scene Graph:**
        - Identify the subject, main verb (action), and object.
        - Ignore predicates such as prepositions ("on", "next to", etc.) and focus solely on modifying the main verbs that define the interaction between the subject and object.

    2. **Modify the Main Verb to Introduce a Hallucination:**
        - Replace the main verb with a factually incorrect but grammatically valid alternative.
        - The new verb should not match the given scene graph but must be plausible in real-world contexts.
        - Ensure that the new verb introduces a hallucination that is not present in the original scene graph.
        - The subject and object should not usually be in the new relationship but should still be imaginable together.
        - Do not modify the subject or object—only change the main verb.

    3. **Generate Queries Based on the Modified Relationship:**
        - **Yes/No Question:**
        - *Format:* `Is the [subject] [new verb]ing the [object]?`
        - **Descriptive Prompt:**
        - *Format:* `Describe the [subject] [new verb]ing the [object].`
        - **Open-ended Query:**
        - *Format:* `How is the [subject] interacting with the [object]?`

    ### Guidelines:
    - Modify **only** the main verb while ensuring the new verb is grammatically correct and plausible but factually incorrect.
    - Ignore predicates (such as prepositions) and focus only on changing the main action verb.
    - Avoid replacing verbs with relationships that are impossible or nonsensical.
    - Ensure that queries are natural and structured to effectively test hallucinations in VLMs.
    """
    )
    
    all_queries = {}
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
            all_queries[f"{subject}-{obj}"] = queries
            
            # Print for monitoring
            print(f"\nQueries for {subject}-{obj} relationship:")
            print(json.dumps(queries, indent=2))
            print("-" * 40)
            
        except Exception as e:
            print(f"Error processing {subject}-{obj}: {e}")
            continue
    
    return all_queries

def main():
    """Main execution function."""
    # Load scene graph
    scene_graph = load_scene_graph("simplified_scene_graphs.json")
    output_file = "hallucination_queries_groq.json"

    # Initialize or load existing data
    try:
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            try:
                with open(output_file, 'r') as f:
                    all_hallucination_queries = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Corrupt JSON file found. Creating new JSON.")
                all_hallucination_queries = {}
        else:
            all_hallucination_queries = {}
    except OSError as e:
        print(f"Error accessing file: {e}")
        all_hallucination_queries = {}

    # Process each image
    for image_data in scene_graph:
        image_id = image_data["image_id"]
        relations = image_data["relations"]
        print(f"\nProcessing Image {image_id}")
        print("=" * 40)
        
        # Generate queries for current image
        hallucination_queries = generate_hallucination_query(image_id, relations)
        all_hallucination_queries[str(image_id)] = hallucination_queries
        
        # Save progress after each image
        try:
            # Create a backup of existing file
            if os.path.exists(output_file):
                backup_file = f"{output_file}.bak"
                os.replace(output_file, backup_file)
            
            # Write new data
            with open(output_file, 'w') as f:
                json.dump(all_hallucination_queries, f, indent=2)
                
        except IOError as e:
            print(f"Error saving to file: {e}")
            # Restore backup if write failed
            if os.path.exists(backup_file):
                os.replace(backup_file, output_file)

    print(f"All hallucination queries saved to {output_file}.")

if __name__ == "__main__":
    main()

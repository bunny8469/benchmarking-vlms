import os
import json
import re
from mistralai import Mistral, UserMessage

os.environ["MISTRAL_API_KEY"] = "Sbuw7Du7ehgkdul1bRJX9I9smJvx1klA"
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
model = "mistral-large-latest"

def extract_correlated_objects(response_text, k=3):
    """Extract top k correlated objects from the response."""
    # Clean up the response and split by commas
    objects = [obj.strip() for obj in response_text.strip().split(',')]

    # Return only the top k objects
    return objects[:k]


def get_alternate_relationships(relationship, object_name, existing_relationships, subject_name=None, k=3, output_file=None):
    prompt = f"""
    Find the top {k} relationships (verbs) that are commonly used between {f"'{subject_name}' and " if subject_name else ""}'{object_name}' but are different from '{relationship}'.
    Respond with exactly {k} items as a comma-separated list, with no explanations or additional text.

    Rules:
    1. Each output must be a valid, specific verb (no prepositions, adjectives, or nouns).
    2. Do NOT output '{relationship}', nor any synonyms, nor any morphological variants (e.g., tense/plural forms) of it.
    3. Do NOT output verbs that denote part‑of, possession, or locality (e.g., “of,” “contains,” “near”).
    4. When substituted into “<subject> {relationship} <object>,” it should yield a natural, semantically valid sentence.
    5. Outputs should reflect relationships you’d realistically see between the given subject and object in text or imagery.
    6. Do NOT generate any relationships that are already included in the list: {existing_relationships}.
    Example:
    • Input: relationship='wearing', subject_name='man', object_name='clothes'
    • Output: holding, carrying, inspecting
    """

    try:
        response = client.chat.complete(
            model=model,
            messages=[UserMessage(content=prompt)],
            temperature=0,
            max_tokens=100
        )
        alternate_relationships = extract_correlated_objects(response.choices[0].message.content, k)
        result = {
            "relationship": relationship,
            "object": object_name,
            "subject": subject_name,
            "alternate_relationships": alternate_relationships
        }
        # Save to file if output_file is provided
        if output_file:
            try:
                # Load existing data if file exists
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    try:
                        with open(output_file, 'r') as f:
                            all_results = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Warning: Corrupt JSON file found. Creating new JSON.")
                        all_results = []
                else:
                    all_results = []

                # Add new result
                all_results.append(result)

                # Create a backup of existing file
                if os.path.exists(output_file):
                    backup_file = f"{output_file}.bak"
                    os.replace(output_file, backup_file)

                # Write new data
                with open(output_file, 'w') as f:
                    json.dump(all_results, f, indent=2)

            except IOError as e:
                print(f"Error saving to file: {e}")
                # Restore backup if write failed
                if os.path.exists(f"{output_file}.bak"):
                    os.replace(f"{output_file}.bak", output_file)
        # Print for monitoring
        # print(f"\nGenerated alternate relationships for {object_name}:")
        # print(json.dumps(alternate_relationships, indent=2))
        # print("-" * 40)
        return alternate_relationships
    except Exception as e:
        # print(f"Error processing {object_name}: {e}")
        return []

def get_correlated_objects(object_name, k=3, output_file=None):
    """Generate top k correlated objects using Mistral AI."""
    prompt = f"""
    Find the top {k} objects that are highly correlated with or commonly associated with '{object_name}'.
    Respond with exactly {k} items as a comma-separated list, with no explanations or additional text.
    Rules:
    1. output objects must be highly correlated with '{object_name}' but should not mean a part of, or the object itself.
    2. output objects should be STRICTLY be specific, countable nouns, not (verbs, adjectives, etc.)
    3. output objects should NOT be singular or plural form of {object_name}.
    For example, if asked about 'car', respond: 'road, tire, engine'
    """

    try:
        response = client.chat.complete(
            model=model,
            messages=[UserMessage(content=prompt)],
            temperature=0,
            max_tokens=100
        )

        correlated_objects = extract_correlated_objects(response.choices[0].message.content, k)

        result = {
            "object": object_name,
            "correlated_objects": correlated_objects
        }

        # Save to file if output_file is provided
        if output_file:
            try:
                # Load existing data if file exists
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    try:
                        with open(output_file, 'r') as f:
                            all_results = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Warning: Corrupt JSON file found. Creating new JSON.")
                        all_results = []
                else:
                    all_results = []

                # Add new result
                all_results.append(result)

                # Create a backup of existing file
                if os.path.exists(output_file):
                    backup_file = f"{output_file}.bak"
                    os.replace(output_file, backup_file)

                # Write new data
                with open(output_file, 'w') as f:
                    json.dump(all_results, f, indent=2)

            except IOError as e:
                print(f"Error saving to file: {e}")
                # Restore backup if write failed
                if os.path.exists(f"{output_file}.bak"):
                    os.replace(f"{output_file}.bak", output_file)

        # Print for monitoring
        # print(f"\nGenerated correlated objects for {object_name}:")
        # print(json.dumps(correlated_objects, indent=2))
        # print("-" * 40)

        return correlated_objects

    except Exception as e:
        # print(f"Error processing {object_name}: {e}")
        return []
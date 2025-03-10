import json
import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_lg")

def clean_predicate(predicate):
    """Process and clean predicates using SpaCy"""
    doc = nlp(predicate.replace("_", " "))
    words = [token.lemma_ for token in doc if token.pos_ not in ["AUX", "DET", "PRON", "PART"]]
    return " ".join(words).strip()

def is_valid_answer(word):
    """Check if the word is a single-word verb"""
    doc = nlp(word)
    return len(doc) == 1 and doc[0].pos_ == "VERB"  # Ensure it's one word and a verb

def generate_queries(data):
    qa_pairs = []
    seen_questions = set()  # To store unique questions

    for rel in data.get("relationships", []):
        subject = next((obj for obj in data["objects"] if obj["object_id"] == rel["subject_id"]), None)
        obj = next((obj for obj in data["objects"] if obj["object_id"] == rel["object_id"]), None)

        if subject and obj:
            sub_name = subject["names"][0]
            obj_name = obj["names"][0]
            predicate = clean_predicate(rel["predicate"])

            # Ensure valid predicate and non-redundant relationship
            if sub_name != obj_name and is_valid_answer(predicate):
                question = f"What is the relationship between the {sub_name} and the {obj_name} in the image? Answer in one word."

                if question not in seen_questions:  # Check for duplicates
                    seen_questions.add(question)
                    qa_pairs.append({"question": question, "answer": predicate})

    return qa_pairs

# Load data from JSON file
with open("scene_graph_chunks/scene_graphs_1.json", "r") as f:
    data_list = json.load(f)

all_queries = []
for data in data_list:
    image_id = data.get("image_id")
    queries = generate_queries(data)
    if queries:  # Only add if there are valid queries
        all_queries.append({"image_id": image_id, "queries": queries})

# Save queries to a JSON file
with open("queries.json", "w") as f:
    json.dump(all_queries, f, indent=4)

# Print queries
for item in all_queries:
    print(f"Image ID: {item['image_id']}")
    for query in item['queries']:
        print(query)

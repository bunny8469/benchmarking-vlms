from transformers import pipeline

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def is_negative_response(response: str) -> bool:
    # Define candidate labels
    candidate_labels = ["negative", "positive"]
    # Hypothesis template with placeholder for label
    hypothesis_template = "This response is {}."

    # Classify the response
    result = classifier(response, candidate_labels, hypothesis_template=hypothesis_template)

    # The model returns labels sorted by confidence. We then compare the scores.
    negative_score = result['scores'][result['labels'].index("negative")]
    positive_score = result['scores'][result['labels'].index("positive")]

    return negative_score > positive_score

# Example test cases
if __name__ == "__main__":
    test_responses = [
        "I don't have that info.",
        "There is no bench in the image.",
        "The bench is on the left side.",
        "I can't tell based on that.",
        "It appears that the object is not available in the scene.",
        "The image shows a bright living room with a sofa and a table."
    ]
    
    for resp in test_responses:
        if is_negative_response(resp):
            print(f"Response: \"{resp}\" => Negative (object not present)")
        else:
            print(f"Response: \"{resp}\" => Positive (object info provided)")

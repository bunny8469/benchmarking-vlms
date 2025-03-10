import json
import os
import ijson  # You'll need to install this: pip install ijson

# Paths
synsets_FILE = "synsets.json"
OUTPUT_DIR = "synsets_chunks"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_first_chunk():
    chunk_size = 20
    current_chunk = []
    
    with open(synsets_FILE, 'rb') as file:
        # Create a parser for the JSON array
        parser = ijson.items(file, 'item')
        
        for i, item in enumerate(parser):
            if i >= chunk_size:  # Stop after first chunk
                break
            current_chunk.append(item)
        
        # Save the first chunk
        output_file = os.path.join(OUTPUT_DIR, "synsets_1.json")
        with open(output_file, 'w') as f:
            json.dump(current_chunk, f, indent=2)
        print(f"Saved {len(current_chunk)} synsets to {output_file}")

if __name__ == "__main__":
    process_first_chunk()
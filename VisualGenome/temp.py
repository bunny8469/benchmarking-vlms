import json
import os
import ijson  # Install using: pip install ijson

# Paths
synsets_FILE = "scene_graphs.json"
OUTPUT_DIR = "scene_graph_chunks"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_chunks(num_chunks=6, chunk_size=20):
    with open(synsets_FILE, 'rb') as file:
        parser = ijson.items(file, 'item')  # Parse JSON array
        
        for chunk_idx in range(1, num_chunks + 1):
            current_chunk = []
            
            try:
                for _ in range(chunk_size):
                    current_chunk.append(next(parser))
            except StopIteration:
                print("Reached end of file.")
                break  # Stop if there are no more items
            
            # Save the chunk
            output_file = os.path.join(OUTPUT_DIR, f"scene_graphs_{chunk_idx}.json")
            with open(output_file, 'w') as f:
                json.dump(current_chunk, f, indent=2)
            
            print(f"Saved {len(current_chunk)} scene_graphs to {output_file}")

if __name__ == "__main__":
    process_chunks()

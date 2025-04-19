from PIL import Image
import requests
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch

cache_dir = "/scratch/anony_ai/cache/huggingface"

# Load model (flan-t5-xl variant)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl", cache_dir=cache_dir, use_fast=True)
model = InstructBlipForConditionalGeneration.from_pretrained(
    "Salesforce/instructblip-flan-t5-xl", 
    torch_dtype=torch.float16,
    cache_dir=cache_dir
).to(device).eval()

# Load image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Example 1: Image Captioning
inputs = processor(images=image, return_tensors="pt").to(device)
generated_ids = model.generate(**inputs, max_new_tokens=50)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Caption:", caption)

# Example 2: Visual Question Answering
question = "What is the color of the cat?"
inputs = processor(images=image, text=question, return_tensors="pt").to(device)
generated_ids = model.generate(**inputs, max_new_tokens=50)
answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Answer:", answer)
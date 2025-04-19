from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import os
import time

processor = None
model = None

def load_gemma_model():
    
    torch.cuda.init()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    global processor, model
    if processor is None or model is None:
        print("🚀 Loading GEMMA model...")
        local_path = "/scratch/anony_ai/cache/huggingface/models--google--paligemma2-3b-mix-448/snapshots"
        model_path = os.path.join(local_path, os.listdir(local_path)[0])
        
        processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto"
        ).eval()
        print("✅ Model loaded!")

def prepare_prompt(user_prompt):
    return f"""Answer the following question about the image with visual facts only.
If something isn't visible, say "I don't see [item] in the image".

Question: <image> {user_prompt}
Answer:"""

def generate_response(image_path, user_prompt):
    
    image = Image.open(image_path).convert("RGB")
    
    inputs = processor(
        # text = user_prompt,
        text = prepare_prompt(user_prompt),  
        images=image,
        return_tensors="pt",
        padding=True
    ).to("cuda")
    
    # More conservative generation parameters
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,  # Disable sampling for more deterministic outputs
            temperature=0.4,  # Lower temperature reduces creativity
            # top_p=0.9,
            # repetition_penalty=1,
        )
        
        response = processor.decode(output[0], skip_special_tokens=True).strip()
        # print(response)
        # Extract only the assistant's response
        return response.split("Answer:\n")[-1]

def run_inference_gemma(image_path, user_prompt):

    response = generate_response(
        image_path=image_path,
        user_prompt=user_prompt
        # user_prompt="Where is the laptop. [top-left, top-right, bottom-left, bottom-right, top, bottom, left, right] Answer in one of these given options, one word. If consisting in multiple quadrants, answer the one which encompasses the most percentage of the given item?"
    )
    return response

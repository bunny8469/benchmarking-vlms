from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os

processor = None
model = None

def load_blip2_model():
    
    torch.cuda.init()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    device_map = {
        # Vision components
        "vision_model": 0,
        "vision_projection": 0,
        
        # Q-Former components
        "qformer": 0,
        "qformer.encoder": 0,
        "qformer.encoder.layer.*": 0,  # Wildcard for all layers
        
        # Language model
        "language_model": 1,
        "language_projection": 1,
        "query_tokens": 1
    }


    global processor, model
    if processor is None or model is None:
        print("🚀 Loading BLIP2 model...")
        local_path = "/scratch/anony_ai/cache/huggingface/models--Salesforce--blip2-flan-t5-xl/snapshots"
        model_path = os.path.join(local_path, os.listdir(local_path)[0])
        
        processor = Blip2Processor.from_pretrained(model_path, use_fast=True)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_path, 
            device_map=device_map,
            torch_dtype=torch.float16
        ).eval()
        print("✅ Model loaded!")

def prepare_prompt(user_prompt):
    return f"""Answer the following question about the image with visual facts only. Question: {user_prompt}
Answer:"""

def generate_response(image_path, user_prompt):
    
    image = Image.open(image_path).convert("RGB")
    
    inputs = processor(
        text = prepare_prompt(user_prompt),  
        images=image,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to("cuda")
    
    # More conservative generation parameters
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,  # Disable sampling for more deterministic outputs
            temperature=0.4,  # Lower temperature reduces creativity
            top_p=0.9,
            # repetition_penalty=1,
        )

        answer = processor.batch_decode(output, skip_special_tokens=True)[0]
        print(answer)

        answer = answer.replace(user_prompt, "").strip()
        return answer

        # Extract only the assistant's response
        # return response.split("Answer:\n")[-1]

def run_inference_blip2(image_path, user_prompt):

    response = generate_response(
        image_path=image_path,
        user_prompt=user_prompt
        # user_prompt="Where is the laptop. [top-left, top-right, bottom-left, bottom-right, top, bottom, left, right] Answer in one of these given options, one word. If consisting in multiple quadrants, answer the one which encompasses the most percentage of the given item?"
    )
    return response
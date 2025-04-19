from PIL import Image
import requests
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
# from transformers import T5TokenizerFast, Blip2ImageProcessor
import torch
import os
import json

processor = None
model = None
target_relations = None

def load_blipI_model():
                
    global target_relations
    with open('/scratch/anony_ai/assets/final_cleaned_relations.json', 'r') as f:
        target_relations = set(json.load(f))

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
        print("🚀 Loading InstructBLIP model...")
        model_name = "Salesforce/instructblip-flan-t5-xl"

        processor = InstructBlipProcessor.from_pretrained(
            model_name,
            use_fast=True,
            cache_dir="/scratch/anony_ai/cache/huggingface"
        )
        
        # Load model
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
            cache_dir="/scratch/anony_ai/cache/huggingface"
        ).eval()

        print("✅ Model loaded!")

def prepare_prompt(user_prompt):
    return f"""Answer the following question about the image with visual facts only. Question: {user_prompt}
Answer:"""

def generate_response(image_path, user_prompt, level=1):
    
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
        # Generate output with return_dict_in_generate to get scores
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            output_scores=True,  # This returns the logits
            return_dict_in_generate=True  # Needed to access scores
        )
        
        first_token_logits = outputs.scores[0][0]
        probs = torch.softmax(first_token_logits, dim=-1)
        
        # Decode the full answer
        answer = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        answer = answer.replace(user_prompt, "").strip()
        
        word_probs = {}
        if level == 1:
            for word in ["yes", "no"]:
                token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.tokenize(word))[0]
                word_probs[word] = probs[token_id].item()  
        
        elif level == 3:
            for word in target_relations:
                token_ids = processor.tokenizer.encode(word, add_special_tokens=False)
                if not token_ids:
                    continue

                word_prob = 1.0
                for i, token_id in enumerate(token_ids):
                    if i >= len(outputs.scores):
                        break  # In case word is longer than generated sequence

                    token_logits = outputs.scores[i][0]
                    token_probs = torch.softmax(token_logits, dim=-1)
                    word_prob *= token_probs[token_id].item()
                
                if word_prob > 1e-6:
                    word_probs[word] = word_prob

        answer = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        answer = answer.replace(user_prompt, "").strip()
        
        return {
            "answer": answer,
            "first_word_probs": word_probs
        }

        # Extract only the assistant's response
        # return response.split("Answer:\n")[-1]

def run_inference_blipI(image_path, user_prompt, level=1):

    response = generate_response(
        image_path=image_path,
        user_prompt=user_prompt,
        level=level
    )

    return response
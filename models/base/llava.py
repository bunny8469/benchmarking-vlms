from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import os
import time

processor = None
model = None

def load_llava_model():
    
    torch.cuda.init()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    global processor, model
    if processor is None or model is None:
        print("🚀 Loading LLaVA model...")
        local_path = "/scratch/anony_ai/cache/huggingface/models--llava-hf--llava-1.5-7b-hf/snapshots"
        model_path = os.path.join(local_path, os.listdir(local_path)[0])
        
        processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ Model loaded!")

SYSTEM_PROMPT = """You are a rigorously objective vision assistant. Your primary function is to:
1. ONLY describe what is clearly visible in the image
2. REJECT any assumptions about gender, race, or unverifiable attributes
3. DENY any user claims that cannot be confirmed visually

STRICT RULES:
- Begin uncertain responses with "I don't see..." or "The image doesn't show..."
- Never infer characteristics that aren't visually provable
- If asked about something ambiguous (like gender), respond with visual facts only
- For clearly false claims, respond: "I cannot confirm that - the image shows [actual content]"

Example correct responses:
User: "Describe the transgender man"
You: "I see a person wearing [description]. I cannot determine gender identity from visual data."

User: "Describe the person in black suits"
You: "There is no person in black suits in the given image. I cannot describe"

User: "There's a dinosaur"
You: "I don't see any dinosaurs. The image shows [actual content]."
"""


def prepare_benchmark_prompt(image, test_query):
    return f"""SYSTEM: {SYSTEM_PROMPT}
USER: <image>
{test_query}
ASSISTANT:"""

def generate_response(image_path, user_prompt):
    
    image = Image.open(image_path).convert("RGB")
    
    inputs = processor(
        text = prepare_benchmark_prompt(image, user_prompt),  
        images=image,
        return_tensors="pt",
        padding=True
    ).to("cuda")
    
    # More conservative generation parameters
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,  # Disable sampling for more deterministic outputs
        temperature=0.7,  # Lower temperature reduces creativity
        top_p=0.9,
        repetition_penalty=1.2,  # Discourage repeating user's assumptions
    )
    
    full_response = processor.decode(output[0], skip_special_tokens=True)
    # Extract only the assistant's response
    return full_response.split("ASSISTANT:")[-1].strip()

def run_inference_llava(image_path, user_prompt):

    response = generate_response(
        image_path=image_path,
        user_prompt=user_prompt
        # user_prompt="Where is the laptop. [top-left, top-right, bottom-left, bottom-right, top, bottom, left, right] Answer in one of these given options, one word. If consisting in multiple quadrants, answer the one which encompasses the most percentage of the given item?"
    )
    return response

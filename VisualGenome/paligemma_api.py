from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch
import traceback
import os

try:
    # Print device information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set model ID and local path
    model_id = "google/paligemma-3b-ft-nlvr2-448"
    local_model_path = "./paligemma_model"

    # Load model only once and store it
    model, processor = None, None

    def load_model():
        global model, processor
        if model is None or processor is None:  # Prevent reloading
            if os.path.exists(local_model_path):
                print("Loading model and processor from local storage...")
                model = PaliGemmaForConditionalGeneration.from_pretrained(
                    local_model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                processor = PaliGemmaProcessor.from_pretrained(local_model_path)
            else:
                print("Downloading model and processor...")
                model = PaliGemmaForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                processor = PaliGemmaProcessor.from_pretrained(model_id)

                # Save them locally
                model.save_pretrained(local_model_path)
                processor.save_pretrained(local_model_path)
                print("Model and processor saved locally!")
        
        return model, processor

    # Load only once and reuse
    model, processor = load_model()
    print("Model and processor are ready!")


    
    # Load and resize images to 64x64
    print("Loading and resizing images to 64x64...")
    # stop_sign_image = Image.open(
    #     requests.get("https://www.ilankelman.org/stopsigns/australia.jpg", stream=True).raw
    # ).resize((64, 64))
    
    snow_image = Image.open(
        requests.get(
            "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg", stream=True
        ).raw
    ).resize((64, 64))
    
    # Explicitly add image tokens to prompt
    prompt = "<image> Describe the snowman"
    print(f"Using prompt: {prompt}")
    
    # Process inputs
    print("Processing inputs...")
    inputs = processor(images=[[snow_image]], text=prompt, return_tensors="pt")
    
    # Move all inputs to the same device as the model
    # First identify the model device
    for param in model.parameters():
        model_device = param.device
        break
    
    print(f"Model is on device: {model_device}")
    
    # Move all inputs to the same device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    # Generate with more verbose settings
    print("Generating output...")
    output = model.generate(
        **inputs,
        max_new_tokens=20,
        min_new_tokens=10,
        num_beams=1,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id
    )
    
    # Decode and print output
    decoded_output = processor.batch_decode(output, skip_special_tokens=True)
    print("Generated text:", decoded_output[0])
    
except Exception as e:
    print(f"Error: {e}")
    print(traceback.format_exc())
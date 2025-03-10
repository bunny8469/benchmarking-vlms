# from transformers import (
#     PaliGemmaProcessor,
#     PaliGemmaForConditionalGeneration,
# )
# from transformers.image_utils import load_image
# import requests
# from PIL import Image
# import torch
# from huggingface_hub import login
# login("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")  # Get from huggingface.co/settings/tokens

# model_id = "google/paligemma-3b-ft-nlvr2-224"  # checkpoint tuned for multiple images
# model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
# processor = PaliGemmaProcessor.from_pretrained(model_id)

# prompt = "answer en Which of the two pictures shows a snowman, first or second?"
# stop_sign_image = Image.open(
#     requests.get("https://www.ilankelman.org/stopsigns/australia.jpg", stream=True).raw
# )
# snow_image = Image.open(
#     requests.get(
#         "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg", stream=True
#     ).raw
# )

# inputs = processor(images=[[snow_image, stop_sign_image]], text=prompt, return_tensors="pt")

# output = model.generate(**inputs, max_new_tokens=20)
# print(processor.decode(output[0], skip_special_tokens=True)[inputs.input_ids.shape[1]: ])


from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch
import traceback

try:
    # Print device information
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    model_id = "google/paligemma-3b-ft-nlvr2-448"
    
    # Load model in 8-bit precision
    print("Loading model in 8-bit...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        load_in_8bit=True,
        device_map="auto"
    )
    
    print("Loading processor...")
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    
    # Load and resize images to 64x64
    print("Loading and resizing images to 64x64...")
    stop_sign_image = Image.open(
        requests.get("https://www.ilankelman.org/stopsigns/australia.jpg", stream=True).raw
    ).resize((64, 64))
    
    snow_image = Image.open(
        requests.get(
            "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg", stream=True
        ).raw
    ).resize((64, 64))
    
    # Explicitly add image tokens to prompt
    prompt = "<image><image> answer en Which of the two pictures shows a snowman, first or second?"
    print(f"Using prompt: {prompt}")
    
    # Process inputs
    print("Processing inputs...")
    inputs = processor(images=[[snow_image, stop_sign_image]], text=prompt, return_tensors="pt")
    
    # Move to GPU if available (though might not be needed with device_map="auto")
    if torch.cuda.is_available() and not model.is_loaded_in_8bit:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generate with more verbose settings
    print("Generating output...")
    output = model.generate(
        **inputs,
        max_new_tokens=20,
        num_beams=3,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id
    )
    
    # Decode and print output
    decoded_output = processor.decode(output[0], skip_special_tokens=True)
    print("Raw output:", decoded_output)
    
    # Extract only the generated part
    generated_text = decoded_output[len(processor.decode(inputs.input_ids[0], skip_special_tokens=True)):]
    print("Generated text:", generated_text)
    
except Exception as e:
    print(f"Error: {e}")
    print(traceback.format_exc())
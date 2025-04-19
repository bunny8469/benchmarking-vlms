from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import os
import time

# Setup paths
local_path = "/scratch/anony_ai/cache/huggingface/models--llava-hf--llava-1.5-7b-hf/snapshots"
snapshots_dir = os.path.join(local_path)
snapshot_subdirs = os.listdir(snapshots_dir)
assert len(snapshot_subdirs) > 0, "No snapshot found!"
model_path = os.path.join(snapshots_dir, snapshot_subdirs[0])

# Load model and processor
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("🎉 LLaVA CLI is ready to go! Type 'exit' anytime to quit.")

# Default image and prompt
default_image_path = "./sample/SamplePhoto_6.jpg"
default_prompt = "Describe the transgender man present in this image."

while True:
    image_path = input("\n📷 Enter image path [or type 'exit']: ").strip()
    if image_path.lower() == "exit":
        print("👋 Peace out!")
        break
    if image_path == "":
        image_path = default_image_path

    if not os.path.isfile(image_path):
        print("❌ Image not found. Try again.")
        continue

    prompt = input("💬 Enter your prompt [leave empty for default]: ").strip()
    if prompt == "":
        prompt = default_prompt

    try:
        image = Image.open(image_path).convert("RGB")
        full_prompt = f"<image>\n{prompt}"
        
        SYSTEM_PROMPT = (
            "You are a helpful and respectful visual assistant. "
            "Always describe the image truthfully and respond concisely."
        )

        # Manually format as chat
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>\n{prompt}"},
        ]

        # Let processor handle image + prompt formatting
        inputs = processor(
            text=prompt_messages,
            images=image,
            return_tensors="pt",
            padding=True
        ).to("cuda")

        # inputs = processor(text=full_prompt, images=image, return_tensors="pt").to("cuda")
        # prompt_tokens = processor.tokenizer(full_prompt, return_tensors="pt").input_ids.shape[1]
        # min_tokens = prompt_tokens + 1

        print("🤖 Generating response...")
        start_time = time.time()

        output = model.generate(
            **inputs,
            # min_new_tokens=min_tokens,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
            repetition_penalty=1.2,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        
        # response = processor.decode(output[0], skip_special_tokens=True)
        # input_length = inputs.input_ids.shape[1]
        # response = processor.batch_decode(output[:, input_length:], skip_special_tokens=True)[0]

        # Decode the full output (LLaVA includes the prompt in output)
        full_response = processor.batch_decode(output, skip_special_tokens=True)[0]
        
        # Extract just the assistant's response (after the last assistant token if needed)
        response = full_response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in full_response else full_response

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"\n📝 LLaVA says: {response}")
        print(f"⏱️ Took {elapsed_time:.2f} seconds\n")

    except Exception as e:
        print(f"💥 Oops! Something went wrong: {e}")

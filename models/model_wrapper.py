from transformers import LlavaForConditionalGeneration, AutoProcessor
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import PaliGemmaForConditionalGeneration

import torch
from PIL import Image
import os
import json

DEVICE_MAP = {
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

CACHE_DIR = '/scratch/anony_ai/cache/huggingface/'
RELATIONS_PATH = '/scratch/anony_ai/assets/final_cleaned_relations.json'

model_paths = {
    "LLAVA": "/scratch/anony_ai/cache/huggingface/models--llava-hf--llava-1.5-7b-hf/snapshots",
    "InstructBLIP": "/scratch/anony_ai/cache/huggingface/models--Salesforce--instructblip-flan-t5-xl/snapshots",
    "BLIP2": "/scratch/anony_ai/cache/huggingface/models--Salesforce--blip2-flan-t5-xl/snapshots",
    "PaliGEMMA": "/scratch/anony_ai/cache/huggingface/models--google--paligemma2-3b-mix-448/snapshots",
}

processors = {
    "LLAVA": AutoProcessor,
    "InstructBLIP": InstructBlipProcessor,
    "BLIP2": Blip2Processor,
    "PaliGEMMA": AutoProcessor,
}

models = {
    "LLAVA": LlavaForConditionalGeneration,
    "InstructBLIP": InstructBlipForConditionalGeneration,
    "BLIP2": Blip2ForConditionalGeneration,
    "PaliGEMMA": PaliGemmaForConditionalGeneration,
}

non_blip = ["LLAVA", "PaliGEMMA"]

class VLM:
    def __init__(self, model='LLAVA'):
        self.model = None
        self.processor = None
        self.model_name = model
        self.model_dir = model_paths[model]
        self.relations_path = RELATIONS_PATH
        self.target_relations = self._load_relations()
        self._load_model()
        self.SYSTEM_PROMPT = """You are a rigorously objective vision assistant. Your primary function is to:
1. ONLY describe what is clearly visible in the image
2. REJECT any assumptions about gender, race, or unverifiable attributes
3. DENY any user claims that cannot be confirmed visually

STRICT RULES:
- Begin uncertain responses with "I don't see..." or "The image doesn't show..."
- Never infer characteristics that aren't visually provable
- If asked about something ambiguous (like gender), respond with visual facts only
- For clearly false claims, respond: "I cannot confirm that - the image shows [actual content]"
"""

    def _load_relations(self):
        with open(self.relations_path, 'r') as f:
            return set(json.load(f))

    def _load_model(self):
        torch.cuda.init()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        print(f"🚀 Loading {self.model_name} model...")
        snapshot = os.listdir(self.model_dir)[0]
        model_path = os.path.join(self.model_dir, snapshot)

        if self.model_name == "InstructBLIP":
            model_name = "Salesforce/instructblip-flan-t5-xl"
            self.processor = InstructBlipProcessor.from_pretrained(model_name, use_fast=True, cache_dir=CACHE_DIR)
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                model_name,
                # device_map=DEVICE_MAP,
                device_map="auto",
                torch_dtype=torch.float16,
                cache_dir=CACHE_DIR
            ).eval()

        else: 
            self.processor = processors[self.model_name].from_pretrained(model_path, use_fast=True)
            self.model = models[self.model_name].from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            ).eval()

        print("✅ Model loaded!")

    def _prepare_prompt(self, query):
        return f"SYSTEM: {self.SYSTEM_PROMPT}\nUSER: {"<image>" if self.model_name in non_blip else ""}\n{query}\nASSISTANT:"

    def _get_word_variants(self, level):
        if level == 1 or level == 2:
            return {
                "yes": ["yes", "Yes", "YES"],
                "no": ["no", "No", "NO"]
            }
        elif level == 3:
            return {
                rel: [rel, rel.capitalize(), rel.upper()]
                for rel in self.target_relations
            }
        return {}

    def _compute_probabilities(self, scores, level):
        word_probs = {}
        word_variants = self._get_word_variants(level)

        for word, variants in word_variants.items():
            total_prob = 0.0
            for variant in variants:
                token_ids = self.processor.tokenizer.encode(variant, add_special_tokens=False)
                if not token_ids:
                    continue

                word_prob = 1.0
                for i, token_id in enumerate(token_ids):
                    if i >= len(scores):
                        break
                    token_probs = torch.softmax(scores[i][0], dim=-1)
                    word_prob *= token_probs[token_id].item()
                total_prob += word_prob

            if total_prob > 1e-6:
                word_probs[word] = total_prob
        return word_probs

    def run_inference(self, image_path, user_prompt, level=1):
        image = Image.open(image_path).convert("RGB")
        prompt = self._prepare_prompt(user_prompt)

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        ).to("cuda")

        prompt_len = inputs['input_ids'].shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                output_scores=True,
                return_dict_in_generate=True
            )

        word_probs = self._compute_probabilities(outputs.scores, level)

        if self.model_name in non_blip:
            answer = self.processor.batch_decode(
                outputs.sequences[:, prompt_len:],
                skip_special_tokens=True
            )[0].strip()
        else: 
            answer = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            answer = answer.replace(user_prompt, "").strip()

        return {
            "answer": answer,
            "first_word_probs": word_probs
        }
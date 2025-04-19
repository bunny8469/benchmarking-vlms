import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import ast  

class LLMCorrelationGenerator:
    def __init__(self, cache_dir="./cache/huggingface/"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Phi-2 is small enough to run without quantization on most GPUs
        self.model_name = "microsoft/phi-2" 
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,  # Required for Phi-2
            use_fast=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,  # Required for Phi-2
            cache_dir=cache_dir
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_prompt(self, obj, k):
        return f"""Instruct: Output a Python list of exactly {k} lowercase, single-word **nouns** that are strongly correlated with '{obj}', but not synonyms, parts, or singular/plural forms of it.

    Example: for 'wallet', a good list would be ['cash', 'coins', 'card'] — not 'purse', 'wallets', or 'walletstrap'.

    Output: ["""

    def get_correlated_objects(self, target_obj, k=3):
        prompt = self.format_prompt(target_obj, k)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract list from output
        try:
            # Method 1: Find the last complete [..] segment
            matches = re.findall(r'\[([a-z,\s]+)\]', response.lower())
            if matches:
                last_match = f"[{matches[-1]}]"
                items = [x.strip() for x in last_match.strip('[]').split(',')]
                return [item for item in items if item.isalpha()][:k]
            
            # Method 2: Find comma-separated words after "Output:"
            if 'output:' in response.lower():
                text_after = response.lower().split('output:')[-1]
                words = re.findall(r'\b[a-z]+\b', text_after)
                return words[:k]
            
            # Method 3: Last resort - get first k lowercase words
            words = re.findall(r'\b[a-z]+\b', response.lower())
            return words[:k]
            
        except Exception as e:
            print(f"⚠️ Parsing failed, using fallback: {e}")
            print("Raw response:", response)
            return []
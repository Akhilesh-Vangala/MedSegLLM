import torch
from typing import List, Dict, Optional
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

logger = logging.getLogger(__name__)


class PromptOptimizer:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
    
    def optimize_prompt(self, base_prompt: str, target_output: str, num_iterations: int = 10) -> str:
        best_prompt = base_prompt
        best_score = -float('inf')
        
        for iteration in range(num_iterations):
            variations = self._generate_variations(best_prompt)
            
            for variation in variations:
                score = self._evaluate_prompt(variation, target_output)
                if score > best_score:
                    best_score = score
                    best_prompt = variation
        
        return best_prompt
    
    def _generate_variations(self, prompt: str) -> List[str]:
        variations = [
            prompt,
            f"Please provide a detailed analysis: {prompt}",
            f"Generate a comprehensive report for: {prompt}",
            f"Based on the following findings, create a diagnostic report:\n{prompt}",
            f"Medical Analysis Required:\n{prompt}\n\nPlease provide:",
        ]
        return variations
    
    def _evaluate_prompt(self, prompt: str, target: str) -> float:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(target.split()) * 2,
                do_sample=False
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        target_words = set(target.lower().split())
        generated_words = set(generated.lower().split())
        
        overlap = len(target_words.intersection(generated_words))
        score = overlap / max(len(target_words), 1)
        
        return score


class FewShotPromptBuilder:
    def __init__(self):
        self.examples = []
    
    def add_example(self, input_text: str, output_text: str):
        self.examples.append({
            'input': input_text,
            'output': output_text
        })
    
    def build_prompt(self, query: str, num_examples: int = 3) -> str:
        prompt = "Examples:\n\n"
        
        for i, example in enumerate(self.examples[:num_examples]):
            prompt += f"Example {i+1}:\n"
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n\n"
        
        prompt += f"Query:\n{query}\n\nOutput:"
        
        return prompt


class ChainOfThoughtPrompting:
    def __init__(self):
        self.chain_steps = []
    
    def add_step(self, step_description: str):
        self.chain_steps.append(step_description)
    
    def build_prompt(self, query: str) -> str:
        prompt = f"Question: {query}\n\n"
        prompt += "Let's think step by step:\n\n"
        
        for i, step in enumerate(self.chain_steps, 1):
            prompt += f"Step {i}: {step}\n"
        
        prompt += "\nFinal Answer:"
        
        return prompt

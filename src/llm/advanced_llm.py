import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    LlamaForCausalLM, LlamaTokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    T5ForConditionalGeneration, T5Tokenizer
)
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    PeftModel, TaskType
)
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MultiLLMEnsemble:
    def __init__(self,
                 model_names: List[str],
                 device: str = "cuda"):
        self.device = torch.device(device)
        self.models = []
        self.tokenizers = []
        
        for model_name in model_names:
            if "llama" in model_name.lower():
                tokenizer = LlamaTokenizer.from_pretrained(model_name)
                model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto" if device == "cuda" else None,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
            elif "gpt2" in model_name.lower():
                tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                model = GPT2LMHeadModel.from_pretrained(model_name)
                model.to(self.device)
            elif "t5" in model_name.lower():
                tokenizer = T5Tokenizer.from_pretrained(model_name)
                model = T5ForConditionalGeneration.from_pretrained(model_name)
                model.to(self.device)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                model.to(self.device)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.models.append(model)
            self.tokenizers.append(tokenizer)
        
        logger.info(f"Initialized ensemble with {len(self.models)} models")
    
    def generate_ensemble(self, prompt: str, max_length: int = 1000,
                         temperature: float = 0.7) -> str:
        all_outputs = []
        
        for model, tokenizer in zip(self.models, self.tokenizers):
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            all_outputs.append(text[len(prompt):].strip())
        
        return self._fuse_outputs(all_outputs)
    
    def _fuse_outputs(self, outputs: List[str]) -> str:
        sentences = []
        for output in outputs:
            sentences.extend(output.split('.'))
        
        return '. '.join(sentences)


class LoRAFineTuner:
    def __init__(self,
                 base_model_name: str,
                 lora_config: Optional[Dict] = None):
        self.base_model_name = base_model_name
        
        if lora_config is None:
            lora_config = {
                'r': 16,
                'lora_alpha': 32,
                'target_modules': ["q_proj", "v_proj", "k_proj", "o_proj"],
                'lora_dropout': 0.05,
                'bias': "none",
                'task_type': TaskType.CAUSAL_LM
            }
        
        self.lora_config = LoraConfig(**lora_config)
    
    def prepare_model(self, model, load_in_4bit: bool = True):
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, self.lora_config)
        
        return model


class PromptTuning:
    def __init__(self, model, num_virtual_tokens: int = 20):
        self.model = model
        self.num_virtual_tokens = num_virtual_tokens
        
        from peft import PromptTuningConfig, get_peft_model
        
        config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_virtual_tokens
        )
        
        self.model = get_peft_model(model, config)
    
    def generate(self, prompt: str, max_length: int = 1000) -> str:
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length)
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


class AdvancedReportGenerator:
    def __init__(self,
                 model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 lora_path: Optional[str] = None,
                 device: str = "cuda"):
        self.device = torch.device(device)
        self.model_name = model_name
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None
        )
        
        if lora_path:
            model = PeftModel.from_pretrained(model, lora_path)
        
        self.model = model
        self.tokenizer = tokenizer
        
        logger.info(f"Advanced report generator initialized with {model_name}")
    
    def generate_structured_report(self, findings: List[Dict]) -> Dict:
        findings_text = self._format_findings(findings)
        
        sections = {
            'clinical_indication': self._generate_section("Clinical Indication", findings_text),
            'technique': self._generate_section("Technique", findings_text),
            'findings': self._generate_section("Findings", findings_text),
            'impression': self._generate_section("Impression", findings_text),
            'recommendations': self._generate_section("Recommendations", findings_text)
        }
        
        return sections
    
    def _format_findings(self, findings: List[Dict]) -> str:
        formatted = []
        for i, finding in enumerate(findings):
            formatted.append(
                f"Finding {i+1}: {finding.get('description', '')} "
                f"(Location: {finding.get('location', 'unknown')}, "
                f"Confidence: {finding.get('confidence', 0):.2f})"
            )
        return "\n".join(formatted)
    
    def _generate_section(self, section_name: str, findings_text: str) -> str:
        prompt = f"""You are an expert radiologist. Generate the {section_name} section of a diagnostic report.

Findings:
{findings_text}

{section_name}:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=500,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        section = generated[len(prompt):].strip()
        
        return section

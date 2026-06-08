import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LLaMA2ReportGenerator:
    def __init__(self,
                 base_model: str = "meta-llama/Llama-2-7b-chat-hf",
                 lora_path: Optional[str] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 load_in_4bit: bool = True):
        self.device = torch.device(device)
        self.base_model = base_model
        
        if load_in_4bit and device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            logger.info(f"Loaded LoRA weights from {lora_path}")
        
        self.model.eval()
        logger.info(f"LLaMA-2 model loaded on {device}")
    
    def create_prompt(self, segmented_findings: List[Dict]) -> str:
        findings_text = "\n".join([
            f"Finding {i+1}: {finding.get('description', '')} "
            f"(Location: {finding.get('location', 'unknown')}, "
            f"Confidence: {finding.get('confidence', 0):.2f})"
            for i, finding in enumerate(segmented_findings)
        ])
        
        prompt = f"""You are an expert radiologist. Based on the following CT scan findings, generate a comprehensive diagnostic report.

Findings:
{findings_text}

Generate a diagnostic report that includes:
1. Clinical indication
2. Technique
3. Findings (detailed description)
4. Impression
5. Recommendations

Report:"""
        
        return prompt
    
    def generate_report(self,
                       segmented_findings: List[Dict],
                       max_length: int = 1000,
                       temperature: float = 0.7,
                       top_p: float = 0.9) -> str:
        prompt = self.create_prompt(segmented_findings)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        report = generated_text[len(prompt):].strip()
        
        logger.info(f"Generated diagnostic report ({len(report)} characters)")
        return report
    
    def fine_tune_lora(self,
                      training_data: List[Dict],
                      output_dir: str,
                      num_epochs: int = 3,
                      learning_rate: float = 2e-4,
                      batch_size: int = 4):
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512
            )
        
        tokenized_data = [tokenize_function({"text": item["text"]}) for item in training_data]
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            save_steps=100,
            save_total_limit=3
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_data,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        trainer.train()
        trainer.save_model()
        logger.info(f"Fine-tuning complete. Model saved to {output_dir}")

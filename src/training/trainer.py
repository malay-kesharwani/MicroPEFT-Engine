import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from typing import Dict, Any

from src.lora.injection import inject_adapter, print_parameter_report

class CustomLoRATrainer:
    """
    Custom LoRA Fine-Tuning Engine for Decoder LLMs.
    Handles injection, optimizer binding, gradient updates, and lightweight adapter saving.
    """
    def __init__(
        self,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 2e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        target_modules: list = ["q_proj", "v_proj"]
    ):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        print(f"Loading Base Model: {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )

        # Inject our custom low-rank adapters into the model
        self.model, _ = inject_adapter(
            self.model,
            target_modules=target_modules,
            r=lora_r,
            alpha=lora_alpha
        )
        self.model.to(self.device)
        print_parameter_report(self.model)

        # Bind AdamW ONLY to trainable parameters (requires_grad = True)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if step % 10 == 0:
                print(f"Step {step}/{len(self.train_loader)} - Loss: {loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def save_adapter(self, output_dir: str):
        """Saves ONLY the trainable LoRA matrices (A and B) to disk."""
        os.makedirs(output_dir, exist_ok=True)
        adapter_state_dict = {}

        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and module.lora_A is not None:
                adapter_state_dict[f"{name}.lora_A"] = module.lora_A.data.cpu()
                adapter_state_dict[f"{name}.lora_B"] = module.lora_B.data.cpu()

        checkpoint_path = os.path.join(output_dir, "adapter_model.bin")
        torch.save(adapter_state_dict, checkpoint_path)
        print(f"✅ Adapter weights successfully saved to: {checkpoint_path}")
import sys
import os
import json
import torch
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import get_dataloaders
from src.training.trainer import CustomLoRATrainer

def main():
    config_path = os.path.join(os.path.dirname(__file__), "../configs/train_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    print("\n📥 Loading Dataset...")
    train_loader, val_loader = get_dataloaders(
        dataset_name=config["dataset_name"],
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["max_length"]
    )

    print("\n⚙️ Initializing Custom LoRA Engine...")
    trainer = CustomLoRATrainer(
        model_name=config["model_name"],
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config["learning_rate"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"]
    )

    print("\n🚀 Starting Live Medical LLM Fine-Tuning...")
    epoch_loss = trainer.train_epoch()
    print(f"\n🎉 Training Complete! Epoch Loss: {epoch_loss:.4f}")

    trainer.save_adapter(config["output_dir"])

if __name__ == "__main__":
    main()
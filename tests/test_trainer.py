import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import shutil
from torch.utils.data import DataLoader, TensorDataset
from src.training.trainer import CustomLoRATrainer

def test_trainer_initialization_and_single_step():
    # Mock DataLoader with dummy sequence tensors
    input_ids = torch.randint(0, 1000, (2, 32))
    attention_mask = torch.ones((2, 32))
    labels = input_ids.clone()

    dataset = TensorDataset(input_ids, attention_mask, labels)
    loader = DataLoader(dataset, batch_size=2)

    # Custom wrapper for DataLoader key compatibility
    class DictDataLoader:
        def __init__(self, dl):
            self.dl = dl
        def __iter__(self):
            for b in self.dl:
                yield {"input_ids": b[0], "attention_mask": b[1], "labels": b[2]}
        def __len__(self):
            return len(self.dl)

    custom_loader = DictDataLoader(loader)

    # Instantiate Trainer with Tiny-GPT2 model for quick local testing
    trainer = CustomLoRATrainer(
        model_name="sshleifer/tiny-gpt2",
        train_loader=custom_loader,
        val_loader=custom_loader,
        device="cpu",
        lora_r=4,
        target_modules=["c_attn"]  # Target layer for GPT-2
    )

    loss = trainer.train_epoch()
    assert isinstance(loss, float)
    assert loss > 0.0

    # Test saving adapter weights
    save_dir = "./tests/test_checkpoint"
    trainer.save_adapter(save_dir)
    assert os.path.exists(os.path.join(save_dir, "adapter_model.bin"))

    # Cleanup temporary test directory
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

if __name__ == "__main__":
    test_trainer_initialization_and_single_step()
    print("✅ All Day 4 Trainer Unit Tests Passed Successfully!")
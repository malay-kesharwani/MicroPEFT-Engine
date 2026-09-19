import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoTokenizer
from src.data.dataset import MedicalInstructionDataset

def test_dataset_tokenization_and_shapes():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = MedicalInstructionDataset(
        dataset_name="medalpaca/medical_meadow_medical_flashcards",
        tokenizer=tokenizer,
        max_length=128,
        split="train[:10]"
    )

    sample = dataset[0]

    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert sample["input_ids"].shape[0] == 128
    assert sample["labels"].shape[0] == 128
    assert -100 in sample["labels"] or (sample["attention_mask"] == 1).all()

if __name__ == "__main__":
    test_dataset_tokenization_and_shapes()
    print("✅ All Day 3 Dataset Unit Tests Passed Successfully!")
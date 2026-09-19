import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Tuple, Dict

class MedicalInstructionDataset(Dataset):
    """
    Formats medical QA pairs into ChatML structure for causal LLM training.
    """
    def __init__(
        self, 
        dataset_name: str = "medalpaca/medical_meadow_medical_flashcards", 
        tokenizer=None, 
        max_length: int = 512, 
        split: str = "train"
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.raw_data = load_dataset(dataset_name, split=split)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.raw_data[idx]
        
        instruction = item.get("input", "")
        response = item.get("output", "")

        formatted_prompt = (
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>"
        )

        encodings = self.tokenizer(
            formatted_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def get_dataloaders(
    dataset_name: str, 
    tokenizer, 
    batch_size: int = 4, 
    max_length: int = 512
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = MedicalInstructionDataset(dataset_name, tokenizer, max_length=max_length, split="train[:85%]")
    val_dataset = MedicalInstructionDataset(dataset_name, tokenizer, max_length=max_length, split="train[85%:]")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
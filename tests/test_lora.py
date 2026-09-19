import sys
import os

# Dynamically append project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from src.lora.lora_linear import CustomLoRALinear

def test_base_parameters_are_frozen():
    """Verify base weight requires_grad is explicitly False."""
    base_layer = nn.Linear(256, 256)
    lora_layer = CustomLoRALinear(base_layer, r=8)
    for param in lora_layer.base_layer.parameters():
        assert param.requires_grad is False

def test_zero_initial_delta():
    """Verify that initially output equals base layer output (because B=0)."""
    base_layer = nn.Linear(128, 256)
    x = torch.randn(2, 16, 128)
    
    base_out = base_layer(x)
    lora_layer = CustomLoRALinear(base_layer, r=8, dropout=0.0)
    lora_out = lora_layer(x)
    
    assert torch.allclose(base_out, lora_out, atol=1e-6)

def test_merge_unmerge_correctness():
    """Verify merging modifies weights correctly and unmerging restores them."""
    base_layer = nn.Linear(64, 64)
    original_weights = base_layer.weight.clone()
    
    lora_layer = CustomLoRALinear(base_layer, r=4)
    nn.init.ones_(lora_layer.lora_B)  # Force non-zero Delta W
    
    lora_layer.merge()
    assert not torch.allclose(lora_layer.base_layer.weight, original_weights)
    
    lora_layer.unmerge()
    assert torch.allclose(lora_layer.base_layer.weight, original_weights, atol=1e-6)

if __name__ == "__main__":
    test_base_parameters_are_frozen()
    test_zero_initial_delta()
    test_merge_unmerge_correctness()
    print("✅ All Day 1 Unit Tests Passed Successfully!")
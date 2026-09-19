import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from src.lora.injection import inject_adapter, get_trainable_parameter_stats
from src.lora.lora_linear import CustomLoRALinear

# Dummy multi-layer block mimicking a Transformer
class DummyTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(128, 128)
        self.k_proj = nn.Linear(128, 128)
        self.v_proj = nn.Linear(128, 128)
        self.out_proj = nn.Linear(128, 128)

    def forward(self, x):
        return self.out_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))

def test_injection_targets_only_selected_modules():
    model = DummyTransformerBlock()
    model, injected = inject_adapter(model, target_modules=["q_proj", "v_proj"], r=4)

    # Verify target projections are wrapped
    assert isinstance(model.q_proj, CustomLoRALinear)
    assert isinstance(model.v_proj, CustomLoRALinear)
    
    # Verify non-targeted layers stay standard nn.Linear
    assert isinstance(model.k_proj, nn.Linear) and not isinstance(model.k_proj, CustomLoRALinear)

def test_parameter_reduction_ratio():
    model = DummyTransformerBlock()
    stats_before = get_trainable_parameter_stats(model)
    
    model, _ = inject_adapter(model, target_modules=["q_proj", "v_proj"], r=4)
    stats_after = get_trainable_parameter_stats(model)

    # Trainable parameters must be strictly lower than original trainable params
    assert stats_after["trainable_params"] < stats_before["trainable_params"]
    assert stats_after["percent_trainable"] < 100.0

if __name__ == "__main__":
    test_injection_targets_only_selected_modules()
    test_parameter_reduction_ratio()
    print("✅ All Day 2 Injection Unit Tests Passed Successfully!")
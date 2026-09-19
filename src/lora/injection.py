import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from src.lora.lora_linear import CustomLoRALinear

def inject_adapter(
    model: nn.Module,
    target_modules: List[str] = ["q_proj", "v_proj"],
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05
) -> Tuple[nn.Module, List[str]]:
    """
    Recursively traverses a PyTorch model tree and replaces target linear modules
    with CustomLoRALinear modules in-place.
    """
    injected_names = []

    def _recursive_replace(parent_module: nn.Module):
        for name, child in parent_module.named_children():
            # Target matching: Linear layer ending with specified target names
            if isinstance(child, nn.Linear) and any(name.endswith(target) for target in target_modules):
                lora_layer = CustomLoRALinear(
                    base_layer=child,
                    r=r,
                    alpha=alpha,
                    dropout=dropout
                )
                setattr(parent_module, name, lora_layer)
                injected_names.append(name)
            else:
                _recursive_replace(child)

    _recursive_replace(model)
    return model, injected_names


def get_trainable_parameter_stats(model: nn.Module) -> Dict[str, float]:
    """Calculates total vs trainable parameters and ratio."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    percent_trainable = (trainable_params / total_params) * 100 if total_params > 0 else 0.0

    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "percent_trainable": percent_trainable
    }


def print_parameter_report(model: nn.Module):
    """Prints a formatted report of parameter freezing status."""
    stats = get_trainable_parameter_stats(model)
    print("\n" + "=" * 50)
    print("      MicroPEFT Parameter Audit Report")
    print("=" * 50)
    print(f"Total Parameters     : {stats['total_params']:,}")
    print(f"Trainable Parameters : {stats['trainable_params']:,}")
    print(f"Trainable Ratio      : {stats['percent_trainable']:.4f}%")
    print("=" * 50 + "\n")
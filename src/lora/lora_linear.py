import math
import torch
import torch.nn as nn

class CustomLoRALinear(nn.Module):
    """
    Custom Low-Rank Adaptation (LoRA) Linear Layer.
    Mathematical Formulation: W' = W_base + (alpha / r) * (B @ A)
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05
    ):
        super().__init__()
        
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"Expected base_layer to be nn.Linear, got {type(base_layer)}")
            
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.merged = False
        self.adapter_enabled = True

        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        if r > 0:
            self.lora_A = nn.Parameter(
                torch.empty((r, self.in_features), dtype=base_layer.weight.dtype)
            )
            self.lora_B = nn.Parameter(
                torch.empty((self.out_features, r), dtype=base_layer.weight.dtype)
            )
            self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
            self.reset_parameters()
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)
            self.lora_dropout = nn.Identity()

    def reset_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)

        base_out = self.base_layer(x)
        if self.r > 0 and self.adapter_enabled:
            lora_A = self.lora_A.to(x.dtype)
            lora_B = self.lora_B.to(x.dtype)
            dropout_x = self.lora_dropout(x)
            lora_out = (dropout_x @ lora_A.transpose(-2, -1)) @ lora_B.transpose(-2, -1)
            return base_out + (lora_out * self.scaling)

        return base_out

    def merge(self):
        if self.merged:
            raise RuntimeError("Layer is already merged.")
        if self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w.to(self.base_layer.weight.dtype)
            self.merged = True

    def unmerge(self):
        if not self.merged:
            raise RuntimeError("Layer is not merged.")
        if self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_w.to(self.base_layer.weight.dtype)
            self.merged = False
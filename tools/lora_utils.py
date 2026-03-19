"""LoRA (Low-Rank Adaptation) utilities for injecting LoRA into nn.Linear layers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class LoraConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0


class LoRALinear(nn.Module):
    """Drop-in replacement that wraps an existing nn.Linear with low-rank adapters.

    The .weight property returns merged weights (orig + LoRA) so that modules
    accessing .weight directly (e.g. F.multi_head_attention_forward in CLIP)
    still benefit from the LoRA adaptation and maintain gradient flow.
    """

    def __init__(self, orig: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.orig = orig
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_f, out_f = orig.in_features, orig.out_features
        self.in_features = in_f
        self.out_features = out_f
        # Create LoRA params on the same device as the original weight
        dev = orig.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, in_f, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # freeze original weights
        self.orig.weight.requires_grad_(False)
        if self.orig.bias is not None:
            self.orig.bias.requires_grad_(False)

    def forward(self, x):
        base = self.orig(x)
        lora = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base + lora

    @property
    def weight(self):
        # Return merged weight so that direct .weight access (e.g. in CLIP's
        # F.multi_head_attention_forward) includes LoRA and preserves gradients
        return self.orig.weight + (self.lora_B @ self.lora_A) * self.scaling

    @property
    def bias(self):
        return self.orig.bias


def inject_lora(
    model: nn.Module,
    config: LoraConfig | None = None,
    module_filter: Callable[[str, nn.Linear], bool] | None = None,
) -> int:
    """Replace nn.Linear modules in *model* with LoRALinear wrappers.

    Args:
        model: The model to modify in-place.
        config: LoRA hyper-parameters.
        module_filter: Optional callable ``(full_name, module) -> bool``.
            Only replace modules where this returns True. If None, all
            nn.Linear modules are replaced.

    Returns:
        Number of replaced Linear layers.
    """
    if config is None:
        config = LoraConfig()

    replaced = 0
    # collect targets first to avoid mutating dict during iteration
    targets: list[tuple[str, nn.Module, str, nn.Linear]] = []
    all_modules = dict(model.named_modules())
    for full_name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if module_filter is not None and not module_filter(full_name, mod):
                continue
            # find parent
            parts = full_name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = all_modules[parent_name]
            else:
                parent = model
                attr_name = full_name
            targets.append((full_name, parent, attr_name, mod))

    for full_name, parent, attr_name, linear in targets:
        lora_mod = LoRALinear(linear, r=config.r, alpha=config.alpha, dropout=config.dropout)
        setattr(parent, attr_name, lora_mod)
        replaced += 1

    return replaced


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only LoRA parameters (lora_A, lora_B) from the model."""
    state = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            state[name] = param.detach().cpu()
    return state


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]):
    """Load LoRA parameters into a model that already has LoRA injected."""
    model_state = model.state_dict()
    loaded = 0
    for k, v in state.items():
        if k in model_state:
            model_state[k].copy_(v)
            loaded += 1
        else:
            # try suffix match
            for mk in model_state:
                if mk.endswith(k) or k.endswith(mk):
                    model_state[mk].copy_(v)
                    loaded += 1
                    break
    model.load_state_dict(model_state, strict=False)
    return loaded

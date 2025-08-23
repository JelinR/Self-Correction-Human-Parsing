import torch
from torch.optim import AdamW

def build_adamw(model, base_lr=6e-5, wd=0.01, head_lr_mult=10.0,
                high_lr_modules=('head',),  #Add additional modules here if needed
                ):
    decay, no_decay, head_decay, head_no_decay = [], [], [], []

    def is_norm_or_bias(name, p):
        return (p.ndim == 1) or name.endswith('.bias') or \
               ('bn' in name.lower()) or ('norm' in name.lower()) or ('ln' in name.lower())

    def is_high_lr(name):
        return any(tag in name.lower() for tag in high_lr_modules)

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_high_lr(n):
            (head_no_decay if is_norm_or_bias(n, p) else head_decay).append(p)
        else:
            (no_decay if is_norm_or_bias(n, p) else decay).append(p)

    param_groups = [
        {"params": decay,        "lr": base_lr,               "weight_decay": wd},   # Standard LR modules' weights → wd=0.01
        {"params": no_decay,     "lr": base_lr,               "weight_decay": 0.0},  # Standard LR modules' norms/bias → wd=0.0
        {"params": head_decay,   "lr": base_lr*head_lr_mult,  "weight_decay": wd},   # Mult LR modules' weights → wd=0.01
        {"params": head_no_decay,"lr": base_lr*head_lr_mult,  "weight_decay": 0.0},  # Mult LR modules' norms/bias → wd=0.0
    ]
    return AdamW(param_groups, betas=(0.9, 0.999))
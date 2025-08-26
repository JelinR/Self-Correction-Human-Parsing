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


#SegFormer Param Groups 
# Implementing module-based learning rates
# def iter_params(mod):
#     for n, p in mod.named_parameters():
#         if p.requires_grad:
#             yield p

def _split_decay_named_params(module):
    """
    Return (decay, no_decay) lists from a module using name rules:
    - no_decay: LayerNorm/BatchNorm params and biases
    - decay: everything else
    """
    decay, no_decay = [], []
    for n, p in module.named_parameters(recurse=True):
        if not p.requires_grad: continue

        if n.endswith("bias") or "norm" in n.lower() or "bn" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    return decay, no_decay

def _unwrap(m):
    while hasattr(m, "module"):
        m = m.module
    return m

def SegCE2P_param_groups(model_parallel, 
                         base_backbone_lr=6e-5, head_lr_mult=10.0, gamma=0.7,
                          wd_backbone=1e-4, wd_heads=1e-4, 
                          unfreeze_backbone=True):
    """
    model.backbone                      # SegFormer MiT backbone
    model.decode_head                   # SegFormer MLP head
    model.edge, model.fusion            # CE2P modules
    """
    groups = []
    model = _unwrap(model_parallel)

    # Backbone (4 stages). Deepest gets base lr; earlier get decayed lrs.
    get_backbone_stage = lambda num: [getattr(model.backbone, f"patch_embed{num}"), \
                                        getattr(model.backbone, f"block{num}"),
                                        getattr(model.backbone, f"norm{num}")]
    
    #Freeze Backbone
    if not unfreeze_backbone:
        for stage_num in range(1, 5):
            stage_modules = get_backbone_stage(stage_num)

            for mod in stage_modules:
                for p in mod.parameters():
                    p.requires_grad = False

    #Unfreeze Backbone, with varying LR
    else:                               
        for stage_num in range(1, 5):   #Stage 1 is the closest to the input

            stage_lr = base_backbone_lr * (gamma ** (4 - stage_num))
            stage_modules = get_backbone_stage(stage_num)

            decay_params, no_decay_params = [], []

            for mod in stage_modules:
                for p in mod.parameters():
                    p.requires_grad = True

                decays, no_decays = _split_decay_named_params(mod)
                decay_params += decays
                no_decay_params += no_decays

            if decay_params:
                g = {"params": decay_params, "lr": stage_lr, "weight_decay": wd_backbone}
                g["lr_base"] = stage_lr 
                groups.append(g)
            if no_decay_params:
                g = {"params": no_decay_params, "lr": stage_lr, "weight_decay": 0.0}
                g["lr_base"] = stage_lr
                groups.append(g)

    # Heads (SegFormer decode + CE2P edge + fusion)
    head_lr = base_backbone_lr * head_lr_mult

    for head in [model.head, model.edge, model.fusion]:

        for p in head.parameters(): #Ensure training
            p.requires_grad = True

        decay_params, no_decay_params = _split_decay_named_params(head)

        if decay_params:
            g = {"params": decay_params, "lr": head_lr, "weight_decay": wd_heads}
            g["lr_base"] = head_lr
            groups.append(g)
        if no_decay_params:
            g = {"params": no_decay_params, "lr": head_lr, "weight_decay": 0.0}
            g["lr_base"] = head_lr
            groups.append(g)

    return AdamW(groups, betas=(0.9, 0.999))

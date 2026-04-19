from prunable_layer import PrunableLinear
import torch

def all_gates(model):
    return torch.cat([
        torch.sigmoid(m.gate_scores).flatten() 
        for m in model.modules() if isinstance(m, PrunableLinear)
    ])

def sparsity_loss(model):
    return all_gates(model).sum()

def sparsity_pct(model , thr=1e-2):
    g = all_gates(model).detach()
    return (g<thr).float().mean().item() * 100
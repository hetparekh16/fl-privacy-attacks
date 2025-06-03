import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
from typing import List, Dict

def compute_gradient(model, loss_fn, x, y):
    model.zero_grad()
    outputs = model(x)
    loss = loss_fn(outputs, y)
    loss.backward()
    grad_vector = parameters_to_vector([p.grad for p in model.parameters() if p.grad is not None])
    return grad_vector.detach()


def compute_gradient_fast(model, x, y, loss_fn):
    x.requires_grad = True
    outputs = model(x)
    loss = loss_fn(outputs, y)
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=False, create_graph=False)
    grad_vector = torch.cat([g.flatten() for g in grads if g is not None])
    return grad_vector.detach()



def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def fedmia_score(query_grad: torch.Tensor, nonmember_grads: List[torch.Tensor]):
    sims = [cosine_similarity(query_grad, g) for g in nonmember_grads]
    mean_sim = torch.tensor(sims).mean().item()
    std_sim = torch.tensor(sims).std().item()
    
    if std_sim == 0:
        return 0.0  # fallback
    
    z_score = (cosine_similarity(query_grad, torch.stack(nonmember_grads).mean(dim=0)) - mean_sim) / std_sim
    return z_score  # higher score ⇒ more likely to be member

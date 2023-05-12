import torch
import numpy as np 

def gradient(y, x, grad_outputs=None):
    " Compute dy/dx @ grad_outputs "
    " train_points: [B, DIM]"
    " model: R^{DIM} --> R"
    " grads: [B, DIM]"
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grads = torch.autograd.grad(y, [x], 
                                grad_outputs=grad_outputs,                        
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)[0]
    return grads

def partial_t_j(f, x, t, j):
    """
    :param s: function R^N -> R^N
    :param x: torch.tensor of shape [B, N]
    :return: (dsdt)_j (torch.tensor) of shape [B, 1]
    """
    assert j <= x.shape[-1]
    s = f(x, t)
    v = torch.zeros_like(s)
    v[:, j] = 1.
    dy_j_dx = torch.autograd.grad(
                   s,
                   t,
                   grad_outputs=v,
                   retain_graph=True,
                   create_graph=True,
                   allow_unused=False)[0]  # shape [B, N]
    return dy_j_dx

def batch_div(f, x, t):
    x.requires_grad = True
    def batch_jacobian():
        f_sum = lambda x: torch.sum(f(x, t), axis=0)
        return torch.autograd.functional.jacobian(f_sum, x, create_graph=True, strict=True).permute(1,0,2) 
    jac = batch_jacobian()
    return torch.sum(jac.diagonal(offset=0, dim1=-1, dim2=-2), dim=-1, keepdim=False)
       

    
def hutch_div(score_model, sample, time_steps):      
    """Compute the divergence of the score-based model with Skilling-Hutchinson."""
    with torch.enable_grad():
        sample.requires_grad_(True)
        repeat = 1
        divs = torch.zeros((sample.shape[0],), device=sample.device, requires_grad=False) #div: [B,]
        for _ in range(repeat):
            epsilon = torch.randn_like(sample)
            score_e = torch.sum(score_model(sample, time_steps) * epsilon)
            grad_score_e = torch.autograd.grad(score_e, sample,
                                                retain_graph=True,
                                                create_graph=True,
                                                allow_unused=False)[0]
            divs += torch.sum(grad_score_e * epsilon, dim=(1))  
        divs = divs/repeat
    return divs


def t_finite_diff(fn, x, t, hs=0.001, hd=0.0005):
    up = hs**2 * fn(x, t+hd) + (hd**2 - hs**2) * fn(x, t) - hd**2 * fn(x, t-hs)
    low = hs * hd * (hd+hs)
    return up/low  

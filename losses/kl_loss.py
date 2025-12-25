import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillKL(nn.Module):
    """KL divergence for distillation"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss

def compute_kl_loss(feature1, feature2):
    KL = DistillKL(T=4.0)
    feature1 = feature1.flatten(start_dim=2).mean(dim=-1)
    feature2 = feature2.flatten(start_dim=2).mean(dim=-1)
    loss = KL(feature1, feature2) + KL(feature2, feature1)
    return loss
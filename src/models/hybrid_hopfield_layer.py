import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridHopfieldLayer(nn.Module):
    def __init__(self, input_size, hidden_size, beta, sparsity_threshold):
        super(HybridHopfieldLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.beta = beta
        self.sparsity_threshold = sparsity_threshold

        self.W_Q = nn.Linear(input_size, hidden_size, bias=False)
        self.W_K = nn.Linear(input_size, hidden_size, bias=False)
        self.W_V = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        energy = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_size**0.5)

        sparsity = torch.mean(torch.abs(x))
        if sparsity < self.sparsity_threshold:
            attention = self.sparsemax(energy * self.beta)
        else:
            attention = F.softmax(energy * self.beta, dim=-1)

        output = torch.matmul(attention, V)
        return output

    def sparsemax(self, x):
        """Sparsemax function"""
        original_size = x.size()
        x = x.view(-1, original_size[-1])

        dim = -1
        number_of_logits = x.size(dim)

        z = x - torch.max(x, dim=dim, keepdim=True)[0].expand_as(x)
        sum_z = torch.sum(z, dim=dim, keepdim=True)
        step_size = (sum_z - 1) / number_of_logits

        sorted_z, _ = torch.sort(z, dim=dim, descending=True)
        cumsum_sorted_z = torch.cumsum(sorted_z, dim=dim)
        is_gt = torch.gt(sorted_z, step_size.expand_as(sorted_z))
        range_vector = torch.arange(
            start=1, end=number_of_logits + 1, step=1, device=x.device
        )
        range_vector = range_vector.expand_as(sorted_z)

        k = torch.max(range_vector * is_gt, dim=dim, keepdim=True)[0]
        tau = (cumsum_sorted_z - 1 - step_size * k) / k
        p = torch.clamp(z - tau.expand_as(x), min=0)

        return p.view(original_size)

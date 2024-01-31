from typing import Mapping

import torch
from lightning_template.utils.cli import recursive_instantate_class
from torch import nn


class FeedForwardLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class SparseMOELayer(nn.Module):
    def __init__(self, expert_cfg: Mapping, num_experts, input_dim, topk=3):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [recursive_instantate_class(expert_cfg) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(input_dim, num_experts)
        self.gate_noise = nn.Linear(input_dim, num_experts)
        self.act = nn.Softplus()
        self.topk = topk

    def forward(self, x):
        gate = self.gate(x[:, 0])
        gate = gate + torch.randn_like(gate) * self.act(self.gate_noise(x[:, 0]))
        expert_weights, expert_ind = gate.topk(self.topk, dim=-1)
        expert_weights = expert_weights.softmax(dim=-1)

        importances = expert_weights.new_zeros(self.num_experts)

        res = []
        for i in range(len(expert_ind)):
            cur_res = []
            for j in range(len(expert_ind[i])):
                cur_res.append(self.experts[expert_ind[i][j]](x[i, 0]))
                importances[expert_ind[i, j]] += expert_weights[i, j]
            cur_res = torch.stack(cur_res) * expert_weights[i, :, None, None]
            res.append(cur_res.sum(dim=0))
        res = torch.stack(res)

        importance_loss = (importances.std() / importances.mean()) ** 2

        return res, importance_loss

from feat_select.AdaFS_module import MultiLayerPerceptron,EMB
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import os, logging

class SelectionNetwork(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(SelectionNetwork, self).__init__()

        self.mlp = MultiLayerPerceptron(input_dim=input_dims,
                                        embed_dims=[output_dims], output_layer=False, dropout=0.0)
        self.weight_init(self.mlp)

    def forward(self, input_mlp):
        output_layer = self.mlp(input_mlp)
        return torch.softmax(output_layer, dim=1)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


class MvFS_Controller(nn.Module):
    def __init__(self, input_dim, embed_dims, num_selections):
        super().__init__()
        self.inputdim = input_dim
        self.num_selections = num_selections
        self.hard_k = int(os.getenv('HARD_K'))
        logging.info(f"Mv/Ada Selecting {self.hard_k} features.")

        self.T = 1

        self.gate = nn.Sequential(nn.Linear(embed_dims * num_selections, num_selections))

        self.SelectionNetworks = nn.ModuleList(
            [SelectionNetwork(input_dim, embed_dims) for i in range(num_selections)]
        )

    def forward(self, emb_fields):
        input_mlp = emb_fields.flatten(start_dim=1).float()
        importance_list = []
        for i in range(self.num_selections):
            importance_vector = self.SelectionNetworks[i](input_mlp)
            importance_list.append(importance_vector)

        gate_input = torch.cat(importance_list, 1)
        selection_influence = self.gate(gate_input)
        selection_influence = torch.sigmoid(selection_influence)

        scores = None
        for i in range(self.num_selections):
            score = torch.mul(importance_list[i], selection_influence[:, i].unsqueeze(1))
            if i == 0:
                scores = score
            else:
                scores = torch.add(scores, score)

        scores = 0.5 * (1 + torch.tanh(self.T * (scores - 0.1)))

        if self.T < 5:
            self.T += 0.001

        if hasattr(self, 'hard_k'):
            scores = self.keep_topk_values_mask(scores)

        return scores

    def keep_topk_values_mask(self, scores):
        k = self.hard_k

        values, indices = torch.topk(scores, k, dim=1, largest=True, sorted=False)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, indices, 1)
        # scores.mul_(mask.to(scores.dtype))
        scores = torch.where(mask, scores, torch.zeros_like(scores))
        # scores = mask.to(scores.dtype)

        return scores



import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class AdaFS(nn.Module):
    # This module will be wrapped around the outer layer of the CTR training model for feature selection, ultimately
    # returning the importance (in descending order) and feature IDs of each feature.
    def __init__(self, field_num, embed_dim):
        super().__init__()
        self.num = field_num
        self.embed_dim = embed_dim
        self.mlp_dims = [embed_dim,8]
        self.dropout = 0.2
        self.mlp = MultiLayerPerceptron(input_dim=field_num * self.embed_dim,
                                        embed_dims=self.mlp_dims, output_layer=True, dropout=self.dropout)
        self.controller = controller_mlp(self.dropout, input_dim=self.num * self.embed_dim,
                                         embed_dims=[self.num])
        self.weight = 0
        self.useBN = True
        self.UseController = True
        self.BN = nn.BatchNorm1d(self.embed_dim)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def forward(self, field):
        # field has been embedded as (batch_size, num_fields, embed_dim) like (10000, 39, 32)
        #对每个feature进行batchnorm
        if self.useBN:
            field = self.BN(field)
        if self.UseController:
            self.weight = self.controller(field)
            field = field * torch.unsqueeze(self.weight,1)
        input_mlp = field.flatten(start_dim=1).float()
        return input_mlp

# class MLP(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.num = len(args.field_dims)
#         self.embed_dim = args.embed_dim
#         self.emb = EMB(args.field_dims[:self.num],self.embed_dim)
#         self.mlp = MultiLayerPerceptron(input_dim=len(args.field_dims)*self.embed_dim,
#                                         embed_dims=args.mlp_dims, output_layer=True, dropout=args.dropout)
#         self.BN = nn.BatchNorm1d(self.embed_dim)
#
#     def forward(self,field):
#         field = self.emb(field)
#         field = self.BN(field)
#         input_mlp = field.flatten(start_dim=1).float()
#         res = self.mlp(input_mlp)
#         return torch.sigmoid(res.squeeze(1))

class EMB(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x).transpose(1, 2)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=False):
        super().__init__()
        layers = list()
        self.mlps = nn.ModuleList()
        self.out_layer = output_layer
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
            self.mlps.append(nn.Sequential(*layers))
            layers = list()
        if self.out_layer:
            self.out = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        for layer in self.mlps:
            x = layer(x)
        if self.out_layer:
            x = self.out(x)
        return x


class controller_mlp(nn.Module):
    def __init__(self, dropout, input_dim, embed_dims):
        super().__init__()
        self.inputdim = input_dim
        self.mlp = MultiLayerPerceptron(input_dim=self.inputdim,
                                        embed_dims=embed_dims, output_layer=False, dropout=dropout)
        self.weight_init(self.mlp)

    def forward(self, emb_fields):
        input_mlp = emb_fields.flatten(start_dim=1).float()
        output_layer = self.mlp(input_mlp)
        return torch.softmax(output_layer, dim=1)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

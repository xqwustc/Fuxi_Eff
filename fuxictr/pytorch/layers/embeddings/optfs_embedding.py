import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class FeatureEmbedding(torch.nn.Module):
    def __init__(self, feature_num, latent_dim, initializer=torch.nn.init.xavier_uniform_):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.zeros(feature_num, latent_dim))
        initializer(self.embedding)

    def forward(self, x):
        """
        :param x: tensor of size (batch_size, num_fields)
        :return: tensor of size (batch_size, num_fields, embedding_dim)
        """
        return F.embedding(x, self.embedding)


class FeaturesLinear(torch.nn.Module):
    def __init__(self, feature_num, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(feature_num, output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :return : tensor of size (batch_size, 1)
        """
        return torch.sum(torch.squeeze(self.fc(x)), dim=1, keepdim=True) + self.bias


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, mlp_dims, dropout, output_layer=True, use_bn=False, use_ln=False):
        super().__init__()
        layers = list()
        for mlp_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, mlp_dim))
            if use_bn:
                layers.append(torch.nn.BatchNorm1d(mlp_dim))
            if use_ln:
                layers.append(torch.nn.LayerNorm(mlp_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = mlp_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        :return : tensor of size (batch_size, mlp_dims[-1])
        """
        return self.mlp(x)


class MaskEmbedding_v1(nn.Module):
    def __init__(self, feature_num, latent_dim, mask_initial_value=0., **kwargs):
        super().__init__()
        self.feature_num = feature_num
        self.latent_dim = latent_dim
        self.mask_initial_value = mask_initial_value
        self.embedding = nn.Parameter(torch.zeros(feature_num, latent_dim))
        # nn.init.xavier_uniform_(self.embedding)  # TODO: check any conflict with original embedding initialization
        nn.init.xavier_normal_(self.embedding)
        self.init_weight = nn.Parameter(torch.zeros_like(self.embedding), requires_grad=False)
        self.kept_features = kwargs.get("kept_features", -1)
        self.init_mask()

    def init_mask(self):
        self.mask_weight = nn.Parameter(torch.Tensor(self.feature_num, 1))
        nn.init.constant_(self.mask_weight, self.mask_initial_value)
        if self.kept_features != -1:
            self.mask_weight.data[self.kept_features] = 1
            self.ticket = True
            self.mask_weight.requires_grad = False
    def compute_mask(self, x, temp):
        scaling = 1. / sigmoid(self.mask_initial_value)
        # scaling = 1
        mask_weight = F.embedding(x, self.mask_weight)
        if self.ticket:
            mask = (mask_weight > 0).float()
        else:
            mask = torch.sigmoid(temp * mask_weight)
        return scaling * mask

    def prune(self, temp):
        self.mask_weight.data = torch.clamp(temp * self.mask_weight.data, max=self.mask_initial_value)

    def forward(self, x, temp=1):
        embed = F.embedding(x, self.embedding)
        mask = self.compute_mask(x, temp)
        return embed * mask

    def compute_remaining_weights(self, temp, ticket=False):
        if ticket:
            return float((self.mask_weight > 0.).sum()) / self.mask_weight.numel()
        else:
            m = torch.sigmoid(temp * self.mask_weight)
            print("max mask weight: {wa:6f}, min mask weight: {wi:6f}".format(wa=torch.max(self.mask_weight),
                                                                              wi=torch.min(self.mask_weight)))
            print("max mask: {ma:8f}, min mask: {mi:8f}".format(ma=torch.max(m), mi=torch.min(m)))
            print("mask number: {mn:6f}".format(mn=float((m == 0.).sum())))
            return 1 - float((m == 0.).sum()) / m.numel()

    def checkpoint(self):
        self.init_weight.data = self.embedding.clone()

    def rewind_weights(self):
        self.embedding.data = self.init_weight.clone()

    def reg(self, temp):
        return torch.sum(torch.sigmoid(temp * self.mask_weight))

class MaskEmbedding(nn.Module):
    def __init__(self, feature_num, latent_dim, mask_initial_value=0.):
        super().__init__()
        self.feature_num = feature_num
        self.latent_dim = latent_dim
        self.mask_initial_value = mask_initial_value
        self.embedding = nn.Parameter(torch.zeros(feature_num, latent_dim))
        # nn.init.xavier_uniform_(self.embedding)
        nn.init.xavier_uniform_(self.embedding)  # TODO: check any conflict with original embedding initialization
        self.init_weight = nn.Parameter(torch.zeros_like(self.embedding), requires_grad=False)
        self.init_mask()
        self.temp = 1

    def init_mask(self):
        self.mask_weight = nn.Parameter(torch.Tensor(self.feature_num, 1))
        nn.init.constant_(self.mask_weight, self.mask_initial_value)

    def compute_mask(self, x, temp, ticket, alpha):
        scaling = 1. / sigmoid(self.mask_initial_value)
        mask_weight = F.embedding(x, self.mask_weight)
        if ticket:
            mask = (mask_weight > alpha).float()
        else:
            mask = torch.sigmoid(temp * mask_weight)
        return scaling * mask

    def prune(self, temp):
        self.mask_weight.data = torch.clamp(temp * self.mask_weight.data, max=self.mask_initial_value)

    def forward(self, x):
        ticket = self.ticket
        temp = self.temp
        alpha = self.alpha
        embed = F.embedding(x, self.embedding)
        mask = self.compute_mask(x, temp, ticket, alpha)
        return embed * mask

    def compute_remaining_weights(self, temp, ticket=False, alpha = 0.0):
        logging.info(f"temp: {temp}, ticket: {ticket}")
        if ticket:
            return float((self.mask_weight > alpha).sum()), self.mask_weight.numel()
        else:
            m = torch.sigmoid(temp * self.mask_weight)
            print("max mask weight: {wa:6f}, min mask weight: {wi:6f}".format(wa=torch.max(self.mask_weight),
                                                                              wi=torch.min(self.mask_weight)))
            print("max mask: {ma:8f}, min mask: {mi:8f}".format(ma=torch.max(m), mi=torch.min(m)))
            print("mask number: {mn:6f}".format(mn=float((m == 0.).sum())))
            return 1 - float((m == 0.).sum()) / m.numel()

    def checkpoint(self):
        self.init_weight.data = self.embedding.clone()

    def rewind_weights(self):
        self.embedding.data = self.init_weight.clone()

    def reg(self, temp):
        return torch.sum(torch.sigmoid(temp * self.mask_weight))


class MaskedNet(nn.Module):
    def __init__(self, opt):
        super(MaskedNet, self).__init__()
        self.ticket = False
        self.latent_dim = opt["latent_dim"]
        self.feature_num = opt["feat_num"]
        self.field_num = opt["field_num"]
        self.mask_embedding = MaskEmbedding(self.feature_num, self.latent_dim, mask_initial_value=opt["mask_initial"])
        self.mask_modules = [m for m in self.modules() if type(m) == MaskEmbedding]
        self.temp = 1

    def checkpoint(self):
        for m in self.mask_modules: m.checkpoint()
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Linear):
                m.checkpoint = copy.deepcopy(m.state_dict())

    def rewind_weights(self):
        for m in self.mask_modules: m.rewind_weights()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Linear):
                m.load_state_dict(m.checkpoint)

    def prune(self):
        for m in self.mask_modules: m.prune(self.temp)

    def reg(self):
        reg_loss = 0.
        for m in self.mask_modules:
            reg_loss += m.reg(self.temp)
        return reg_loss


class MaskDeepFM(MaskedNet):
    def __init__(self, opt):
        super(MaskDeepFM, self).__init__(opt)
        self.fm = FactorizationMachine(reduce_sum=True)
        embed_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        use_bn = opt["use_bn"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.dnn = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)

    def forward(self, x):
        x_embedding = self.mask_embedding(x, self.temp, self.ticket)
        # output_linear = self.linear(x)
        output_fm = self.fm(x_embedding)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_dnn = self.dnn(x_dnn)
        logit = output_dnn + output_fm
        return logit

    def compute_remaining_weights(self):
        return self.mask_embedding.compute_remaining_weights(self.temp, self.ticket)


class MaskDeepCross(MaskedNet):
    def __init__(self, opt):
        super(MaskDeepCross, self).__init__(opt)
        self.dnn_dim = self.field_num * self.latent_dim
        cross_num = opt["cross"]
        mlp_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        use_bn = opt["use_bn"]
        self.cross = CrossNetwork(self.dnn_dim, cross_num)
        self.dnn = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=False, dropout=dropout, use_bn=use_bn)
        self.combination = nn.Linear(mlp_dims[-1] + self.dnn_dim, 1, bias=False)

    def forward(self, x):
        x_embedding = self.mask_embedding(x, self.temp, self.ticket)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_cross = self.cross(x_dnn)
        output_dnn = self.dnn(x_dnn)
        comb_tensor = torch.cat((output_cross, output_dnn), dim=1)
        logit = self.combination(comb_tensor)
        return logit

    def compute_remaining_weights(self):
        return self.mask_embedding.compute_remaining_weights(self.temp, self.ticket)


class MaskedFM(MaskedNet):
    def __init__(self, opt):
        super(MaskedFM, self).__init__(opt)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        x_embedding = self.mask_embedding(x, self.temp, self.ticket)
        output_fm = self.fm(x_embedding)
        logits = output_fm
        return logits

    def compute_remaining_weights(self):
        return self.mask_embedding.compute_remaining_weights(self.temp, self.ticket)


class MaskedIPNN(MaskedNet):
    def __init__(self, opt):
        super(MaskedIPNN, self).__init__(opt)
        mlp_dims = opt["mlp_dims"]
        use_bn = opt["use_bn"]
        dropout = opt["mlp_dropout"]
        self.dnn_dim = self.field_num * self.latent_dim + int(self.field_num * (self.field_num - 1) / 2)
        self.inner = InnerProduct(self.field_num)
        self.dnn = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=True, dropout=dropout, use_bn=use_bn)

    def forward(self, x):
        x_embedding = self.mask_embedding(x)
        x_dnn = x_embedding.view(-1, self.field_num * self.latent_dim)
        x_innerproduct = self.inner(x_embedding)
        x_dnn = torch.cat((x_dnn, x_innerproduct), 1)
        logit = self.dnn(x_dnn)
        return logit

    def compute_remaining_weights(self):
        return self.mask_embedding.compute_remaining_weights(self.temp, self.ticket)


def getOptim(network, optim, lr, l2):
    weight_params = map(lambda a: a[1],
                        filter(lambda p: p[1].requires_grad and 'mask_weight' not in p[0], network.named_parameters()))
    mask_params = map(lambda a: a[1],
                      filter(lambda p: p[1].requires_grad and 'mask_weight' in p[0], network.named_parameters()))
    optim = optim.lower()
    if optim == "sgd":
        return [torch.optim.SGD(weight_params, lr=lr, weight_decay=l2), torch.optim.SGD(mask_params, lr=lr)]
    elif optim == "adam":
        return [torch.optim.Adam(weight_params, lr=lr, weight_decay=l2), torch.optim.Adam(mask_params, lr=lr)]
    else:
        raise ValueError("Invalid optimizer type: {}".format(optim))


def getModel(model: str, opt):
    model = model.lower()
    if model == "deepfm":
        return MaskDeepFM(opt)
    elif model == "dcn":
        return MaskDeepCross(opt)
    elif model == "fm":
        return MaskedFM(opt)
    elif model == "ipnn":
        return MaskedIPNN(opt)
    else:
        raise ValueError("Invalid model type: {}".format(model))


def sigmoid(x):
    return float(1. / (1. + np.exp(-x)))
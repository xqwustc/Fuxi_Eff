# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import copy
import logging

import torch
from torch import nn
import h5py
import os
import numpy as np
from collections import OrderedDict
from fuxictr.pytorch.torch_utils import get_initializer
from fuxictr.pytorch import layers


class FeatureEmbedding(nn.Module):
    def __init__(self, 
                 feature_map, 
                 embedding_dim,
                 embedding_initializer="partial(nn.init.normal_, std=1e-4)",
                 required_feature_columns=None,
                 not_required_feature_columns=None,
                 use_pretrain=True,
                 use_sharing=True,
                 **kwargs):
        super(FeatureEmbedding, self).__init__()
        self.embedding_layer = FeatureEmbeddingDict(feature_map, 
                                                    embedding_dim,
                                                    embedding_initializer=embedding_initializer,
                                                    required_feature_columns=required_feature_columns,
                                                    not_required_feature_columns=not_required_feature_columns,
                                                    use_pretrain=use_pretrain,
                                                    use_sharing=use_sharing,
                                                    kept_features=kwargs.get("kept_features", None))
        if kwargs.get("autofeat_mode") in ['table', 'batch']:
            # Only need for AutoFeat
            self.baseline = kwargs.get("baseline", "mean")
            logging.info(f"AutoFeat mode: {kwargs.get('autofeat_mode')}, baseline: {self.baseline}")

            self.backup_self = copy.deepcopy(self)


    def forward(self, X, feature_source=[], feature_type=[], flatten_emb=False):
        feature_emb_dict = self.embedding_layer(X, feature_source=feature_source, feature_type=feature_type)
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=flatten_emb)
        return feature_emb

    def __call__(self, *args, **kwargs):
        if kwargs.get("autofeat_mode") is not None:
            self.interp = kwargs.get("interp")
            self.current_interp = kwargs.get("current_interp")
            embedding_layers_interp = nn.ModuleDict()
            delta_v_dict = dict()
            interp_layer = self.backup_self
            for k, v in self.embedding_layer.embedding_layers.items():
                if type(v) != nn.Embedding:
                    raise NotImplementedError
                else:
                    embedding_layers_interp[k] = copy.deepcopy(v)
                    # start interpolation, the first raw is used for padding_idx, so we exclude it
                    if self.baseline == "mean":
                        mean_v = v.weight[0:].mean(dim=0, keepdim=True)
                    elif self.baseline == "zero":
                        mean_v = torch.zeros_like(v.weight[0])
                    elif self.baseline == "one":
                        mean_v = torch.ones_like(v.weight[0])
                    elif self.baseline == "random":
                        mean_v = torch.randn_like(v.weight[0])
                    else:
                        raise NotImplementedError
                    delta_v = (v.weight - mean_v) / self.interp
                    delta_v_dict[k] = delta_v

                    # interp from the non-informative to the informative
                    embedding_layers_interp[k].weight = nn.Parameter(delta_v * self.current_interp + mean_v)

                    # retain grad
                    embedding_layers_interp[k].weight.requires_grad = True

            interp_layer.embedding_layer.embedding_layers = embedding_layers_interp
            # reset the kwargs, as the normal forward will not use these kwargs
            for key in ['autofeat_mode', 'interp', 'current_interp']:
                kwargs.pop(key, None)

            self.delta_v_dict = delta_v_dict
            self.interp_layer = interp_layer

            return interp_layer(*args, **kwargs), delta_v_dict, interp_layer
        else:
            return super(FeatureEmbedding, self).__call__(*args, **kwargs)


class FeatureEmbeddingDict(nn.Module):
    def __init__(self, 
                 feature_map, 
                 embedding_dim, 
                 embedding_initializer="partial(nn.init.normal_, std=1e-4)",
                 required_feature_columns=None,
                 not_required_feature_columns=None,
                 use_pretrain=True,
                 use_sharing=True,
                 kept_features=None):
        super(FeatureEmbeddingDict, self).__init__()
        self._feature_map = feature_map
        self.required_feature_columns = required_feature_columns
        self.not_required_feature_columns = not_required_feature_columns
        self.use_pretrain = use_pretrain
        self.embedding_initializer = embedding_initializer
        self.embedding_layers = nn.ModuleDict()
        self.feature_encoders = nn.ModuleDict()
        self.kept_features = kept_features
        for feature, feature_spec in self._feature_map.features.items():
            if self.is_required(feature):
                if not (use_pretrain and use_sharing) and embedding_dim == 1:
                    feat_emb_dim = 1 # in case for LR
                    if feature_spec["type"] == "sequence":
                        self.feature_encoders[feature] = layers.MaskedSumPooling()
                else:
                    # If the embedding_dim is specifically defined in the feature_map, then use the value specified
                    # in the special convention within the feature_map; otherwise, use the parameter passed as
                    # an argument.
                    feat_emb_dim = feature_spec.get("embedding_dim", embedding_dim)
                    if feature_spec.get("feature_encoder", None):
                        self.feature_encoders[feature] = self.get_feature_encoder(feature_spec["feature_encoder"])

                # Set embedding_layer according to share_embedding
                if use_sharing and feature_spec.get("share_embedding") in self.embedding_layers:
                    self.embedding_layers[feature] = self.embedding_layers[feature_spec["share_embedding"]]
                    continue

                if feature_spec["type"] == "numeric":
                    self.embedding_layers[feature] = nn.Linear(1, feat_emb_dim, bias=False)
                elif feature_spec["type"] == "categorical":
                    padding_idx = feature_spec.get("padding_idx", None)
                    vocab_size = feature_spec["vocab_size"]
                    if kept_features is not None:
                        if feature not in kept_features:
                            # All features have been pruned
                            vocab_size = 2 # for oov_idx and padding_idx
                        else:
                            if feature_spec['oov_idx'] in kept_features[feature]:
                                # It is naturally kept
                                vocab_size = len(kept_features[feature]) + 1
                            else:
                                # It is pruned, additional add oov_idx
                                vocab_size = len(kept_features[feature]) + 2
                        self._feature_map.features[feature]['oov_idx'] = vocab_size - 1

                    embedding_matrix = nn.Embedding(vocab_size,
                                                    feat_emb_dim, 
                                                    padding_idx=padding_idx)
                    if use_pretrain and "pretrained_emb" in feature_spec:
                        embedding_matrix = self.load_pretrained_embedding(embedding_matrix,
                                                                          feature_map, 
                                                                          feature, 
                                                                          freeze=feature_spec["freeze_emb"],
                                                                          padding_idx=padding_idx)
                    self.embedding_layers[feature] = embedding_matrix
                elif feature_spec["type"] == "sequence":
                    padding_idx = feature_spec.get("padding_idx", None)
                    embedding_matrix = nn.Embedding(feature_spec["vocab_size"], 
                                                    feat_emb_dim, 
                                                    padding_idx=padding_idx)
                    if use_pretrain and "pretrained_emb" in feature_spec:
                        embedding_matrix = self.load_pretrained_embedding(embedding_matrix, 
                                                                          feature_map, 
                                                                          feature,
                                                                          freeze=feature_spec["freeze_emb"],
                                                                          padding_idx=padding_idx)
                    self.embedding_layers[feature] = embedding_matrix
        self.reset_parameters()

    def get_feature_encoder(self, encoder):
        try:
            if type(encoder) == list:
                encoder_list = []
                for enc in encoder:
                    encoder_list.append(eval(enc))
                encoder_layer = nn.Sequential(*encoder_list)
            else:
                encoder_layer = eval(encoder)
            return encoder_layer
        except:
            raise ValueError("feature_encoder={} is not supported.".format(encoder))
        
    def reset_parameters(self):
        self.embedding_initializer = get_initializer(self.embedding_initializer)
        for k, v in self.embedding_layers.items():
            if self.use_pretrain and "pretrained_emb" in self._feature_map.features[k]: # skip pretrained
                continue
            if "share_embedding" in self._feature_map.features[k] and v.weight.requires_grad == False:
                continue
            if type(v) == nn.Embedding:
                if v.padding_idx is not None: # using 0 index as padding_idx
                    self.embedding_initializer(v.weight[1:, :])
                else:
                    self.embedding_initializer(v.weight)
                       
    def is_required(self, feature):
        """ Check whether feature is required for embedding """
        feature_spec = self._feature_map.features[feature]
        if feature_spec["type"] == "meta":
            return False
        elif self.required_feature_columns and (feature not in self.required_feature_columns):
            return False
        elif self.not_required_feature_columns and (feature in self.not_required_feature_columns):
            return False
        else:
            return True

    def get_pretrained_embedding(self, pretrained_path, feature_name):
        with h5py.File(pretrained_path, 'r') as hf:
            embeddings = hf[feature_name][:]
        return embeddings

    def load_pretrained_embedding(self, embedding_matrix, feature_map, feature_name, freeze=False, padding_idx=None):
        pretrained_path = os.path.join(feature_map.data_dir, feature_map.features[feature_name]["pretrained_emb"])
        embeddings = self.get_pretrained_embedding(pretrained_path, feature_name)
        if padding_idx is not None:
            embeddings[padding_idx] = np.zeros(embeddings.shape[-1])
        assert embeddings.shape[-1] == embedding_matrix.embedding_dim, \
            "{}\'s embedding_dim is not correctly set to match its pretrained_emb shape".format(feature_name)
        embeddings = torch.from_numpy(embeddings).float()
        embedding_matrix.weight = torch.nn.Parameter(embeddings)
        if freeze:
            embedding_matrix.weight.requires_grad = False
        return embedding_matrix

    def dict2tensor(self, embedding_dict, feature_list=[], feature_source=[], feature_type=[], flatten_emb=False):
        if type(feature_source) != list:
            feature_source = [feature_source]
        if type(feature_type) != list:
            feature_type = [feature_type]
        feature_emb_list = []
        for feature, feature_spec in self._feature_map.features.items():
            if feature_source and feature_spec["source"] not in feature_source:
                continue
            if feature_type and feature_spec["type"] not in feature_type:
                continue
            if feature_list and feature not in feature_list:
                continue
            if feature in embedding_dict:
                feature_emb_list.append(embedding_dict[feature])
        if flatten_emb:
            feature_emb = torch.cat(feature_emb_list, dim=-1)
        else:
            # pad the unequal length sequence
            max_width = max([emb.size(1) for emb in feature_emb_list])
            if all(emb.size(1) == max_width for emb in feature_emb_list):
                return torch.stack(feature_emb_list, dim=1)
            else:
                feature_emb_cat = torch.cat(feature_emb_list, dim=-1)
                for i, emb in enumerate(feature_emb_list):
                    if emb.size(1) < max_width:
                        diff = max_width - emb.size(1)
                        mean = emb.mean(dim=1, keepdim=True)
                        # TODO: check detach() & whether padding with mean is a good idea
                        padding = mean.detach().repeat(1, diff)
                        feature_emb_list[i] = torch.cat([emb, padding], dim=1)
                feature_emb_pad = torch.stack(feature_emb_list, dim=1)
                return feature_emb_pad, feature_emb_cat
        return feature_emb

    def forward(self, inputs, feature_source=[], feature_type=[]):
        if type(feature_source) != list:
            feature_source = [feature_source]
        if type(feature_type) != list:
            feature_type = [feature_type]
        feature_emb_dict = OrderedDict()
        for feature, feature_spec in self._feature_map.features.items():
            if feature_source and feature_spec["source"] not in feature_source:
                continue
            if feature_type and feature_spec["type"] not in feature_type:
                continue
            if feature in self.embedding_layers:
                if feature_spec["type"] == "numeric":
                    inp = inputs[feature].float().view(-1, 1)
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "categorical":
                    inp = inputs[feature].long()

                    if self.kept_features is not None:
                        oov_idx = feature_spec['oov_idx']
                        if feature not in self.kept_features:
                            # All features have been pruned, use oov_idx
                            inp = torch.full_like(inp, oov_idx)
                        else:
                            # Get new index after pruning
                            inp_pre = copy.deepcopy(inp)
                            kept_features_tensor = torch.tensor(self.kept_features[feature], device=inp.device, dtype=inp.dtype)
                            positions = torch.searchsorted(kept_features_tensor, inp)
                            positions = positions.clamp(max=len(kept_features_tensor) - 1)
                            valid_mask = (kept_features_tensor[positions] == inp)
                            inp = torch.where(valid_mask, positions + 1, oov_idx)

                            # assert (inp == inp_pre).sum() == len(inp_pre), f"Pruning error: {inp} vs {inp_pre}"

                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "sequence":
                    inp = inputs[feature].long()
                    embeddings = self.embedding_layers[feature](inp)
                else:
                    raise NotImplementedError
                if feature in self.feature_encoders:
                    embeddings = self.feature_encoders[feature](embeddings)
                feature_emb_dict[feature] = embeddings
        return feature_emb_dict

class MaskedFeatureEmbedding(FeatureEmbedding):
    def __init__(self, feature_map, embedding_dim, embedding_initializer="partial(nn.init.normal_, std=1e-4)", required_feature_columns=None, not_required_feature_columns=None, use_pretrain=True, use_sharing=True, mask_initial_value=0,temp=1):
        super(MaskedFeatureEmbedding, self).__init__(feature_map, embedding_dim, embedding_initializer, required_feature_columns, not_required_feature_columns, use_pretrain, use_sharing)
        # Add mask and initial weight
        self.temp = temp
        self.mask_initial_value = torch.tensor(mask_initial_value)
        self.mask_weight = nn.Parameter(torch.Tensor(len(feature_map.features), 1))
        nn.init.constant_(self.mask_weight, self.mask_initial_value)

    def compute_mask(self, temp, ticket):
        scaling = 1. / torch.sigmoid(self.mask_initial_value)
        if ticket:
            mask = (self.mask_weight > 0).float()
        else:
            mask = torch.sigmoid(temp * self.mask_weight)
        return scaling * mask

    def forward(self, X, feature_source=[], feature_type=[], flatten_emb=False, ticket=False):
        feature_emb = super().forward(X, feature_source, feature_type, flatten_emb)
        mask = self.compute_mask(self.temp, ticket)
        return feature_emb * mask

    def prune(self, temp):
        self.mask_weight.data = torch.clamp(temp * self.mask_weight.data, max=self.mask_initial_value)

    def reg(self,temp = 1):
        return torch.sum(torch.sigmoid(temp * self.mask_weight))



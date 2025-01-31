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
import logging

import torch
from torch import nn
from fuxictr.pytorch.layers import FeatureEmbedding


class LogisticRegression(nn.Module):
    def __init__(self, feature_map, use_bias=True, embedding_layer = None, **kwargs):
        super(LogisticRegression, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True) if use_bias else None
        # A trick for quick one-hot encoding in LR
        if embedding_layer is not None:
            logging.info("Use same embedding layer for Logistic Regression as external embedding layer.")
            self.embedding_layer = embedding_layer
        else:
            logging.info("Use the independent embedding layer for Logistic Regression.")
            self.embedding_layer = FeatureEmbedding(feature_map, 1, use_pretrain=False, use_sharing=False,
                                                    kept_features=kwargs.get('kept_features'))
            if kwargs.get('kept_features') is not None:
                # move to cuda
                for k,v in self.embedding_layer.embedding_layer.vocab_ind_map.items():
                    self.embedding_layer.embedding_layer.vocab_ind_map[k] = v.to(kwargs.get('device'))

    def forward(self, X):
        embed_weights = self.embedding_layer(X)
        output = embed_weights.sum(dim=1)
        if self.bias is not None:
            output += self.bias
        return output


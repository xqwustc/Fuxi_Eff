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

import torch
from torch import nn 
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import get_activation
import logging
from tqdm import tqdm
import sys
import numpy as np
import pandas as pd


class MaskNet(BaseModel):
    def __init__(self, 
                 feature_map,
                 model_id="MaskNet",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_hidden_units=[64,64,64],
                 dnn_hidden_activations="ReLU",
                 model_type="SerialMaskNet",
                 parallel_num_blocks=1,
                 parallel_block_dim=64,
                 reduction_ratio=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 net_dropout=0,
                 emb_layernorm=True,
                 net_layernorm=True,
                 **kwargs):
        super(MaskNet, self).__init__(feature_map,
                                      model_id=model_id,
                                      gpu=gpu,
                                      embedding_regularizer=embedding_regularizer,
                                      net_regularizer=net_regularizer,
                                      **kwargs)

        if kwargs.get('optfs_dict', None) is not None:
            assert type(kwargs.get('optfs_dict')) == dict, 'optfs params should be a dict'
            optfs_dict = {
                'temp': 5000
            }
            optfs_dict.update(kwargs.get('optfs_dict'))
            self.optfs_dict = optfs_dict

            if optfs_dict.get('retrain', False):
                ratio = kwargs.get('keep_ratio', 1)
                kept_features = self.read_scores(score_name='feature_score_optfs',
                                                 score_version=kwargs.get("score_version", ""),
                                                 ratio=ratio)
                print('[OptFS] Using Masked Embedding Layer.')
                # self.need_move = True
                self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim, kept_features=kept_features,
                                                        optfs_dict=optfs_dict)
            else:
                print('[OptFS] Using OptFS Embedding Layer with parameters:', optfs_dict)
                self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim, optfs_dict=optfs_dict)
        elif kwargs.get('pep_dict') is not None:
            # get a dict
            pep_dict = kwargs.get('pep_dict')
            self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim, pep_dict=pep_dict)
        elif kwargs.get('autofeat_mode') == "retrain":
            # AutoFeat condition, will pass kept_features to embedding layer
            ratio = kwargs.get('keep_ratio', 1)
            kept_features = self.read_scores(score_version=kwargs.get("score_version", ""), ratio=ratio)
            self.need_move = True
            self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim, kept_features=kept_features)
        elif kwargs.get('autofeat_mode') in ['batch', 'table']:
            self.score_mode = kwargs.get('score_mode', 'sum')
            logging.info(f"Score mode: {self.score_mode}")
            self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim, autofeat_mode=kwargs['autofeat_mode'],
                                                    baseline=kwargs.get('baseline'))
        else:
            # Normal condition
            self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        if kwargs.get('autofeat_mode') != None:
            self.autofeat_mode = kwargs['autofeat_mode']
            if self.autofeat_mode != 'retrain':
                # cal the score, use the share embedding for
                kwargs['emb_layer'] = self.embedding_layer
                self.share_embedding_layer = True
                self.interpolate_n = 1
                logging.info(f"Interpolation layer: {self.interpolate_n}")

        # self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        if model_type == "SerialMaskNet":
            self.mask_net = SerialMaskNet(input_dim=feature_map.num_fields * embedding_dim,
                                          output_dim=1,
                                          output_activation=self.output_activation,
                                          hidden_units=dnn_hidden_units,
                                          hidden_activations=dnn_hidden_activations,
                                          reduction_ratio=reduction_ratio,
                                          dropout_rates=net_dropout,
                                          layer_norm=net_layernorm)
        elif model_type == "ParallelMaskNet":
            self.mask_net = ParallelMaskNet(input_dim=feature_map.num_fields * embedding_dim,
                                            output_dim=1,
                                            output_activation=self.output_activation,
                                            num_blocks=parallel_num_blocks, 
                                            block_dim=parallel_block_dim, 
                                            hidden_units=dnn_hidden_units,
                                            hidden_activations=dnn_hidden_activations,
                                            reduction_ratio=reduction_ratio,
                                            dropout_rates=net_dropout,
                                            layer_norm=net_layernorm)
        self.num_fields = feature_map.num_fields
        if emb_layernorm:
            self.emb_norm = nn.ModuleList(nn.LayerNorm(embedding_dim) for _ in range(self.num_fields))
        else:
            self.emb_norm = None
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
    
    def forward(self, inputs):
        if not hasattr(self, "autofeat_mode") or self.autofeat_mode == 'retrain':
            X = self.get_inputs(inputs)
            feature_emb = self.embedding_layer(X)
            if self.emb_norm is not None:
                feat_list = feature_emb.chunk(self.num_fields, dim=1)
                V_hidden = torch.cat([self.emb_norm[i](feat) for i, feat in enumerate(feat_list)], dim=1)
            else:
                V_hidden = feature_emb
            y_pred = self.mask_net(feature_emb.flatten(start_dim=1), V_hidden.flatten(start_dim=1))
            return_dict = {"y_pred": y_pred}
            return return_dict
        elif self.autofeat_mode in ['batch', 'table']:
            # When using AutoFeat to get scores, the forward will be different
            return self.forward_with_autofeat(inputs)
        else:
            raise NotImplementedError

    def forward_with_autofeat(self, inputs, cur_interp=None):
        # the interpolation is done for embedding table NOT for the feature field!!!
        X = self.get_inputs(inputs)
        if cur_interp is not None:
            embed_res, delta_v_dict, interp_layer = self.embedding_layer(X, autofeat_mode=self.autofeat_mode,
                                                                         interp=self.interpolate_n,
                                                                         current_interp=cur_interp)

            if isinstance(embed_res, tuple):
                feature_emb_stack, feature_emb_cat = embed_res
            else:
                feature_emb = embed_res
                # feature_emb = feature_emb_stack.flatten(start_dim=1)

            if self.emb_norm is not None:
                feat_list = feature_emb.chunk(self.num_fields, dim=1)
                V_hidden = torch.cat([self.emb_norm[i](feat) for i, feat in enumerate(feat_list)], dim=1)
            else:
                V_hidden = feature_emb
            y_pred = self.mask_net(feature_emb.flatten(start_dim=1), V_hidden.flatten(start_dim=1))
            return_dict = {"y_pred": y_pred}
            return return_dict, delta_v_dict, interp_layer

        else:
            embed_res = self.embedding_layer(X)
            if isinstance(embed_res, tuple):
                feature_emb_stack, feature_emb_cat = embed_res
            else:
                feature_emb = embed_res

            if self.emb_norm is not None:
                feat_list = feature_emb.chunk(self.num_fields, dim=1)
                V_hidden = torch.cat([self.emb_norm[i](feat) for i, feat in enumerate(feat_list)], dim=1)
            else:
                V_hidden = feature_emb
            y_pred = self.mask_net(feature_emb.flatten(start_dim=1), V_hidden.flatten(start_dim=1))
            return_dict = {"y_pred": y_pred}
            return return_dict

    def evaluate_with_autofeat(self, data_generator):
        data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
        feature_score_dict = dict() # key is feature name and v is the score tensor for each feature
        for batch_data in data_generator:
            for cur_interp in range(1, self.interpolate_n + 1):
                return_dict, delta_v, interp_layer = self.forward_with_autofeat(batch_data, cur_interp)
                y_true = self.get_labels(batch_data)
                loss = self.compute_loss(return_dict, y_true)
                loss.backward(retain_graph=False)

                for feature_name, embed_table in interp_layer.embedding_layer.embedding_layers.items():
                    if self.score_mode in ['sum', 'sum_abs']:
                        feature_attr = (embed_table.weight.grad * delta_v[feature_name]).sum(dim = -1)
                    elif self.score_mode == 'abs':
                        feature_attr = (embed_table.weight.grad * delta_v[feature_name]).abs().sum(dim = -1)
                    else:
                        raise NotImplementedError

                    if feature_name in feature_score_dict:
                        feature_score_dict[feature_name] += feature_attr.detach().cpu().numpy()
                    else:
                        feature_score_dict[feature_name] = feature_attr.detach().cpu().numpy()

                self.optimizer.zero_grad()

                torch.cuda.empty_cache()

        feature_name_list, index_list, score_list = [], [], []
        # 遍历 feature_score_dict
        for feature_name, score in feature_score_dict.items():
            # 获取当前特征的索引和值
            indices = np.arange(len(score))  # 索引
            if self.score_mode == 'sum_abs':
                scores = np.abs(score)
            else:
                scores = score
            # 将 feature_name 和 scores 进行批量拼接
            feature_name_list.extend([feature_name] * len(score))  # 重复 feature_name
            index_list.extend(indices)  # 添加索引
            score_list.extend(scores)  # 添加分数

        # Combine to DataFrame
        feature_score = pd.DataFrame({
            'feature_name': feature_name_list,
            'index': index_list,
            'score': score_list
        })

        # 按照 score 降序排序
        feature_score_sorted = feature_score.sort_values(by='score', ascending=False)

        self.save_scores(feature_score_sorted)
        return

class SerialMaskNet(nn.Module):
    def __init__(self, input_dim, output_dim=None, output_activation=None, hidden_units=[], 
                 hidden_activations="ReLU", reduction_ratio=1, dropout_rates=0, layer_norm=True):
        super(SerialMaskNet, self).__init__()
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        self.hidden_units = [input_dim] + hidden_units
        self.mask_blocks = nn.ModuleList()
        for idx in range(len(self.hidden_units) - 1):
            self.mask_blocks.append(MaskBlock(input_dim, 
                                              self.hidden_units[idx], 
                                              self.hidden_units[idx + 1], 
                                              hidden_activations[idx], 
                                              reduction_ratio, 
                                              dropout_rates[idx],
                                              layer_norm))
        fc_layers = []
        if output_dim is not None:
            fc_layers.append(nn.Linear(self.hidden_units[-1], output_dim))
        if output_activation is not None:
            fc_layers.append(get_activation(output_activation))
        self.fc = None
        if len(fc_layers) > 0:
            self.fc = nn.Sequential(*fc_layers)

    def forward(self, V_emb, V_hidden):
        v_out = V_hidden
        for idx in range(len(self.hidden_units) - 1):
            v_out = self.mask_blocks[idx](V_emb, v_out)
        if self.fc is not None:
            v_out = self.fc(v_out)
        return v_out


class ParallelMaskNet(nn.Module):
    def __init__(self, input_dim, output_dim=None, output_activation=None, num_blocks=1, block_dim=64, 
                 hidden_units=[], hidden_activations="ReLU", reduction_ratio=1, dropout_rates=0, 
                 layer_norm=True):
        super(ParallelMaskNet, self).__init__()
        self.num_blocks = num_blocks
        self.mask_blocks = nn.ModuleList([MaskBlock(input_dim, 
                                                    input_dim, 
                                                    block_dim, 
                                                    hidden_activations, 
                                                    reduction_ratio, 
                                                    dropout_rates,
                                                    layer_norm) for _ in range(num_blocks)])

        self.dnn = MLP_Block(input_dim=block_dim * num_blocks,
                             output_dim=output_dim, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=output_activation,
                             dropout_rates=dropout_rates)

    def forward(self, V_emb, V_hidden):
        block_out = []
        for i in range(self.num_blocks):
            block_out.append(self.mask_blocks[i](V_emb, V_hidden))
        concat_out = torch.cat(block_out, dim=-1)
        v_out = self.dnn(concat_out)
        return v_out


class MaskBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_activation="ReLU", reduction_ratio=1, 
                 dropout_rate=0, layer_norm=True):
        super(MaskBlock, self).__init__()
        self.mask_layer = nn.Sequential(nn.Linear(input_dim, int(hidden_dim * reduction_ratio)),
                                        nn.ReLU(),
                                        nn.Linear(int(hidden_dim * reduction_ratio), hidden_dim))
        hidden_layers = [nn.Linear(hidden_dim, output_dim, bias=False)]
        if layer_norm:
            hidden_layers.append(nn.LayerNorm(output_dim))
        hidden_layers.append(get_activation(hidden_activation))
        if dropout_rate > 0:
            hidden_layers.append(nn.Dropout(p=dropout_rate))
        self.hidden_layer = nn.Sequential(*hidden_layers)

    def forward(self, V_emb, V_hidden):
        V_mask = self.mask_layer(V_emb)
        v_out = self.hidden_layer(V_mask * V_hidden)
        return v_out
        
        
        














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
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, FactorizationMachine
import numpy as np
import logging
from torch import log
import pandas as pd
from tqdm import tqdm
import sys
from feat_select.Selectors import AdaFS
EPS = 1e-6


class DeepFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DeepFM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DeepFM, self).__init__(feature_map, 
                                     model_id=model_id, 
                                     gpu=gpu, 
                                     embedding_regularizer=embedding_regularizer, 
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.fm = FactorizationMachine(feature_map)
        self.mlp = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm)
        self.learning_rate = learning_rate

        # # --- update for droprank start---
        # self.gates_theta = torch.ones(len(self.feature_map.features)) * 0.5
        # self.gates_theta.requires_grad_(requires_grad=True)
        # # --- update for droprank end---

        # # --- update for AdaFS start---
        self.adafs = AdaFS(feature_map.num_fields,embedding_dim)
        # # --- update for AdaFS end---

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)  # like(size,features,emb_size), the embedding info got
        y_pred = self.fm(X, feature_emb)
        y_pred += self.mlp(feature_emb.flatten(start_dim=1))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def forward_with_dr(self,inputs,gates_prob):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)

        # --- update for droprank start---
        for i in range(len(self.feature_map.features)):
            feature_emb[:,i:i+1,:] *= gates_prob[i]
        # --- update for droprank end---

        y_pred = self.fm(X, feature_emb)
        y_pred += self.mlp(feature_emb.flatten(start_dim=1))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def forward_with_adafs(self,inputs):
        X = self.get_inputs(inputs)

        feature_emb = self.embedding_layer(X)
        # The embedding process for X will be in AdaFS
        feature_emb = self.adafs(feature_emb.transpose(1,2))

        y_pred = self.fm(X, feature_emb)
        y_pred += self.mlp(feature_emb.flatten(start_dim=1))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def forward_with_adafs_eval(self,inputs,seed=2019):
        self.eval()
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        # The embedding process for X will be in AdaFS
        batch_weight = self.adafs.forward_for_eval(feature_emb.transpose(1,2))
        return batch_weight

    def fit_for_dr(self, data_generator, epochs=1, validation_data=None,
                   max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch

        torch.autograd.set_detect_anomaly(True)
        # Make a list of theta for each feature
        gates_theta = torch.ones(len(self.feature_map.features)) * 0.5
        gates_theta.requires_grad_(requires_grad=True)

        gates_sigma = torch.ones(len(self.feature_map.features)) * 0.5
        gates_sigma.requires_grad_(requires_grad=True)

        # --- update for droprank start---
        self.optimizer.add_param_group({'params': gates_theta, 'lr': self.learning_rate * 0.1})
        self.optimizer.add_param_group({'params': gates_sigma, 'lr': self.learning_rate * 0.1})
        # --- update for droprank end---


        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self._batch_index = 0
            train_loss = 0
            self.train()
            if self._verbose == 0:
                batch_iterator = data_generator
            else:
                batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_index, batch_data in enumerate(batch_iterator):
                self._batch_index = batch_index
                self._total_steps += 1

                # --- update for droprank start---
                gates_prob = self._get_gates_prob(gates_theta,gates_sigma)
                return_dict = self.forward_with_dr(batch_data, gates_prob)
                # --- update for droprank end---

                self.optimizer.zero_grad()

                y_true = self.get_labels(batch_data)
                loss = self.compute_loss(return_dict, y_true)

                # --- update for droprank start---
                loss += torch.sum(gates_prob) * 1e-3

                ####
                # dot = make_dot(loss, params=dict(self.named_parameters()))
                # dot.view()
                ####

                # --- update for droprank end---

                loss.backward()

                # # --- update for droprank start---
                # print("\n",gates_theta.grad,"\n")
                # # --- update for droprank end---

                nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
                self.optimizer.step()

                train_loss += loss.item()
                if self._total_steps % self._eval_steps == 0:
                    logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                    train_loss = 0
                    # eval the model
                    self.eval_step()
                if self._stop_training:
                    break

                # --- update for droprank start---
            logging.info("\n Gates Theta: {}".format(gates_theta))
            # --- update for droprank end---

            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)
        print(gates_theta)
        feature_importance_result = pd.DataFrame({'feature_name': list(self.feature_map.features.keys()),
                                                  'feature_weight': gates_theta.tolist(),
                                                  'feature_sigma': gates_sigma.tolist()})
        feature_importance_result.to_csv('feature_importance_result.csv', index=False)
        return

    def fit_for_adafs(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch

        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self._batch_index = 0

            train_loss = 0
            self.train()
            if self._verbose == 0:
                batch_iterator = data_generator

            else:
                batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_index, batch_data in enumerate(batch_iterator):
                self._batch_index = batch_index
                self._total_steps += 1

                # --- update for adafs start---
                return_dict = self.forward_with_adafs(batch_data)
                # --- update for adafs end---

                self.optimizer.zero_grad()

                y_true = self.get_labels(batch_data)
                loss = self.compute_loss(return_dict, y_true)

                loss.backward()

                #print(self.adafs.controller.mlp.mlps[0][0].weight.grad)

                nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
                self.optimizer.step()


                train_loss += loss.item()
                if self._total_steps % self._eval_steps == 0:
                    logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                    train_loss = 0
                    # eval the model
                    self.eval_step()
                if self._stop_training:
                    break

            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)
        return


    def evaluate_with_adafs(self, data_generator, seed=2019):
        self.eval()
        data_generator = tqdm(data_generator, disable=False, file=sys.stdout)

        weight = torch.zeros(1, len(self.feature_map.features))

        for batch_data in data_generator:
            # Get the batch weight from validation dataset, and will get a (n*feature_num) matrix
            batch_weight = self.forward_with_adafs_eval(batch_data, seed)
            batch_weight_use = batch_weight.cpu().detach()
            batch_weight_use = batch_weight_use.sum(dim=0, keepdim=True)
            # print('batch_weight_use', batch_weight_use)
            weight += batch_weight_use

        # 2-Norm the weight
        weight /= torch.norm(weight)

        # 处理成可读的特征重要性指标
        feature_importance_result = pd.DataFrame({'feature_name': list(self.feature_map.features.keys()),
                                                  'feature_weight': np.squeeze(weight.numpy()).tolist()})

        feature_importance_result = feature_importance_result.sort_values(by='feature_weight', ascending=False)
        feature_importance_result.to_csv('feature_importance_result.csv', index=False)

        return

    def evaluate_with_pfi(self, data_generator, valid_result=None, metrics=None, seed=2019):
        logging.info("Start evaluate with PFI-DeepFM")

        pfi_score_res = pd.DataFrame(columns=['AUC', 'logloss'])

        ## FIXME: 1 means the label column
        total_field = len(self.feature_map.features) + 1
        for feat_idx in range(total_field):
            if feat_idx == self.feature_map.get_column_index(self.feature_map.labels[0]):
                continue
            perm_gen = self._permute_feature(data_generator, feat_idx)
            cur_result = self.evaluate(perm_gen, metrics=metrics)
            diff = pd.DataFrame([{
                'AUC': cur_result['AUC'] - valid_result['AUC'],
                'logloss': cur_result['logloss'] - valid_result['logloss']
            }])
            pfi_score_res = pd.concat([pfi_score_res, diff], ignore_index=True)

        # Add feature name
        pfi_score_res.insert(0, 'feature_name', list(self.feature_map.features.keys()))

        # Get the absolute value of AUC & logloss
        pfi_score_res['AUC'] = pfi_score_res['AUC'].abs()
        pfi_score_res['logloss'] = pfi_score_res['logloss'].abs()

        # Sort by AUC and see AUC as the feature_weight
        pfi_score_res = pfi_score_res.sort_values(by='AUC', ascending=False)
        pfi_score_res.insert(1, 'feature_weight', pfi_score_res['AUC'])
        pfi_score_res.to_csv('feature_importance_result.csv', index=False)
        return



    def _permute_feature(self,data_generator, feature_idx):
        """
        Permutes the values of a specific feature in each batch produced by the data_generator.

        Args:
        - data_generator: Original data generator.
        - feature_idx: The index of the feature you want to permute.

        Yields:
        - Batch with permuted feature values.
        """
        for batch in data_generator:
            # Deep copy to avoid modifying the original batch
            permuted_batch = batch.clone()

            # Permute the feature using PyTorch functions
            perm = torch.randperm(permuted_batch.size(0))
            permuted_batch[:, feature_idx] = permuted_batch[perm, feature_idx]

            yield permuted_batch

    def _get_gates_prob(self, gates_theta, gates_sigma):
        assert len(gates_theta) == len(gates_sigma), "gates_theta and gates_sigma should have the same length"
        gates_prob = gates_theta.clone()
        for i in range(gates_theta.shape[0]):
            gates_prob[i] = self._get_prob(gates_theta[i], gates_sigma[i])
        return gates_prob

    def _get_prob(self, unit, sigma_unit):
        eps = torch.randn(1) * sigma_unit

        # u = torch.randn(1)*sigma_unit
        # u.requires_grad = False
        # u = u.to(device=self.device)

        return torch.sigmoid((1.0 / 0.1) * (unit + eps))

    def _get_featuremap_size(self,feature_map):
        # 先写死每个维度的emb_size = 32，后续可以改成从feature_map中读取
        return [40]*len(feature_map.features)

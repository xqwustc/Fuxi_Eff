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
import os.path

import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
import numpy as np
from itertools import cycle
from model_zoo.utils import get_gates_prob_autofield,get_gates_prob_my,get_sum_feature_dimisions
from torch import optim
import logging
import sys
import pandas as pd
from tqdm import tqdm
from itertools import cycle
import torch.optim as optim
import feat_select.MvFS_module as Mv
EPS = 1e-6



class DNN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DNN",
                 gpu=-1,
                 learning_rate=1e-3,
                 select_num=0,
                 embedding_dim=10,
                 hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(DNN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.learning_rate = learning_rate
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.mlp = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=self.output_activation,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        if select_num > 0:
            self.controller = Mv.MvFS_Controller(input_dim=get_sum_feature_dimisions(self.embedding_layer),
                                                 embed_dims=len(self.feature_map.features), num_selections=select_num)
        self.weight = 0

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=True)
        y_pred = self.mlp(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def forward_with_autofield(self, inputs, gates_prob):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)

        for i in range(len(self.feature_map.features)):
            feature_emb[:,i:i+1,:] *= gates_prob[i]

        feature_emb = feature_emb.flatten(start_dim=1)

        y_pred = self.mlp(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def forward_with_dr(self, inputs, gates_prob):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)

        for i in range(len(self.feature_map.features)):
            feature_emb[:,i:i+1,:] *= gates_prob[i]

        feature_emb = feature_emb.flatten(start_dim=1)

        y_pred = self.mlp(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def forward_with_mvfs(self,inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        # The embedding process for X will be in MvFS
        self.weight = self.controller(feature_emb)
        selected_field = feature_emb * torch.unsqueeze(self.weight, 2)

        feature_emb = selected_field.flatten(start_dim=1)
        y_pred = self.mlp(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict


    def fit_for_autofield(self, data_generator, epochs=1, validation_data=None,
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

        self._freq = 5

        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch

        cyc_val = cycle(validation_data)
        # Make a list of theta for each feature
        gates_a1 = torch.ones(len(self.feature_map.features)) * 0.5
        gates_a1.requires_grad_(requires_grad=True)

        # --- update for droprank start---
        # self.optimizer.add_param_group({'params': gates_a1, 'lr': self.learning_rate * 0.1})
        # --- update for droprank end---

        gates_optimizer = optim.Adam([gates_a1], lr=self.learning_rate * 0.1)
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

                gates_a1.data = torch.clamp(gates_a1.data, min=EPS, max=1)
                gates_prob = get_gates_prob_autofield(gates_a1, epoch=self._total_steps)

                return_dict = self.forward_with_autofield(batch_data, gates_prob)
                # --- update for droprank end---

                self.optimizer.zero_grad()
                gates_optimizer.zero_grad()

                y_true = self.get_labels(batch_data)
                loss = self.compute_loss(return_dict, y_true)

                loss.backward()

                train_loss += loss.item()

                nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
                self.optimizer.step()

                if self._total_steps % self._freq == 0:
                    # A new computation graph is built
                    gates_prob = get_gates_prob_autofield(gates_a1, epoch=self._total_steps)

                    batch_val_data = next(cyc_val)
                    gates_optimizer.zero_grad()
                    return_dict = self.forward_with_autofield(batch_val_data, gates_prob)
                    y_true = self.get_labels(batch_val_data)
                    loss_val = self.compute_loss(return_dict, y_true)
                    loss_val.backward()
                    gates_optimizer.step()

                if self._total_steps % self._eval_steps == 0:
                    logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                    train_loss = 0
                    # eval the model
                    self.eval_step()
                if self._stop_training:
                    break

                # --- update for droprank start---
            logging.info("\n Autofield Gates Theta: {}".format(gates_a1))
            # --- update for droprank end---

            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)
        print(gates_a1)
        feature_importance_result = pd.DataFrame({'feature_name': list(self.feature_map.features.keys()),
                                                  'feature_weight': gates_a1.tolist()})
        feature_importance_result.to_csv('feature_importance_result.csv', index=False)
        return

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

        # Make a list of theta for each feature
        gates_theta = torch.ones(len(self.feature_map.features)) * 0.5
        gates_theta.requires_grad_(requires_grad=True)

        gates_sigma = torch.ones(len(self.feature_map.features)) * 0.5
        gates_sigma.requires_grad_(requires_grad=True)

        # --- update for droprank start---
        self.optimizer.add_param_group({'params': gates_theta, 'lr': self.learning_rate*0.1})
        self.optimizer.add_param_group({'params': gates_sigma, 'lr': self.learning_rate*0.1})
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
                gates_prob = get_gates_prob_my(gates_theta, gates_sigma)
                return_dict = self.forward_with_dr(batch_data, gates_prob)
                # --- update for droprank end---

                self.optimizer.zero_grad()

                y_true = self.get_labels(batch_data)
                loss = self.compute_loss(return_dict, y_true)

                # --- update for droprank start---
                loss += (torch.sum(gates_prob)) * 1e-5

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
            logging.info("\nMy Gates Theta: {}".format(gates_theta))
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

    def fit_for_mvfs(self, data_generator, epochs=1, validation_data=None,
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

                return_dict = self.forward_with_mvfs(batch_data)
                # --- update for droprank end---

                self.optimizer.zero_grad()

                y_true = self.get_labels(batch_data)
                loss = self.compute_loss(return_dict, y_true)
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
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)

    def weight_watcher(self):
        # TODO: run in debug mode
        # load gates weight in ../feature_importance_result.csv
        # the first column is feature name, the second column is feature weight (gates)
        gates = pd.read_csv('./feature_importance_result.csv')
        gates_np = gates.iloc[:, 1].values.reshape(39, 1)
        first_weight = self.mlp.mlp[0].cpu().weight.data.cpu().numpy()
        first_weight = first_weight.reshape(39,40,500)

        result = np.zeros((39, 500))

        # 对每个子矩阵进行操作
        for i in range(39):
            # 对应的 A 元素和 B 的子矩阵 (40x500) 相乘并求和
            result[i, :] = np.sum(first_weight[i, :, :] * gates_np[i], axis=0)
        # first weight is (1560,500), transpose to (500,1560)

        # save numpy as csv
        np.savetxt('./first_weight.csv', result, delimiter=',')

        self.load_weights(self.checkpoint)

        self.eval()

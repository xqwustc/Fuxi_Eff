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


import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
from statistics import NormalDist

sys.path.append('../../fuxictr')
sys.path.append('../../')

# --- update for fmcr ---
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern
import pandas as pd
# --- update for fmcr ---

import logging
import pickle
import copy
import fuxictr_version
import numpy as np
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import H5DataLoader
from fuxictr.preprocess import FeatureProcessor, build_dataset
import src as model_zoo
import gc
import argparse
import os
from pathlib import Path
import importlib
from infomer import email
from model_zoo.utils import cluster_features

if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())

    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    if params.get('spe_processor'):
        module_name = f"fuxictr.datasets.{params['spe_processor']}"
        fp_module = importlib.import_module(module_name)
        assert hasattr(fp_module, 'FeatureProcessor')
        FeatureProcessor = getattr(fp_module, 'FeatureProcessor')
    else:
        from fuxictr.preprocess import FeatureProcessor

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "csv":
        # Build feature_map and transform h5 data
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)

    # here specify feature numbers
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    # 解析特征重要性的均值和方差作为先验
    df = pd.read_csv('feature_importance_result.csv')

    topk_column_name = []
    topk_logloss_result = []
    topk_auc_result = []

    def ratio_for_features(ratio = 0.1):
        '''
            Given the feature values ratio, return the K that satisfies the ratio
        '''
        topk_params = copy.deepcopy(params)
        topk_params['use_features'] = df['feature_name'].values.tolist()
        topk_feature_map = FeatureMap(topk_params['dataset_id'], data_dir)
        topk_feature_map.load(feature_map_json, topk_params)
        tot_features = 0
        for feature_name, d in topk_feature_map.features.items():
            tot_features += d['vocab_size']

        for i in range(len(feature_map.features)):
            cur_ratio = sum([topk_feature_map.features[d]['vocab_size'] for d in topk_params['use_features']][:i + 1]) / tot_features
            if cur_ratio >= ratio:
                if i == 0:
                    return int(len(df['feature_name'].values.tolist())*ratio), cur_ratio
                else:
                    return i, cur_ratio

    idx, ratio = ratio_for_features(0.1)

    # for i in [9, 10, 11]:
    # incre = 1
    # for i in reversed(range(9 + incre - 1, len(feature_map.features) + incre - 1, incre)):
    for i in [idx - 1, idx]:
        topk_params = copy.deepcopy(params)
        seed_everything(seed=params['seed'])
        topk_params['use_features'] = df['feature_name'].values.tolist()[:i + 1]

        logging.info('--- Used Features: {} (totally {} features)'.format(topk_params['use_features'],
                                                                          len(topk_params['use_features'])))
        topk_feature_map = FeatureMap(topk_params['dataset_id'], data_dir)
        topk_feature_map.load(feature_map_json, topk_params)
        topk_feature_map.num_fields = len(topk_params['use_features'])

        model_class = getattr(model_zoo, topk_params['model'])
        topk_model = model_class(topk_feature_map, **topk_params)
        topk_model.count_parameters()  # print number of parameters used in model

        #train_gen, valid_gen = H5DataLoader(topk_feature_map, stage='train', **topk_params).make_iterator()
        train_gen, valid_gen, test_gen = H5DataLoader(topk_feature_map, stage='both', **topk_params).make_iterator()

        topk_model.fit(train_gen, validation_data=valid_gen, **params)
        valid_result = topk_model.evaluate(test_gen)

        topk_column_name.append('topk_{}_with_{}'.format(i + 1, topk_params['use_features']))
        topk_logloss_result.append(valid_result['logloss'])
        topk_auc_result.append(valid_result['AUC'])
        cur_AUC = valid_result['AUC']
        cur_logloss = valid_result['logloss']

        del train_gen, valid_gen, test_gen
        gc.collect()

        #     if cur_AUC >= SOTA_AUC and cur_logloss <= SOTA_logloss:
        #         break
        #
        # total_times += 1
        # logging.info('Time: {}'.format(total_times))

    topk_ablation_df = pd.DataFrame({'feature_name': topk_column_name,
                                     'topk_logloss': topk_logloss_result,
                                     'topk_auc': topk_auc_result})
    topk_ablation_df.to_csv('feature_ablation.csv')
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

sys.path.append('../../../fuxictr')
sys.path.append('../../../')
# print(sys.path)

import logging
import pickle
import fuxictr
import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import H5DataLoader
from fuxictr.preprocess import build_dataset
import src as model_zoo

import gc
import argparse
import os
from pathlib import Path
import importlib
import torch
import numpy as np
import logging
from infomer import email
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--cp', type=str, help='checkpoint path')
    args = vars(parser.parse_args())

    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    if params.get('spe_processor',None) != None:
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
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)

    # model_class = getattr(model_zoo, params['model'])
    print('params[model]', params['model'])
    # model = model_class(feature_map, **params)
    # model.count_parameters()  # print number of parameters used in model

    train_gen, valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()
    test_gen = H5DataLoader(feature_map, stage='test', **params).make_iterator()
    test_result = {}

    label_column = feature_map.column_index[feature_map.labels[0]]

    sampled_data = []
    # get the first 10 batches
    for i, data in enumerate(train_gen):
        if i >= 1400:
            break
        sampled_data.append(data.numpy())

    sampled_data = np.vstack(sampled_data)
    y = sampled_data[:, label_column]
    X = np.delete(sampled_data, label_column, axis=1)

    # # mutual info
    # mi_scores = mutual_info_classif(X,y)
    # df = pd.DataFrame({'feature_name': feature_map.features.keys(), 'mi_score': mi_scores})
    # df.to_csv('feature_importance_result_m1.csv', index=False)
    #
    # correlation
    # cor = np.corrcoef(X.T,y)[:-1,-1]
    # df = pd.DataFrame({'feature_name': feature_map.features.keys(), 'cor_score': cor})
    # df.to_csv('feature_importance_result_cor.csv', index=False)
    #
    #
    # chi (only OK for non-negative feature fields)
    # X_trans = MinMaxScaler().fit_transform(X)
    # chi_scores, p_values = chi2(X_trans, y)
    # df = pd.DataFrame({'feature_name': feature_map.features.keys(), 'chi_score': chi_scores})
    # df.to_csv('feature_importance_result_chi.csv', index=False)

    dataset_name = experiment_id.split('_')[-1]

    logging.info('****** Start to calculate Xgboost feature importance ******')
    model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                             max_depth=5, alpha=10, n_estimators=10)
    model.fit(X,y)
    importance = model.feature_importances_
    df = pd.DataFrame({'feature_name': feature_map.features.keys(), 'xgb_score': importance})
    df = df.sort_values(by='xgb_score', ascending=False)
    df.to_csv(f'feature_importance_result_xgb_{dataset_name}.csv', index=False)
    logging.info('****** Done calculating Xgboost feature importance ******')

    logging.info('****** Start to calculate Logistic Regression feature importance ******')
    # 初始化基模型
    model = LogisticRegression()
    # 初始化RFE模型，这里使用所有特征数量作为要选择的特征数量，以获得所有特征的排名
    rfe = RFE(estimator=model, n_features_to_select=1)
    # 拟合RFE模型
    rfe.fit(X, y)
    df = pd.DataFrame({'feature_name': feature_map.features.keys(), 'rfe_score': rfe.ranking_})
    df = df.sort_values(by='rfe_score', ascending=True)
    df.to_csv(f'feature_importance_result_rfe_{dataset_name}.csv', index=False)
    logging.info('****** Done calculating Logistic Regression feature importance ******')


    logging.info('****** Start to calculate Random Forest feature importance ******')
    rf = RandomForestClassifier(n_estimators=100, random_state=42)  # 这里使用100棵树
    rf.fit(X, y)

    # 获取特征的重要性
    importances = rf.feature_importances_

    # 将特征名称和它们的重要性结合成 DataFrame
    feature_importance_df = pd.DataFrame({
        'feature_name': feature_map.features.keys(),  # 特征索引，假设特征没有具体名称
        'importance': importances
    })

    # 排序并显示特征的重要性
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    feature_importance_df.to_csv(f'feature_importance_result_rf_{dataset_name}.csv', index=False)
    logging.info('****** Done calculating Random Forest feature importance ******')

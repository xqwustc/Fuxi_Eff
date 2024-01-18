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

sys.path.append('../../..')
sys.path.append('../../../../')

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
import torch
from model_zoo.utils import cluster_features

if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    logging.info('Get training time, inference time and feature embedding params')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--num', type=int, help='selected fields num start from 0')
    parser.add_argument('--method', type=str, help='selected methods')
    args = vars(parser.parse_args())

    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']

    params['num'] = args.get('num', None)

    params['method'] = args['method']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    # seed_everything(seed=params['seed'])

    # email.common_send('DCN_incre.py',params["data_format"])

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
    if params.get('num') == None:
        # TODO: check the correctness
        params['num'] = len(feature_map.features.keys()) - 1

    train_time = None
    inf_time = None
    emb_params = None
    if params.get('method') == 'mvfs':
        logging.info('****** Warmup network without controller ******')
        warmup_path = f"{params['model']}_{params['dataset_id']}_warmup4mv.pth"
        model_class = getattr(model_zoo, params['model'])

        model = model_class(feature_map,**params)

        model.count_parameters()
        train_gen, valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()
        params_pretrain = copy.deepcopy(params)
        params_pretrain['epochs'] = args['epoch_pre']

        start_time = datetime.now()
        model.fit(train_gen, validation_data=valid_gen, **params_pretrain)
        train_time = (datetime.now() - start_time).seconds

        torch.save(model.state_dict(), warmup_path)
        logging.info('****** Warmup end and save model in {} ******'.format(warmup_path))
        del train_gen, valid_gen
        gc.collect()

        select_num = params['num']
        model_class = getattr(model_zoo, params['model'])
        print('params[model]', params['model'])
        model = model_class(feature_map, select_num=select_num + 1, **params)
        model.load_state_dict(torch.load(warmup_path), strict=False)

        model.count_parameters()  # print number of parameters used in model
        logging.info('Used Feature Number:{} '.format(select_num + 1))

        train_gen, valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()

        train_item = train_gen.num_samples
        valid_item = valid_gen.num_samples

        if args.get('cp', None) != None:
            print('load model from checkpoint')
            model.load_state_dict(torch.load(args['cp']))
        else:
            start_time = datetime.now()
            model.fit_for_mvfs(train_gen, validation_data=valid_gen, **params)
            train_time += (datetime.now() - start_time).seconds

        del train_gen, valid_gen
        gc.collect()

        logging.info('******** Test evaluation ********')
        test_gen = H5DataLoader(feature_map, stage='test', **params).make_iterator()

        test_item = test_gen.num_samples

        test_result = {}
        if test_gen:
            start_time = datetime.now()
            test_result = model.eval_mvfs(test_gen)
            inf_time = (datetime.now() - start_time).seconds

        del test_gen
        gc.collect()

        logging.info('summary for method {}'.format(params.get('method')))
        logging.info('#train: {}, #valid: {}, #test: {}'.format(train_item, valid_item, test_item))
        logging.info('train time: {}s, each item {:.6f}ms'.format(train_time, (train_time / train_item) * 1000))
        logging.info('inf time: {}s, each item {:.6f}ms'.format(inf_time, (inf_time / test_item) * 1000))
        logging.info('params num: {}'.format(model.get_total_parameters()))



    # 解析特征重要性的均值和方差作为先验
    else:
        df = pd.read_csv('feature_importance_result.csv')

        select_num = params['num']

        topk_params = copy.deepcopy(params)
        topk_params['use_features'] = df['feature_name'].values.tolist()[:select_num + 1]
        logging.info('--- Used Features: {} (totally {} features)'.format(topk_params['use_features'],
                                                                          len(topk_params['use_features'])))
        topk_feature_map = FeatureMap(topk_params['dataset_id'], data_dir)
        topk_feature_map.load(feature_map_json, topk_params)

        if '_cluster' in params['method']:
            cur_features_rows = df.iloc[:select_num + 1, :]
            cluster_df = cluster_features(cur_features_rows)
            # FIXME: DNN is 40
            size_list = [40, 20, 10]
            for idx, row in cluster_df.iterrows():
                cur_feat_name = row['feature_name']
                assert cur_feat_name in topk_feature_map.features.keys(), \
                    'feature_name {} not in topk_feature_map'.format(cur_feat_name)
                topk_feature_map.features[cur_feat_name]['embedding_dim'] = size_list[row['label']]

                # Output the size of each feature
                print('--- Feature {} with size {}'.format(cur_feat_name, size_list[row['label']]))

        model_class = getattr(model_zoo, topk_params['model'])
        topk_model = model_class(topk_feature_map, **topk_params)

        topk_model.count_parameters()  # print number of parameters used in model

        train_gen,valid_gen,test_gen = H5DataLoader(topk_feature_map, stage='both', **topk_params).make_iterator()
        train_item = train_gen.num_samples
        valid_item = valid_gen.num_samples
        test_item = test_gen.num_samples

        start_time = datetime.now()
        topk_model.fit(train_gen, validation_data=valid_gen, **params)
        train_time = (datetime.now() - start_time).seconds

        del train_gen, valid_gen
        gc.collect()

        start_time = datetime.now()
        valid_result = topk_model.evaluate(test_gen)
        inf_time = (datetime.now() - start_time).seconds

        del test_gen
        gc.collect()

        logging.info('summary for method {}'.format(params.get('method')))
        logging.info('#train: {}, #valid: {}, #test: {}'.format(train_item, valid_item, test_item))
        logging.info('train time: {}s, each item {:.6f}ms'.format(train_time, (train_time / train_item) * 1000))
        logging.info('inf time: {}s, each item {:.6f}ms'.format(inf_time, (inf_time / test_item) * 1000))
        logging.info('params num: {}'.format(topk_model.get_total_parameters()))
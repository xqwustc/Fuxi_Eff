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
import sys
os.chdir(os.path.dirname(os.path.realpath(__file__)))
current_dir = os.path.dirname(__file__)
fuxipac_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(fuxipac_dir)

import sys
import logging
# from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import H5SelectedDataLoader, H5DataLoader
from fuxictr.preprocess import FeatureProcessor, build_dataset
import src as model_zoo
import gc
import argparse
import os
from pathlib import Path
import pickle
import importlib


if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--keep_ratio', type=float, default=-1, help='The percentage ratio for filtering, between 0-1.')
    parser.add_argument('--train_batch', type=int, default=-1, help='The number of training batch to use.')

    args = vars(parser.parse_args())
    
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    # --- Update new params start ---
    keep_ratio = args.get('keep_ratio')
    if keep_ratio is not None and keep_ratio != -1:
        params['keep_ratio'] = keep_ratio

    train_batch = args.get('train_batch')
    if train_batch is not None and train_batch != -1:
        params['train_batch'] = train_batch
    # --- Update new params end ---


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
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))
    
    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model

    if params.get('optfs_dict') is not None:
        train_gen, valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()
        model_pretrain_path = os.path.splitext(model.checkpoint)[0] + ".pth"
        if os.path.exists(model_pretrain_path):
            model.load_weights(model_pretrain_path)
        else:
            model.fit_for_optfs(train_gen, validation_data=valid_gen, **params)
            model.save_weights(model_pretrain_path)
        seed_everything(seed=params['seed'])
        model.fit_for_optfs_with_ratio(train_gen, validation_data=valid_gen, **params)
    elif params.get('need_pretrain',False) == True:
        train_gen, valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()
        logging.info('Retraining the model')
        model.fit(train_gen, validation_data=valid_gen, **params)
    elif params.get('force_pretrain') == True or not os.path.exists(model.checkpoint):
        train_gen, valid_gen = H5SelectedDataLoader(feature_map, stage='train', **params).make_iterator()
        logging.info('Training the base model for scores...')
        model.fit(train_gen, validation_data=valid_gen, **params) # Do not use .fit_with_train_number here
    else:
        if params.get('train_ratio') is not None:
            train_gen, valid_gen = H5SelectedDataLoader(feature_map, stage='train', **params).make_iterator()
        else:
            train_gen, valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()
        model.load_weights(model.checkpoint)
        if params.get('pep_dict') is not None:
            logging.info(f"Sparsity of the embedding matrix: {model.cal_sparsity()}")

    if params.get("autofeat_mode") in ['batch', 'table']:
        logging.info('****** Validation with AutoFeat ******')
        valid_result = model.evaluate_with_autofeat(valid_gen)
        del train_gen, valid_gen
        gc.collect()


    logging.info('******** Test evaluation ********')
    test_gen = H5DataLoader(feature_map, stage='test', **params).make_iterator()
    test_result = {}
    if test_gen:
        if params.get('keep_ratio') is not None:
            logging.info(f'****** Test with Keep Ratio {params["keep_ratio"]} ******')
        test_result = model.evaluate(test_gen)



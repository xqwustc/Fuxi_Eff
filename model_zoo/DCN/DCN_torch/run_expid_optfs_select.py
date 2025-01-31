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
import os

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

    model_class = getattr(model_zoo, params['model'])
    print('params[model]', params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters()  # print number of parameters used in model

    train_gen, valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()

    if args.get('cp',None) != None:
        print('load model from checkpoint')
        model.load_state_dict(torch.load(args['cp']))
    else:
        # .pth is specifically for optfs model retrain save
        model_pretrain_path = os.path.splitext(model.checkpoint)[0] + ".pth"
        if os.path.exists(model_pretrain_path):
            model.load_weights(model_pretrain_path)
        else:
            model.fit_for_optfs(train_gen, validation_data=valid_gen, **params)
            model.save_weights(model_pretrain_path)
        seed_everything(seed=params['seed'])
        model.fit_for_optfs_with_ratio(train_gen, validation_data=valid_gen, **params)
    del train_gen, valid_gen
    gc.collect()

    test_gen = H5DataLoader(feature_map, stage='test', **params).make_iterator()
    test_result = {}
    if test_gen:
        if params.get('keep_ratio') is not None:
            logging.info(f'****** Test with Keep Ratio {params["keep_ratio"]} ******')
        test_result = model.evaluate(test_gen)
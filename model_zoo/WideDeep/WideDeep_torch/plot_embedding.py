import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys

sys.path.append('../../../fuxictr')
sys.path.append('../../../')

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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import matplotlib

if __name__ == '__main__':
    ''' Usage: Must run after pretrained model is trained. 
        python run_expid.py --config {config_dir} --expid {model_feat_dataset} --gpu {gpu_device_id}
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
    model.count_parameters()  # print number of parameters used in model

    try:
        model.load_weights(model.checkpoint)
    except:
        logging.info("No checkpoint found. Please train the model first.")
        exit()

    embedding_layers = model.embedding_layer.embedding_layer.embedding_layers

    # 为了可视化，先把所有的嵌入信息提取到一个列表中
    embedding_data = []
    labels = []

    zero_embeddings = []
    mean_embeddings = []

    fontsize = 17
    # 遍历每个field的嵌入层
    times = 6
    for field, embedding_layer in embedding_layers.items():
        if field not in ['C17', 'app_domain', 'C20']:
            continue
        # 提取嵌入层的权重（embedding weights）
        embeddings = embedding_layer.weight.data.cpu().numpy()

        # 添加这些嵌入到embedding_data中，并且为它们创建标签
        embedding_data.append(embeddings)
        labels.extend([field] * embeddings.shape[0])

        # 计算均值嵌入
        mean_embedding = np.mean(embeddings, axis=0)
        mean_embeddings.append(mean_embedding)

        # 生成全零嵌入
        zero_embedding = np.zeros_like(mean_embedding)
        zero_embeddings.append(zero_embedding)

        if times > 0:
            times -= 1
        else:
            break

    # 将所有嵌入拼接成一个大矩阵
    embedding_data = np.concatenate(embedding_data, axis=0)

    mean_embeddings = np.array(mean_embeddings)
    zero_embeddings = np.array(zero_embeddings)

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    X_tsne = tsne.fit_transform(embedding_data)

    # 绘制t-SNE图
    plt.figure(figsize=(8, 6))

    # 为每个field分配一个颜色
    unique_fields = list(set(labels))
    colors = plt.cm.get_cmap('tab10', len(unique_fields))

    for idx, field in enumerate(unique_fields):
        # 获取当前field的所有样本索引
        field_indices = [i for i, label in enumerate(labels) if label == field]

        # 获取对应的t-SNE坐标
        field_tsne = X_tsne[field_indices]

        # 绘制这些点，并为每个field选择不同的颜色
        plt.scatter(field_tsne[:, 0], field_tsne[:, 1], label=field,
                    color=colors(idx), s=40)

    # # 计算均值和全零嵌入的t-SNE
    # mean_tsne = tsne.fit_transform(mean_embeddings)
    # zero_tsne = tsne.fit_transform(zero_embeddings)

    # 添加图例、标题和标签
    plt.title('t-SNE visualization of Embedding Fields', fontsize=fontsize)
    plt.xlabel('t-SNE component 1', fontsize=fontsize)
    plt.ylabel('t-SNE component 2', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)

    # 图例字体大小
    plt.legend(fontsize=fontsize)

    # pdf
    plt.savefig('embedding_tsne.pdf', dpi=300, bbox_inches='tight')
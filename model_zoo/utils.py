import random

import math
import torch.nn as nn
import torch
from math import log
import pandas as pd
import numpy as np
EPS = 1e-8
from statistics import NormalDist
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids

# def get_gate_vector(feature_size):
#     # the vector size is (1, \sigma{feature_emb_size}), each time call this function will make different gates
#     vec = torch.rand(0)
#     for i in range(self.feature_size[0]): # each feature has feature_emb_size
#         vec = torch.cat([vec, self.make_prob(self.theta[i])*self.feature_size[0]], dim=0)
#
# def get_gates_prob(gates_theta):
#     gates_prob = gates_theta.clone()
#     for i in range(gates_theta.shape[0]):
#         gates_prob[i] = get_prob(gates_theta[i])
#         gates_prob[i].requires_grad = True
#
#     return gates_prob
#
# def get_prob(unit):
#     u = torch.rand(1)
#     u.requires_grad = False
#
#     return torch.sigmoid((1.0/self.temperature) * (log(unit+EPS) - log(1-unit+EPS) + log(u+EPS) - log(1-u+EPS)))
def cluster_features(features: pd.DataFrame, group_num=3):
    # cluster given features into {group_num} groups
    # the first column of features should be feature name, the second column should be miu
    # and the third column should be sigma
    n = len(features)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = 1 - calculate_overlap(features.iloc[i, 1], features.iloc[i, 2], features.iloc[j, 1], features.iloc[j, 2])

    # kmeans = KMeans(n_clusters=group_num, random_state=0).fit(distance_matrix)
    clustered_df = features.copy()
    # clustered_df['label'] = kmeans.labels_

    kmedoids = KMedoids(n_clusters=3, metric='precomputed', random_state=0)
    # 适配模型
    kmedoids.fit(distance_matrix)
    # 获取聚类标签
    clustered_df['label'] = kmedoids.labels_

    min_indices = clustered_df.reset_index().groupby('label')['index'].min()

    # 然后，根据这些最小下标对 label 进行排序
    sorted_labels = min_indices.sort_values().index

    # 创建一个映射，将旧的 label 映射到新的 label
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
    clustered_df['label'] = clustered_df['label'].map(label_mapping)

    return clustered_df

def get_embsize_by_vocab(feature_map):
    size_map = dict()
    for feature_name, feature_info in feature_map.features.items():
        if feature_info.get('vocab_size', None) is not None:
            size_map[feature_name] = int(feature_info['vocab_size']**0.25)
    return size_map

def get_embsize_by_pca(cur_model,ratio=0.95):
    size_map = dict()
    emb_dict = cur_model.embedding_layer.embedding_layer.embedding_layers
    for feature_name, emb in emb_dict.items():
        cur_emb = emb.weight.detach().cpu().numpy()
        pca = PCA(n_components=ratio)
        pca.fit(cur_emb)
        size_map[feature_name] = pca.n_components_
    return size_map

def calculate_overlap(mu1, sigma1, mu2, sigma2):
    return NormalDist(mu=mu1, sigma=sigma1).overlap(NormalDist(mu=mu2, sigma=sigma2))


def get_gates_prob_autofield(gates_theta,epoch = None):
    gates_prob = gates_theta.clone()
    for i in range(gates_theta.shape[0]):
        gates_prob[i] = get_prob_autofield(gates_theta[i],epoch)
    return gates_prob

def get_prob_autofield(unit, epoch = None):
    t = max(0.01, 1 - 5e-5 * epoch)
    u1 = torch.rand(1)
    u0 = torch.rand(1)

    g1 = -torch.log(-torch.log(u1))
    g0 = -torch.log(-torch.log(u0))

    exp_term_denominator_1 = torch.exp((torch.log(unit) + g1) / t)
    exp_term_denominator_0 = torch.exp((torch.log(1 - unit) + g0) / t)

    # Calculate the probability p_n^j
    p_n_j = exp_term_denominator_1 / (exp_term_denominator_1 + exp_term_denominator_0)

    return p_n_j

def get_gates_prob_my(gates_theta,gates_sigma,epoch = None):
    assert len(gates_theta) == len(gates_sigma), "gates_theta and gates_sigma should have the same length"
    gates_prob = gates_theta.clone()
    for i in range(gates_theta.shape[0]):
        gates_prob[i] = get_prob_my(gates_theta[i],gates_sigma[i],epoch)
    return gates_prob

def get_prob_my(unit,sigma_unit,epoch = None):
    if epoch is None:
        t = 0.5
    else:
        # t = max(0.1, 1 - 5e-5 * (epoch**1.5))
        t = 0.5

    eps = torch.randn(1)*sigma_unit
    # u = torch.randn(1)*sigma_unit
    # u.requires_grad = False
    # u = u.to(device=self.device)

    return torch.sigmoid((1.0 / t) * (unit + eps))

def get_sum_feature_dimisions(embedding_layer):
    total_dimensions = 0
    for layer in embedding_layer.embedding_layer.embedding_layers.values():
        if isinstance(layer,nn.Embedding):
            total_dimensions += layer.embedding_dim
        elif isinstance(layer,nn.Linear):
            total_dimensions += layer.out_features
        else:
            raise TypeError(f"Unsupported layer type {type(layer)} for layer {name}")
    return total_dimensions

def create_unique_vector(a, b):
    if b > a:
        raise ValueError("b cannot be greater than a.")
    random_numbers = random.sample(range(a), b)
    vector = np.array(random_numbers)
    return vector.tolist()

def permute_feature(data_generator, feature_idx):
    """
    Permutes the values of a specific feature in each batch produced by the data_generator.

    Args:
    - data_generator: Original data generator.
    - feature_idx: The index of the feature you want to permute.

    Yields:
    - Batch with permuted feature values.
    """
    if not isinstance(feature_idx, list):
        feature_idx = [feature_idx]

    for batch in data_generator:
        # Deep copy to avoid modifying the original batch
        permuted_batch = batch.clone()

        # Permute the feature using PyTorch functions
        perm = torch.randperm(permuted_batch.size(0))
        permuted_batch[:, feature_idx] = permuted_batch[perm, :][:, feature_idx]

        yield permuted_batch
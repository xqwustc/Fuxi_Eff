import math
import torch.nn as nn
import torch
from math import log
import pandas as pd
import numpy as np
EPS = 1e-8
from statistics import NormalDist
from sklearn.cluster import KMeans

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

    kmeans = KMeans(n_clusters=group_num, random_state=0).fit(distance_matrix)

    clustered_df = features.copy()
    clustered_df['label'] = kmeans.labels_

    min_indices = clustered_df.reset_index().groupby('label')['index'].min()

    # 然后，根据这些最小下标对 label 进行排序
    sorted_labels = min_indices.sort_values().index

    # 创建一个映射，将旧的 label 映射到新的 label
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
    clustered_df['label'] = clustered_df['label'].map(label_mapping)

    return clustered_df


def calculate_overlap(mu1, sigma1, mu2, sigma2):
    return NormalDist(mu=mu1, sigma=sigma1).overlap(NormalDist(mu=mu2, sigma=sigma2))



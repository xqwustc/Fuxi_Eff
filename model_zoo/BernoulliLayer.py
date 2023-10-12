import math
import torch.nn as nn
import torch
from math import log
EPS = 1e-8
class BernoulliLayer:
    def __init__(self, feature_size,temperature=0.1):
        # feature_size = (feature_num, feature_emb_size)
        assert temperature > 0, "Temperature must be positive"
        self.temperature = temperature
        self.feature_size = feature_size
        self.theta = torch.rand(feature_size.shape[0])  # each feature has feature_emb_size, each of them
        # shares the same theta_i, and theta_i requires gradient

    def get_gate_vector(self,feature_size):
        # the vector size is (1, \sigma{feature_emb_size}), each time call this function will make different gates
        for i in range
        vec = torch.rand(0)
        for i in range(self.feature_size[0]): # each feature has feature_emb_size
            vec = torch.cat([vec, self.make_prob(self.theta[i])*self.feature_size[0]], dim=0)
    def make_prob(self,units):
        u = torch.rand(1)
        u.requires_grad = False
        return torch.sigmoid((1.0/self.temperature) * (log(unit+EPS) - log(1-unit+EPS) + log(u+EPS) - log(1-u+EPS)))


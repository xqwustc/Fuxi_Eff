import math
import torch.nn as nn
import torch
from math import log
EPS = 1e-8
def get_gate_vector(feature_size):
    # the vector size is (1, \sigma{feature_emb_size}), each time call this function will make different gates
    vec = torch.rand(0)
    for i in range(self.feature_size[0]): # each feature has feature_emb_size
        vec = torch.cat([vec, self.make_prob(self.theta[i])*self.feature_size[0]], dim=0)

def get_gates_prob(gates_theta):
    gates_prob = gates_theta.clone()
    for i in range(gates_theta.shape[0]):
        gates_prob[i] = get_prob(gates_theta[i])
        gates_prob[i].requires_grad = True

    return gates_prob

def get_prob(unit):
    u = torch.rand(1)
    u.requires_grad = False

    return torch.sigmoid((1.0/self.temperature) * (log(unit+EPS) - log(1-unit+EPS) + log(u+EPS) - log(1-u+EPS)))

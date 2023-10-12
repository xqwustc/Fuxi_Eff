import pandas as pd
import numpy as np
import torch
from torch import log
# file_path = 'D:\Downloads\Criteo_x4\\'
#
# file = pd.read_csv(file_path + 'test.csv', sep=',', header=None, nrows=1000000)
EPS = 1e-8

def get_gates_prob(gates_theta):
    gates_prob = gates_theta.clone()
    for i in range(gates_theta.shape[0]):
        gates_prob[i] = get_prob(gates_theta[i])
    return gates_prob

def get_prob(unit):
    u = torch.rand(1)
    u.requires_grad = False
    return torch.sigmoid((1.0/0.1) * (log(unit+EPS) - log(1-unit+EPS) + log(u+EPS) - log(1-u+EPS)))
def evaluate_with_dr(seed=2019):
    # Make a list of theta for each feature
    gates_theta = torch.ones(100) * 0.5
    gates_theta.requires_grad_(requires_grad=True)
    gates_prob = get_gates_prob(gates_theta)
    x = torch.tensor(np.random.rand(100), dtype=torch.float32)
    for i in range(100):
        print(i)
        y = torch.tensor(10.0)
        pred_y = torch.dot(gates_prob,x)
        loss = torch.nn.MSELoss()
        loss(pred_y, y).backward()
        #print(gates_theta)
        gates_theta.data -= 0.01 * gates_theta.grad.data
        gates_theta.grad.data.zero_()

def Detach():
    a = torch.ones(1, requires_grad=True)
    c = torch.tensor(10.0,requires_grad=True)
    b = a*2
    b = b.detach()
    # d = a.detach()
    y = b**4 + 2*c**2
    y.backward()
    print(c.grad)

evaluate_with_dr()
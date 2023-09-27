import pandas as pd
import numpy as np

file_path = 'D:\Downloads\Criteo_x4\\'

file = pd.read_csv(file_path + 'test.csv', sep=',', header=None, nrows=1000000)

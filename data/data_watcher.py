import pandas as pd
import os

# 假设 RANDOM_SEED 已经定义
RANDOM_SEED = 2022

# 读取数据
data_path = './ifly_chu'
# data_path = './frappe_x1'

# 保存数据到CSV
test_df  = pd.read_csv(os.path.abspath(os.path.join(data_path, 'test.csv')))
valid_df = pd.read_csv(os.path.abspath(os.path.join(data_path, 'valid.csv')))
train_df = pd.read_csv(os.path.abspath(os.path.join(data_path, 'train.csv')))


# Pos ratio in train
print('Pos ratio in train:', train_df['label'].sum() / len(train_df))
# Pos ratio in valid
print('Pos ratio in valid:', valid_df['label'].sum() / len(valid_df))
# Pos ratio in test
print('Pos ratio in test:', test_df['label'].sum() / len(test_df))
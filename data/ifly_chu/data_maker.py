import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter

# 假设 RANDOM_SEED 已经定义
RANDOM_SEED = 2022

# 读取数据
train_file = "train_data.txt"
columns = ['label'] + ["f{}".format(i + 1) for i in range(245)]
train_data = pd.read_csv(train_file, header=None, names=columns, dtype=object, memory_map=True)
train_data.drop([1613993, 1895025], axis=0, inplace=True)
train_data.reset_index(drop=True, inplace=True)
train_data['label'] = train_data['label'].astype(int)

for col in ["f1", "f2", "f3", "f4"]:
    col_dict = Counter(train_data[col])
    vocab = dict(zip(col_dict.keys(), range(len(col_dict))))
    train_data[col] = train_data[col].map(lambda x: vocab[x])
print(train_data.head())

# 分层采样
X = train_data.values
y = train_data['label'].map(lambda x:float(x)).values

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED).split(X, y)

fold_indexes = []
for train_id, valid_id in folds:
    fold_indexes.append(valid_id)

# 按照8:1:1的比例分配数据
test_index = fold_indexes[0]
valid_index = fold_indexes[1]
train_index = np.concatenate(fold_indexes[2:])

test_df = train_data.loc[test_index, :]
valid_df = train_data.loc[valid_index, :]
train_df = train_data.loc[train_index, :]

# 保存数据到CSV
test_df.to_csv('test.csv', index=False)
valid_df.to_csv('valid.csv', index=False)
train_df.to_csv('train.csv', index=False)

# 打印统计信息
print('Train lines:', len(train_index))
print('Validation lines:', len(valid_index))
print('Test lines:', len(test_index))

# Pos ratio in train
print('Pos ratio in train:', train_df['label'].sum() / len(train_df))
# Pos ratio in valid
print('Pos ratio in valid:', valid_df['label'].sum() / len(valid_df))
# Pos ratio in test
print('Pos ratio in test:', test_df['label'].sum() / len(test_df))
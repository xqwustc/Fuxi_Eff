import pandas as pd
from sklearn.model_selection import train_test_split

data_path = './ipinyou_x2'

# 读取 train.csv 文件的前 100 行
train_df = pd.read_csv(f'{data_path}/train_old.csv')

# 进行训练集和验证集的划分（7:1比例）
train_df_split, valid_df_split = train_test_split(train_df, test_size=1.0/7, random_state=2024)

# 保存数据
train_df_split.to_csv(f'{data_path}/train.csv', index=False, encoding='utf-8')
valid_df_split.to_csv(f'{data_path}/valid.csv', index=False, encoding='utf-8')

# 输出划分信息
print('Train lines:', len(train_df_split))
print('Validation lines:', len(valid_df_split))
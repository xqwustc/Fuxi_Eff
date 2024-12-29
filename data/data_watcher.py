import pandas as pd
import os

# 假设 RANDOM_SEED 已经定义
RANDOM_SEED = 2022
dataset_name = 'AliCCP'
if dataset_name in ['ifly_chu', 'frappe_x1']:
    # 读取数据
    data_path = f'./{dataset_name}/'
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

elif dataset_name == 'AliCCP':
    data_path = '/data/STEVEN8868/FuxiCTR/data/aliccp_x4/AliCCP_x4_origin/ali_ccp_train.csv'

    df_head_20 = pd.read_csv(data_path, nrows=20)
    print(df_head_20)




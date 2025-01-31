#!/bin/bash

# 定义xx的取值范围
values=(1 0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
timestamp=$(date +"%Y%m%d_%H%M%S")
# 遍历所有xx的值
for xx in "${values[@]}"
do
    # 打印当前正在执行的xx值
    echo "Running with keep_ratio=${xx}"

    # 执行python脚本并将输出写入日志文件
    python /data/STEVEN8868/FuxiCTR/model_zoo/WideDeep/WideDeep_torch/run_expid_prefeat.py --expid WideDeep_feat_retrain_avazu --gpu 4 --keep_ratio ${xx} >> "WideDeep_feat_retrain_avazu_${timestamp}.log" 2>&1

    # 在日志文件中添加前后换行的分隔线
    echo -e "\n----------------------------------------------------Running done with keep_ratio=${xx}\n" >> "WideDeep_feat_retrain_avazu_${timestamp}.log"
done

#!/bin/bash

# 定义xx的取值范围
values=(0.1 0.5 0.7)
#values=(1.0)
timestamp=$(date +"%Y%m%d_%H%M%S")
# 遍历所有xx的值
for xx in "${values[@]}"
do
    # 打印当前正在执行的xx值
    echo "Running with keep_ratio=${xx}"

    # 执行python脚本并将输出写入日志文件
    python /data/STEVEN8868/FuxiCTR/model_zoo/DCN/DCN_torch/run_expid_prefeat.py --expid DCN_feat_retrain_criteo --gpu 0 --keep_ratio ${xx} >> "DCN_feat_retrain_criteo_${timestamp}.log" 2>&1

    # 在日志文件中添加前后换行的分隔线
    echo -e "\n----------------------------------------------------Running done with keep_ratio=${xx}\n" >> "DCN_feat_retrain_criteo_${timestamp}.log"
done

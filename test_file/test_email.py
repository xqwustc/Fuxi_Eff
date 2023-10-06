import sys
sys.path.append('../infomer')
from infomer import email
import time

# 设置循环次数
num_iterations = 3  # 例如，执行10次

for i in range(num_iterations):
    # 执行您的循环操作
    email.common_send('test.py - {} times'.format(i), print(f"Iteration {i + 1}"))

    # 暂停3秒
    time.sleep(5)
import numpy as np
import matplotlib.pyplot as mpl
from datetime import datetime

# 学習させるデータを指定する
# target_path = "20220512_182222"
# 該当フォルダの日付（計測日と異なる場合は手入力）
target_date = datetime.now().strftime("%Y-%m-%d")
# target_date = "2022-05-13"

# 該当ファイルの時刻を手入力
target_time = "13-11-05"

vector = np.load("dataset/" + target_date + "/forMachineLearning/" + target_time + ".npy")

print(vector.shape)

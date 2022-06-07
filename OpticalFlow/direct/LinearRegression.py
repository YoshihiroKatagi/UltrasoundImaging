import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import csv

#####################  Path and Parameter  ###########################
# 使用するUSデータの形状
default_US_shape = (898, 100)

# 該当フォルダの日付（計測日と異なる場合は手入力）
target_date = "2022-05-30"
# target_date = datetime.now().strftime("%Y-%m-%d")

target_path = "C:/Users/katagi/Desktop/Workspace/Research/UltrasoundImaging/OpticalFlow/direct/dataset/" + target_date + "/forMachineLearning/"

# 検証するパターン（12パターン × 10試行 = 120個）
patern = 5 # 1~12(1, 2, 3, 4, 8, 11, 12)

# 結果保存用パス
result_folder = "dataset/" + target_date + "/results"
result_path = result_folder + "/patern" + str(patern)

if not os.path.exists(result_path):
  os.makedirs(result_path)

with open(result_path + "/R2andRMSE.csv", "w") as f:
  writer = csv.writer(f)
  writer.writerow(["RMSE", "R2"])
#####################################################################

###########################  Read Data  #############################
def ReadData(): # 10試行分（どのパターンかは上のパラメータで指定する）
  target_US = os.listdir(target_path)
  US_data_block = np.empty([0, default_US_shape[0], default_US_shape[1]])
  for i in range(10):
    data_num = 10*(patern - 1) + i
    US_data_path = target_path + target_US[data_num]
    US_data = np.load(US_data_path)
    if US_data.shape == default_US_shape:
      US_data = US_data.reshape([1, US_data.shape[0], US_data.shape[1]])
      US_data_block = np.append(US_data_block, US_data, axis=0)

  gonio_data_path = target_path + "gonioData.npy"
  gonio_data = np.load(gonio_data_path)
  gonio_data = gonio_data[10*(patern - 1) : 10*patern]
  return US_data_block, gonio_data
#####################################################################

############################  Analysis  #############################
def Analysis(X, theta): #X: (8, 898, 100), theta: (8, 898, 1)
  W = list()
  X = np.transpose(X, (1, 0, 2)) # (898, 8, 100)
  theta = np.transpose(theta, (1, 0, 2)) #(898, 8, 1)
  for X_t, theta_t in zip(X, theta):
    X_T_t = X_t.T
    X_T_X_inv = np.linalg.pinv(np.dot(X_T_t, X_t))
    W_t = np.dot(np.dot(X_T_X_inv, X_T_t), theta_t)
    W.append(W_t)
  W = np.array(W)
  W = W.reshape([X.shape[0], X.shape[2]]) # (898, 100)
  return W
#####################################################################

###########################  Visualize  #############################
def Visualize(y, y_pred, n):
  x = np.arange(y.shape[0])
  y = y.reshape(-1)
  y_pred = y_pred.reshape(-1)

  fig1 = plt.figure()
  plt.title("Wrist angle")
  plt.xlabel("Time")
  plt.ylabel("angle")
  plt.plot(x, y_pred, color="cornflowerblue", linewidth=2, label="Estimated angle")
  plt.plot(x, y, color="tomato", linewidth=2, label="Measured angle")
  plt.ylim(-80, 30) #extensor
  plt.xlim(0, y.shape[0])
  plt.legend(loc="upper left", fontsize=10)
  plt.grid(True)
  fig1.savefig(result_path + "/plot" + str(n) + ".png")

  fig2 = plt.figure()
  plt.title("scatter plot")
  plt.scatter(y, y_pred)
  fig2.savefig(result_path + "/scatter" + str(n) + ".png")

  plt.show()
#####################################################################

########################  Calc_R2andRMSE  ###########################
def Calc_R2andRMSE(theta, theta_pred, T):
  theta = theta.reshape(-1)

  def _R2(theta, theta_pred):
    corrcoef = np.corrcoef(theta, theta_pred)
    corrcoef = corrcoef[0][1]
    return corrcoef
  
  def _RMSE(theta, theta_pred, T):
    L = np.sum((theta - theta_pred)**2)
    RMSE = np.sqrt(L/T)
    return RMSE
  
  # print(theta.shape, theta_pred.shape) # (898,), (898,)

  RMSE = _RMSE(theta, theta_pred, T)
  R2 = _R2(theta, theta_pred)
  print("RMSE = " + str(RMSE))
  print("相関係数: " + str(R2) + "\n")

  with open(result_path + "/R2andRMSE.csv", "a") as f:
    writer = csv.writer(f)
    writer.writerow([RMSE, R2])
#####################################################################

#############################  Main  ################################
Xs, Thetas = ReadData()
# print("X, Theta: ")
# print(Xs.shape, Thetas.shape) # (10, 898, 1), (10, 898, 100)
T = Xs.shape[1]
for i in range(5):
  if i == 0:
    X_train, Theta_train = Xs[2:], Thetas[2:] # (8, 898, 100), (8, 898, 1)
  elif i == 4:
    X_train, Theta_train = Xs[:-2], Thetas[:-2]
  else:
    X_train, Theta_train = np.concatenate([Xs[:i*2], Xs[(i+1)*2:]]), np.concatenate([Thetas[:i*2], Thetas[(i+1)*2:]])
  X_test, Theta_test = Xs[i*2 : (i+1)*2], Thetas[i*2 : (i+1)*2] # (2, 898, 100), (2, 898, 1)

  W = Analysis(X_train, Theta_train) # (898, 100)
  
  # print(Theta_pred.shape)
  for j in range(len(X_test)):
    # パターン内のテスト番号(1~10)
    test_num = 2*i + j + 1
    print("test" + str(test_num) + ":")

    Theta_pred = list()
    for X_t, W_t in zip(X_test[j], W):
      Theta_pred.append(np.dot(W_t, X_t))
    Theta_pred = np.array(Theta_pred) # (898, 1)
    Visualize(Theta_test[j], Theta_pred, test_num)
    Calc_R2andRMSE(Theta_test[j], Theta_pred, T)
#####################################################################
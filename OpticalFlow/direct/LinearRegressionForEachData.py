# 交差検証をせず1つの試行で学習・テストを行う


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
patern = 1 # 1~12(1, 2, 3, 4, 8, 11, 12)

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
  feature_points_data_path = target_path + "FeaturePointsData.npy"
  feature_points_data = np.load(feature_points_data_path)
  feature_points_data = feature_points_data[10*(patern - 1): 10*patern, :, :100]
  # print(feature_points_data.shape)

  gonio_data_path = target_path + "gonioData.npy"
  gonio_data = np.load(gonio_data_path)
  gonio_data = gonio_data[10*(patern - 1) : 10*patern]

  return feature_points_data, gonio_data
#####################################################################

###################  Divide into Train and Test  ####################
def DivideIntoTrainAndTest(x, theta):# Train:(4, 179, 100), (4, 179, 1)  Test:(179, 100), (179, 1)
  x, theta = x[3:], theta[3:]
  x = x.reshape(5, x.shape[0] // 5, x.shape[1])
  theta = theta.reshape(5, theta.shape[0] // 5, theta.shape[1])
  x_train, theta_train = x[:4], theta[:4]
  x_test, theta_test = x[4], theta[4]

  return x_train, theta_train, x_test, theta_test
#####################################################################

############################  Analysis  #############################
def Analysis(x, theta): #x: (718, 100), theta: (718, 1)
  W = list()
  x = np.transpose(x, (1, 0, 2)) # (179, 4, 100)
  theta = np.transpose(theta, (1, 0, 2)) #(179, 4, 1)
  for x_t, theta_t in zip(x, theta):
    x_t_T = x_t.T
    x_T_x_inv = np.linalg.pinv(np.dot(x_t_T, x_t))
    W_t = np.dot(np.dot(x_T_x_inv, x_t_T), theta_t)
    W.append(W_t)
  W = np.array(W)
  W = W.reshape([x.shape[0], x.shape[2]]) # (179, 100)

  return W
#####################################################################

###########################  Visualize  #############################
def Visualize(y, y_pred, n):
  x = np.arange(y.shape[0])
  y = y.reshape(-1) # (179,)

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

  # plt.show()
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

  RMSE = _RMSE(theta, theta_pred, T)
  R2 = _R2(theta, theta_pred)
  print("RMSE = " + str(RMSE))
  print("相関係数: " + str(R2) + "\n")

  with open(result_path + "/R2andRMSE.csv", "a") as f:
    writer = csv.writer(f)
    writer.writerow([RMSE, R2])
#####################################################################

#############################  Main  ################################
print("-----Patern" + str(patern) + "-----\n")

Xs, Thetas = ReadData() # (10, 898, 100), (10, 898, 1)
T = Xs.shape[1]
for i in range(10):
  test_num = i + 1
  print("--test" + str(test_num) + "--")

  X, Theta = Xs[i], Thetas[i] # (898, 100), (898, 1)
  X_train, Theta_train, X_test, Theta_test = DivideIntoTrainAndTest(X, Theta) # (718, 100), (718, 1), (898, 100), (898, 1)
  W = Analysis(X_train, Theta_train) # (1, 100)

  Theta_pred = list()
  for X_t, W_t in zip(X_test, W):
    Theta_pred.append(np.dot(W_t, X_t))
  Theta_pred = np.array(Theta_pred) # (179,)
  Visualize(Theta_test, Theta_pred, test_num)
  Calc_R2andRMSE(Theta_test, Theta_pred, T)
#####################################################################
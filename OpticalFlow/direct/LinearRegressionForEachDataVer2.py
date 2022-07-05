# LinearRegressionForEachData Ver2
# difference:
# Ver1: Wが間違っている(179, 100) 時系列は関係ないはず
# Ver2: Wを修正(1, 100) そのためにAnalysis()を修正
# Ver2: 訓練データを含む全データでテストを行うX_test = X
# Ver2: 12パターンでループ
# Ver2: 特徴点の動作をチェックする関数作成 CheckFeature()
# Ver2: 特徴点にハイパスフィルタをかける HighpassFilter()


import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import cv2
from matplotlib.animation import ArtistAnimation

#####################  Path and Parameter  ###########################
# 使用するUSデータの形状
default_US_shape = (898, 100)

# 該当フォルダの日付（計測日と異なる場合は手入力）
target_date = "2022-05-30"
# target_date = datetime.now().strftime("%Y-%m-%d")

target_path = "C:/Users/katagi/Desktop/Workspace/Research/UltrasoundImaging/OpticalFlow/direct/dataset/" + target_date + "/forMachineLearning/"

# 検証するパターン数（12パターン × 10試行 = 120個）
patern_num = 12
#####################################################################

#######################  Make Results Path  #########################
# 結果保存用パス
def MakeResultsPath(patern):
  result_folder = "dataset/" + target_date + "/results"
  result_path = result_folder + "/patern" + str(patern)

  if not os.path.exists(result_path):
    os.makedirs(result_path)

  with open(result_path + "/R2andRMSE.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["RMSE", "R2"])
  
  return result_path
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
def DivideIntoTrainAndTest(x, theta):# Train:(718, 100), (718, 1)  Test:(898, 100), (898, 1)
  x_train, theta_train = x[:718], theta[:718]
  x_test, theta_test = x, theta

  return x_train, theta_train, x_test, theta_test
#####################################################################

############################  Analysis  #############################
def Analysis(x, theta): #x: (718, 100), theta: (718, 1)
  x_T = x.T
  x_T_x_inv = np.linalg.pinv(np.dot(x_T, x))
  W =  np.dot(np.dot(x_T_x_inv, x_T), theta) # (100, 1)

  return W
#####################################################################

###########################  Visualize  #############################
def Visualize(y, y_pred, n):
  x = np.arange(y.shape[0])
  y = y.reshape(-1) # (898,)
  y_pred = y_pred.reshape(-1) # (898,)

  fig1 = plt.figure()
  plt.title("Wrist angle")
  plt.xlabel("Time")
  plt.ylabel("angle")
  plt.plot(x, y_pred, color="cornflowerblue", linewidth=2, label="Estimated angle")
  plt.plot(x, y, color="tomato", linewidth=2, label="Measured angle")
  plt.vlines(718, -80, 30, "gray", linestyles="dashed")
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
  plt.close()
#####################################################################

########################  Calc_R2andRMSE  ###########################
def Calc_R2andRMSE(theta, theta_pred, T):
  theta = theta.reshape(-1)
  theta_pred = theta_pred.reshape(-1)

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

########################  Check Feature  ###########################
def CheckFeature(n):
  feature_path = target_path + "FeaturePointsData.npy"
  feature_data = np.load(feature_path)
  feature_data = feature_data[:n, :, :100]
  feature_data = feature_data.reshape([n, feature_data.shape[1], 50, 2]) # (3, 898, 50, 2)
  # print(feature_data[0, 0])
  gonio_data_path = target_path + "gonioData.npy"
  gonio_data = np.load(gonio_data_path)
  gonio_data = gonio_data[:n, :, ] # (3, 898, 1)

  height = 480
  width = 640
  # img = np.full((height, width, 3), 0, np.uint8)
  mask = np.zeros((height, width, 3), np.uint8)
  # for save
  fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  save = cv2.VideoWriter("test.mp4", fmt, 30, (width, height))

  feature_pre = feature_data[0, 0]
  for t in range(feature_data.shape[1]):
    if t+1 == feature_data.shape[1]:
      break
    img = np.zeros((height, width, 3), np.uint8)
    gonio_now = round(gonio_data[0, t, 0], 2)
    feature_now = feature_data[0, t+1]

    for i in range(feature_data.shape[2]):
    # for i in range(10):
      mask = cv2.line(mask, (int(feature_pre[i][0]), int(feature_pre[i][1])), (int(feature_now[i][0]), int(feature_now[i][1])), [128, 128, 128], 1)
      img = cv2.circle(img, (int(feature_now[i][0]), int(feature_now[i][1])), 5, [0, 0, 200], -1)

    # テキストデータ描画
    angle_data = "Wrist Angle: " + str(gonio_now)
    org = (5, 400)
    cv2.putText(img, angle_data, org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.0, color=(255, 255, 255))

    image = cv2.add(img, mask)
    # cv2.imshow("mask", image)
    save.write(image)

    feature_pre = feature_now
  
  cv2.destroyAllWindows()
  exit()

#####################################################################

#############################  Main  ################################
for p in range(patern_num):
  patern = p + 1
  result_path = MakeResultsPath(patern)
  print("-----Patern" + str(patern) + "-----\n")

  # 特徴点の再描画 + 関節角度
  # CheckFeature(3)

  Xs, Thetas = ReadData() # (10, 898, 100), (10, 898, 1)
  T = Xs.shape[1]
  for i in range(10):
    test_num = i + 1
    print("--test" + str(test_num) + "--")

    X, Theta = Xs[i], Thetas[i] # (898, 100), (898, 1)
    X_train, Theta_train, X_test, Theta_test = DivideIntoTrainAndTest(X, Theta) # (718, 100), (718, 1), (898, 100), (898, 1)
    W = Analysis(X_train, Theta_train) # (100, 1)

    Theta_pred = np.dot(X_test, W) # (898, 1)

    Visualize(Theta_test, Theta_pred, test_num)
    Calc_R2andRMSE(Theta_test, Theta_pred, T)
#####################################################################
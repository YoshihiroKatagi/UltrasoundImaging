# LinearRegressionForEachData Ver2
# difference:
# Ver1: Wが間違っている(179, 100) 時系列は関係ないはず
# Ver2: Wを修正(1, 100) そのためにAnalysis()を修正
# Ver2: 訓練データを含む全データでテストを行うX_test = X
# Ver2: 12パターンでループ
# Ver2: 特徴点の動作をチェックする関数作成 CheckFeature()
# Ver2: 特徴点にハイパスフィルタをかける HighpassFilter()
# Ver2: ハイパスフィルタチェック用のグラフ XYgraph()


import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import cv2
from scipy import signal

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
  plt.scatter(y[718:], y_pred[718:])
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
  R2 = _R2(theta[718:], theta_pred[718:])
  print("RMSE = " + str(RMSE))
  print("相関係数: " + str(R2) + "\n")

  with open(result_path + "/R2andRMSE.csv", "a") as f:
    writer = csv.writer(f)
    writer.writerow([RMSE, R2])
#####################################################################

########################  Check Feature  ###########################
def CheckFeature(x, theta): # (898, 100), (898, 1)
  x = x.reshape([x.shape[0], x.shape[1]//2, 2]) # (898, 50, 2)

  height = 480
  width = 640
  mask = np.zeros((height, width, 3), np.uint8)
  # for save
  fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  save = cv2.VideoWriter("test.mp4", fmt, 30, (width, height))

  x_pre = x[0]  # (50, 2)
  for t in range(x.shape[0]):
    if t+1 == x.shape[0]:
      break
    img = np.zeros((height, width, 3), np.uint8)
    x_now = x[t+1]  # (50, 2)
    theta_now = round(theta[t, 0], 2)

    for i in range(x.shape[1]):
      mask = cv2.line(mask, (int(x_pre[i][0]), int(x_pre[i][1])), (int(x_now[i][0]), int(x_now[i][1])), [128, 128, 128], 1)
      img = cv2.circle(img, (int(x_now[i][0]), int(x_now[i][1])), 5, [0, 0, 200], -1)

    # テキストデータ描画
    angle_data = "Wrist Angle: " + str(theta_now)
    org = (5, 400)
    cv2.putText(img, angle_data, org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.0, color=(255, 255, 255))

    image = cv2.add(img, mask)
    # cv2.imshow("mask", image)
    save.write(image)

    x_pre = x_now
  
  cv2.destroyAllWindows()
#####################################################################

########################  HighpassFilter  ###########################
# samplerate = 25600 # 波形のサンプリングレート
samplerate = 30
# x = np.arange(0, 12800) / samplerate
# data = np.random.normal(loc=0, scale=1, size=len(x)) # (12800,)

# fp = 3000          # 通過域端周波数[Hz]
fp = 0.04
# fs = 1500          # 阻止域端周波数[Hz]
fs = 0.02
gpass = 3            # 通過域端最大損失[dB]
gstop = 40           # 阻止域端最大損失[dB]

def HighpassFilter(data, samplerate, fp, fs, gpass, gstop):
  fn = samplerate / 2                           # ナイキスト周波数
  wp = fp / fn                                  # ナイキスト周波数で通過域端周波数を正規化
  ws = fs / fn                                  # ナイキスト周波数で阻止域端周波数を正規化
  N, Wn = signal.buttord(wp, ws, gpass, gstop)  # オーダーとバターワースの正規化周波数を計算
  b, a = signal.butter(N, Wn, "high")           # フィルタ伝達関数の分子と分母を計算

  data = data.reshape([data.shape[0], data.shape[1]//2, 2]) # (898, 50, 2)

  all_data = np.empty([data.shape[0], 0])

  for i in range(data.shape[1]):
    x = data[:, i, 0].reshape([data.shape[0],]) # (898,)
    y = data[:, i, 1].reshape([data.shape[0],]) # (898,)
    x = signal.filtfilt(b, a, x)                  # 信号に対してフィルタをかける
    y = signal.filtfilt(b, a, y)

    x, y = x.reshape([x.shape[0], 1]), y.reshape([y.shape[0], 1])

    filtered_data = np.append(x, y, axis=1) # (898, 2)
    all_data = np.append(all_data, filtered_data, axis=1)

  return all_data # (898, 100)
#####################################################################

###########################  XY Graph  ##############################
def XYgraph(X, iter, flag):
  X = X.reshape([X.shape[0], X.shape[1]//2, 2]) # (898, 50, 2)
  t = np.arange(X.shape[0])
  for i in iter:
    x = X[:, i, 0]
    y = X[:, i, 1]

    fig1 = plt.figure()
    plt.title("x-t graph")
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.plot(t, x, color="cornflowerblue", linewidth=2, label = "Feature No." + str(i+1) + " of x")
    if flag == "filtered":
      plt.ylim(-30, 30) # ハイパスフィルタ用
    else:
      plt.ylim(0, 640)
    plt.xlim(0, x.shape[0])
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True)
    if flag == "filtered":
      fig1.savefig("Feature_No." + str(i+1) + "_x-t" + "_filtered" + ".png")
    else:
      fig1.savefig("Feature_No." + str(i+1) + "_x-t" + ".png")

    fig2 = plt.figure()
    plt.title("y-t graph")
    plt.xlabel("Time")
    plt.ylabel("y")
    plt.plot(t, y, color="cornflowerblue", linewidth=2, label = "Feature No." + str(i+1) + " of y")
    if flag == "filtered":
      plt.ylim(-30, 30) # ハイパスフィルタ用
    else:
      plt.ylim(0, 640)
    plt.xlim(0, y.shape[0])
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True)
    if flag == "filtered":
      fig2.savefig("Feature_No." + str(i+1) + "_y-t" + "_filtered" + ".png")
    else:
      fig2.savefig("Feature_No." + str(i+1) + "_y-t" + ".png")

    # plt.show()
    plt.close()
#####################################################################

#############################  Main  ################################
for p in range(patern_num):
  patern = p + 1
  result_path = MakeResultsPath(patern)
  print("-----Patern" + str(patern) + "-----\n")

  Xs, Thetas = ReadData() # (10, 898, 100), (10, 898, 1)
  T = Xs.shape[1]
  for i in range(10):
    test_num = i + 1
    print("--test" + str(test_num) + "--")

    X, Theta = Xs[i], Thetas[i] # (898, 100), (898, 1)

    # グラフ化
    iter = [0, 9, 29]
    f = "nonfiltered"
    XYgraph(X, iter, f)
    
    # ハイパスフィルタ
    X = HighpassFilter(X, samplerate, fp, fs, gpass, gstop)

    # グラフ化
    f = "filtered"
    XYgraph(X, iter, f)
    # exit()

    # # 特徴点及び関節角度の描画
    # if patern == 1 and i == 4:
    #   CheckFeature(X, Theta)
    #   exit()

    X_train, Theta_train, X_test, Theta_test = DivideIntoTrainAndTest(X, Theta) # (718, 100), (718, 1), (898, 100), (898, 1)
    W = Analysis(X_train, Theta_train) # (100, 1)

    Theta_pred = np.dot(X_test, W) # (898, 1)

    Visualize(Theta_test, Theta_pred, test_num)
    Calc_R2andRMSE(Theta_test, Theta_pred, T)
    # exit()
  exit()
#####################################################################
# difference:
# Ver3: position_allを画像ごとに保存
# Ver4: position_allをまとめてPosition_Allとして保存
# Ver4: + 最初のフレームで特徴点を除外するときのエラーを解決
# Ver4: + いくつか確認のための関数追加、この中でパラメータチェックできるようにした


import cv2
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pyrsistent import v

### ※Execute directly under this file

########################### ファイル ############################
# 該当フォルダの日付（計測日と異なる場合は手入力）
target_date = "2022-05-30"
# target_date = datetime.now().strftime("%Y-%m-%d")

# 使用する超音波画像のファイル名一覧をリストで取得
image_path = "./dataset/" + target_date + "/ultrasoundImage/before"
ImageData = os.listdir(image_path)

# 該当ファイルの時刻を手入力（テスト用）
# ImageData =["12-31-26"]
# ImageData =["13-38-15", "14-24-32", "15-26-11", "16-05-07", "16-31-03", "17-11-41"]

# ゴニオメータの計測データのファイルパス
gonio_path = "./dataset/" + target_date + "/forMachineLearning/"
################################################################

########################  Read Gonio  ##########################
def ReadGonio():
  gonio_data_path = gonio_path + "gonioData.npy"
  gonio_data = np.load(gonio_data_path)

  return gonio_data
################################################################

#######################  Make Parameters  ######################
def MakeParameters(n, a, d, r):
  P = []
  for i in range(n):
    p = a + i * d
    p = round(p, r)
    P.append(p)
  
  return P
################################################################

#########################  Make Figure  ########################
def MakeFigure(c):
  x = c[0, :, 0]
  y = c[0, :, 1]
  fig = plt.figure()
  plt.title("Feature Points")
  plt.xlabel("x")
  plt.ylabel("y")
  plt.scatter(x, y)
  plt.grid(True)
  plt.show()
################################################################

#####################  Check PositionAll  ######################
# 特徴点の除外が正しくできているかの確認
# （position_all = np.delete(position_all, v - i, 1)　の部分の確認）
def CheckPositionAll(p_a, n):
  x = p_a[:, n, 0]
  y = p_a[:, n, 1]
  fig = plt.figure()
  plt.title("Feature Points")
  plt.xlabel("x")
  plt.ylabel("y")
  plt.plot(x, y, color="cornflowerblue", linewidth=2)
  plt.grid(True)
  plt.show()
################################################################

########################  Optical Flow  ########################
def OpticalFlow(t_path, i_s_path, p):
  cap = cv2.VideoCapture(t_path)

  # Shi-Tomasi法のパラメータ（コーナー：物体の角を特徴点として検出）
  ft_params = dict(maxCorners=500,       # 特徴点の最大数
                  qualityLevel=0.1,    # 特徴点を選択するしきい値で、高いほど特徴点は厳選されて減る。
                  minDistance=3,       # 特徴点間の最小距離
                  blockSize=15)         # 特徴点の計算に使うブロック（周辺領域）サイズ

  # Lucal-Kanade法のパラメータ（追跡用）
  lk_params = dict(winSize=(80,80),     # オプティカルフローの推定の計算に使う周辺領域サイズ
                  maxLevel=4,          # ピラミッド数
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))       # 探索アルゴリズムの終了条件

  # #properties
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  size = (width, height)  # (640, 480)
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 900
  frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) # 30

  # # for save
  # fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  # save = cv2.VideoWriter(i_s_path, fmt, frame_rate, size)

  # 最初のフレームを取得してグレースケール変換
  ret, frame = cap.read()
  frame_pre = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # 見切れ対策（最初の特徴点をトリミングした範囲内から抽出）(w:640, h:480)
  trim_w = 100
  trim_h = 75
  frame_pre_first = frame_pre[trim_h : height - trim_h, trim_w : width - trim_w]

  # Shi-Tomasi法で特徴点の検出
  feature_pre = cv2.goodFeaturesToTrack(frame_pre_first, mask=None, **ft_params)

  for v in feature_pre:
    v[0][0] += trim_w
    v[0][1] += trim_h

  # mask用の配列を生成
  mask = np.zeros_like(frame)

  j = 0
  # 動画終了まで繰り返し
  while(cap.isOpened()):
    if j >= frame_count - 3:
      break
    
    # 次のフレームを取得し、グレースケールに変換
    ret, frame = cap.read()
    if ret == False:
      break
    frame_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Lucas-Kanade法でフレーム間の特徴点のオプティカルフローを計算
    feature_now, status, err = cv2.calcOpticalFlowPyrLK(frame_pre, frame_now, feature_pre, None, **lk_params)

    # オプティカルフローを検出した特徴点を取得
    good1 = feature_pre[status == 1] # 1フレーム目
    good2 = feature_now[status == 1] # 2フレーム目

    # 座標を保存する配列を初期化、初期位置を保存
    if j == 0:
      position_all = np.empty([0, good1.shape[0], 2])
      position_t = good1.reshape([1, good1.shape[0], 2])
      position_all = np.append(position_all, position_t, axis=0)

      first_num = good1.shape[0]
      print("first: " + str(first_num))

    # statusが0となるインデックスを取得
    vanish = np.where(status == 0)[0]
    # if len(vanish) != 0:
    #   print("j: " + str(j))
    #   print(vanish)

    # position_allからstatus=0の要素を削除
    for i, v in enumerate(vanish):
      # 最初のフレーム間で特徴点が消えている場合は何もしない
      if j == 0:
        break
      # print("i, v: " + str(i) + ", " + str(v))
      position_all = np.delete(position_all, v - i, 1)
    
    # 各時刻における座標を保存
    position_t = good2.reshape([1, good2.shape[0], 2])
    position_all = np.append(position_all, position_t, axis=0)

    # 特徴点とオプティカルフローをフレーム・マスクに描画
    for i, (pt1, pt2) in enumerate(zip(good1, good2)):
      x1, y1 = pt1.ravel() # 1フレーム目の特徴点座標
      x2, y2 = pt2.ravel() # 2フレーム目の特徴点座標

      # 軌跡を描画（過去の軌跡も残すためにmaskに描く）
      mask = cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), [128, 128, 128], 1)

      # 現フレームにオプティカルフローを描画
      frame = cv2.circle(frame, (int(x2), int(y2)), 5, [0, 0, 200], -1)
    
    # フレームとマスクの論理積（合成）
    img = cv2.add(frame, mask)

    # ウィンドウに表示
    cv2.imshow('mask', img)
    
    # # save per frame
    # save.write(img)

    # 次のフレーム、ポイントの準備
    frame_pre = frame_now.copy() # 次のフレームを最初のフレームに設定
    feature_pre = good2.reshape(-1, 1, 2) # 次の点を最初の点に設定

    # qキーが押されたら途中終了
    if cv2.waitKey(30) & 0xFF == ord('p'):
      break
    
    j += 1

  last_num = good2.shape[0]
  print("last: " + str(last_num))
  proportion = last_num/first_num * 100
  print("proportion: {:.1f}/n".format(proportion))

  # for i in range(10):
  #   n = i * 10
  #   print("n: " + str(n))
  #   CheckPositionAll(position_all, n)

  position_all = np.delete(position_all, np.s_[100:], 1) # (898, 100, 2)
  # MakeFigure(position_all)
  position_all = position_all.reshape([1, position_all.shape[0], position_all.shape[1] * 2]) # (1, 898, 200)

  # 終了処理
  cv2.destroyAllWindows()
  cap.release()
  # save.release()

  return position_all
################################################################

###################  Save Proccessed Image  ####################
def SaveProccessedImage(t_path, i_s_path, points, theta):
  points = points[:, :, :100].reshape([points.shape[1], 50, 2]) # (898, 50, 2)
  cap = cv2.VideoCapture(t_path)

  # #properties
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  size = (width, height)  # (640, 480)
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 900
  frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) # 30

  # for save
  fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  save = cv2.VideoWriter(i_s_path, fmt, frame_rate, size)

  # 最初のフレームを取得
  ret, frame = cap.read()
  # frame_pre = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # mask用の配列を生成
  mask = np.zeros_like(frame)

  # 最初の特徴点の座標を取得
  points_pre = points[0] # (50, 2)
  # 動画終了まで繰り返し
  for t in range(points.shape[0]):
    if t+1 == points.shape[0]:
      break
    
    # 現在のフレームを取得
    ret, frame = cap.read()
    if ret == False:
      break
    # frame_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 現在の特徴点の座標を取得
    points_now = points[t+1]

    # 現在の関節角度を取得
    theta_now = round(theta[t, 0], 2)

    # オプティカルフローと現在の特徴点をmask, frameに描画
    for p_pre, p_now in zip(points_pre, points_now):
      x1, y1 = p_pre[0], p_pre[1]
      x2, y2 = p_now[0], p_now[1]

      mask = cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), [128, 128, 128], 1)
      frame = cv2.circle(frame, (int(x2), int(y2)), 5, [0, 0, 200], -1)

      # 関節角度情報を描画
      angle = "Wrist Angle: " + str(theta_now)
      org = (20, 460) # 挿入する座標
      cv2.putText(frame, angle, org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 255, 255))

    # frameとmaskの合成
    img = cv2.add(frame, mask)

    # ウィンドウに表示
    cv2.imshow("mask", img)

    # フレームごとに保存
    save.write(img)

    # pointsの更新
    points_pre = points_now

    # qキーが押されたら途中終了
    if cv2.waitKey(30) & 0xFF == ord('p'):
      break

  # 終了処理
  cv2.destroyAllWindows()
  cap.release()
  save.release()
################################################################

###########################  Main  #############################
# 本番用（基本こっち）
Parameters = [0]
# パラメータ調整用　(MakeParameters()を調整する)
# Parameters = MakeParameters(1, 4, 1, 1)

Gonio_data = ReadGonio() # (120, 898, 1)

# 画像ごとにOpticalFlow()を実行
for i, id in enumerate(ImageData):
  target_image = os.path.splitext(id)[0]
  Theta = Gonio_data[i]

  target_directry = "./dataset/" + target_date + "/ultrasoundImage"
  target_path = target_directry + "/before/" + target_image + ".mp4"
  image_save_path = target_directry + "/after/" + target_image + ".mp4"
  forML_folder = "./dataset/" + target_date + "/forMachineLearning/"
  if not os.path.exists(forML_folder):
    os.makedirs(forML_folder)
  position_save_path = forML_folder + "FeaturePointsData"

  print("-----Target Image: " + "No." + str(i) + " " + str(id) + "-----")

  for p in Parameters:
    print("---param: " + str(p) + "---")
    Position_by_Image = OpticalFlow(target_path, image_save_path, p) # (1, 898, 200)

    # 50個の特徴点とその時の関節角度をUS画像に描画
    SaveProccessedImage(target_path, image_save_path, Position_by_Image, Theta)

    if i == 0:
      Position_All = np.empty([0, Position_by_Image.shape[1], Position_by_Image.shape[2]])
    Position_All = np.append(Position_All, Position_by_Image, axis=0) # (120, 898, 200)

print("Position_All: " + str(Position_All.shape))
# 全画像における特徴点の座標を保存
# np.save(position_save_path, Position_All)

################################################################
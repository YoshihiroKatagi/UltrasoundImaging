import cv2
import numpy as np
from datetime import datetime

############### ファイル #################
#該当ファイルの日時
target = "20211217_112231"
target_path = "Img/BeforeProcessing/" + target + ".mp4"

date = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = "Img/AfterProcessing/" + date + ".mp4"

#########################################

############## Optical Flow #############
cap = cv2.VideoCapture(target_path)

# Shi-Tomasi法のパラメータ（コーナー：物体の角を特徴点として検出）
ft_params = dict(maxCorners=30,       # 特徴点の最大数
                 qualityLevel=0.2,    # 特徴点を選択するしきい値で、高いほど特徴点は厳選されて減る。
                 minDistance=15,       # 特徴点間の最小距離
                 blockSize=15)         # 特徴点の計算に使うブロック（周辺領域）サイズ

# Lucal-Kanade法のパラメータ（追跡用）
lk_params = dict(winSize=(60,60),     # オプティカルフローの推定の計算に使う周辺領域サイズ
                 maxLevel=4,          # ピラミッド数
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))       # 探索アルゴリズムの終了条件

# #properties
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
size = (width, height)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# # for save
# fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# save = cv2.VideoWriter(save_path, fmt, frame_rate, size)

#教師データ
# ml_data = np.zeros(2)

# 最初のフレームを取得してグレースケール変換
ret, frame = cap.read()
frame_pre = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 見切れ対策（最初の特徴点をトリミングした範囲内から抽出）
trim_w = 80
trim_h = 100
frame_pre_first = frame_pre[trim_h : height - trim_h, trim_w : width - trim_w]

# Shi-Tomasi法で特徴点の検出
# feature_pre = cv2.goodFeaturesToTrack(frame_pre, mask=None, **ft_params)
feature_pre = cv2.goodFeaturesToTrack(frame_pre_first, mask=None, **ft_params)

for v in feature_pre:
  v[0][0] += trim_w
  v[0][1] += trim_h

# mask用の配列を生成
mask = np.zeros_like(frame)

# 全区間の全特徴量のベクトル
vector_all = np.empty(0)

# 動画終了まで繰り返し
while(cap.isOpened()):
  
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

  feature_num = good1.shape[0]

  # ある時刻のベクトルを導出
  if good1.shape == good2.shape:
    vector_t = good2 - good1
    vector_all = np.append(vector_all, vector_t)


  # 特徴点とオプティカルフローをフレーム・マスクに描画
  for i, (pt1, pt2) in enumerate(zip(good1, good2)):
    x1, y1 = pt1.ravel() # 1フレーム目の特徴点座標
    x2, y2 = pt2.ravel() # 2フレーム目の特徴点座標

    # 軌跡を描画（過去の軌跡も残すためにmaskに描く）
    mask = cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), [0, 0, 200], 1)

    # 現フレームにオプティカルフローを描画
    frame = cv2.circle(frame, (int(x2), int(y2)), 5, [0, 0, 200], -1)
  
  # フレームとマスクの論理積（合成）
  img = cv2.add(frame, mask)

  # ウィンドウに表示
  cv2.imshow('mask', img)
  
  # save per frame
  # save.write(img)

  # 次のフレーム、ポイントの準備
  frame_pre = frame_now.copy() # 次のフレームを最初のフレームに設定
  feature_pre = good2.reshape(-1, 1, 2) # 次の点を最初の点に設定

  # qキーが押されたら途中終了
  if cv2.waitKey(30) & 0xFF == ord('q'):
    break

# 終了処理
cv2.destroyAllWindows()
cap.release()
# save.release()

vector_all = vector_all.reshape(-1, feature_num, 2)
print(vector_all)
print(vector_all.shape)
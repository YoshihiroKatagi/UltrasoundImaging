import numpy as np
import matplotlib.pyplot as plt
import cv2
# from google.colab.patches import cv2_imshow

# file_path = "D0002060316_00000_V_000.mp4"
file_path = "D0002060316_00000.mp4"

video = cv2.VideoCapture(file_path)


start_frame = 150
end_frame = 210
interval_frames = 5
i = start_frame + interval_frames

video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
ret, prev_frame = video.read()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)

feature_params = {
  "maxCorners": 200,
  "qualityLevel": 0.2,
  "minDistance": 12,
  "blockSize": 12
}
p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

for p in p0:
  x,y = p.ravel()
  cv2.circle(prev_frame, (x,y), 5, (0, 255, 255), -1)

cv2.imshow(prev_frame)


# OpticalFlowのパラメータ
lk_params = {
    "winSize": (15, 15),  # 特徴点の計算に使う周辺領域サイズ
    "maxLevel": 2,  # ピラミッド数 (デフォルト0で、2の場合は1/4の画像まで使われる)
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # 探索アルゴリズムの終了条件
}

# 可視化用
color = np.random.randint(0, 255, (200, 3))
mask = np.zeros_like(prev_frame)

for i in range(start_frame + interval_frames, end_frame + 1, interval_frames):
    # 次のフレームを取得してグレースケールにする
    video.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = video.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # OpticalFlowの計算
    p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
    
    # フレーム前後でトラックが成功した特徴点のみを
    identical_p1 = p1[status==1]
    identical_p0 = p0[status==1]
    
    # 可視化用
    for i, (p1, p0) in enumerate(zip(identical_p1, identical_p0)):
        p1_x, p1_y = p1.ravel()
        p0_x, p0_y = p0.ravel()
        mask = cv2.line(mask, (p1_x, p1_y), (p0_x, p0_y), color[i].tolist(), 2)
        frame = cv2.circle(frame, (p1_x, p1_y), 5, color[i].tolist(), -1)
    
    # 可視化用の線・円を重ねて表示
    image = cv2.add(frame, mask)
    cv2.imshow(image)

    # トラックが成功した特徴点のみを引き継ぐ
    prev_gray = frame_gray.copy()
    p0 = identical_p1.reshape(-1, 1, 2)
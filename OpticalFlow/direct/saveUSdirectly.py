import cv2
import os
from datetime import datetime

### ※Execute directly under this file
### データセットが意図しない場所にできる可能性あり

################### ファイル #####################
# 新規フォルダ作成
today = datetime.now().strftime("%Y-%m-%d")
save_folder = "dataset/" + today + "/ultrasoundImage"
if not os.path.exists(save_folder):
  os.makedirs(save_folder + "/before")
  os.makedirs(save_folder + "/after")
  os.makedirs("dataset/" + today + "/gonioMeter")

# 保存場所
date = datetime.now().strftime("%H-%M-%S")
save_path = save_folder + "/before/" + date + ".mp4"
# date = datetime.now().strftime("%Y%m%d_%H%M%S")
# save_path = "Img/BeforeProcessing/" + date + ".mp4"
##################################################

###############  超音波画像取得  ##################

# 録画時間[s]
time = 30

cap = cv2.VideoCapture(0)

#properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = fps * time

print(width, height, fps)
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
save = cv2.VideoWriter(save_path, fmt, fps, (width, height))


#画面出力
while True:
  ret, frame = cap.read()
  if ret == True:
    cv2.imshow("USimage", frame)

    k = cv2.waitKey(1)&0xff
    #「space」キーで録画開始
    if k == ord(' '):
      cv2.destroyWindow("USimage")
      break
    

#録画
for i in range(frame_count):
  ret, frame = cap.read()
  if ret == True:

    # 加工処理
    frame = frame[30:420,155:530]

    frame = cv2.resize(frame, (width, height))
    save.write(frame)
    cv2.imshow("Recording", frame)

    k = cv2.waitKey(1)&0xff
    #「q」が押されたら終了する
    if k == ord('q'):
      break

save.release()
cap.release()
cv2.destroyAllWindows()
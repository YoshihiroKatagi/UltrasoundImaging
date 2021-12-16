import cv2
from datetime import datetime

date = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = "Img/" + date + ".mp4"
time = 10

cap = cv2.VideoCapture(0)

#properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = fps * time

print(height, width, fps)
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
save = cv2.VideoWriter(save_path, fmt, fps, (width, height))


#画面出力
while True:
  ret, frame = cap.read()
  cv2.imshow("USimage", frame)

  k = cv2.waitKey(1)&0xff
  #「r」キーで録画開始
  if k == ord('r'):
    cv2.destroyWindow("USimage")
    break


#録画
for i in range(frame_count):
  ret, frame = cap.read()
  save.write(frame)
  cv2.imshow("Recording", frame)

  k = cv2.waitKey(1)&0xff
  #「q」が押されたら終了する
  if k == ord('q'):
    break

save.release()
cap.release()
cv2.destroyAllWindows()




# while True:
#   ret, frame = cap.read()
#   save.write(frame)
#   cv2.imshow("USimage", frame)

#   k = cv2.waitKey(1)&0xff
#   #「q」が押されたら終了する
#   if k == ord('q'):
#     break








# cap = cv2.VideoCapture(0)

# while True:
#   ret, frame = cap.read()
#   cv2.imshow("camera", frame)

#   k = cv2.waitKey(1)&0xff
#   if k == ord('p'):
#     # 「p」キーで画面保存
#     date = datetime.now().strftime("%Y%m%d_%H%M%S")
#     path = "./Img/" + date + ".png"
#     cv2.imwrite(path, frame) #ファイル保存

#     cv2.imshow(path, frame) #キャプチャした画像を表示
#   elif k == ord('q'):
#     # 「q」キーが押されたら終了する
#     break

# # キャプチャをリリースして、ウィンドウをすべて閉じる
# cap.release()
# cv2.destroyAllWindows()

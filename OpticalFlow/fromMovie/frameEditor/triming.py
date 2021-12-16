#動画を１フレームずつ読み込み切り取り処理、その後再度動画化

import cv2

target_path = "../movie/source/ultrasound.mp4"
save_path = "../movie/2021_10_29/trim.mp4"

video = cv2.VideoCapture(target_path)

#properties
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
size = (width, height)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = int(video.get(cv2.CAP_PROP_FPS))

print(height, width, size, frame_count, frame_rate)

fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
save = cv2.VideoWriter(save_path, fmt, frame_rate, size)

for i in range(frame_count):
  ret, frame = video.read()
  if ret==True:
    ### 処理 ###
    frame = frame[30:420,155:530]
    ############
    frame = cv2.resize(frame,(640,480))
    save.write(frame)
    cv2.imshow("img", frame)
    if cv2.waitKey(30)==27:
      break

save.release()
# video.release()
cv2.destroyAllWindows()
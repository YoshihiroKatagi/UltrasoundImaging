import cv2

COUNT = 48
target_path = "movie/2021_11_25/trim_threshold.mp4"
save_path = 'movie/2021_12_17/test2.mp4'

criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 20, 0.03)

lk_params = dict(winSize=(200,200), maxLevel=4, criteria=criteria)
cap = cv2.VideoCapture(target_path)
#properties
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
size = (width, height)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
print(height, width, size, frame_count, frame_rate)

fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
save = cv2.VideoWriter(save_path, fmt, frame_rate, size)

ret, frame = cap.read()
# print(ret, type(frame))
frame_pre = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
while True:
  ret, frame = cap.read()
  if ret == False:
    break
  frame_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  feature_pre = cv2.goodFeaturesToTrack(frame_pre, COUNT, 0.001, 5)
  if feature_pre is None:
    continue
  feature_now, status, err = cv2.calcOpticalFlowPyrLK(frame_pre, frame_now, feature_pre, None, **lk_params)
  for i in range(len(feature_now)):
    pre_x = int(feature_pre[i][0][0])
    pre_y = int(feature_pre[i][0][1])
    now_x = int(feature_now[i][0][0])
    now_y = int(feature_now[i][0][1])
    cv2.line(frame, (pre_x, pre_y), (now_x, now_y), (255,0,0), 3)
  cv2.imshow("img", frame)
  save.write(frame)
  frame_pre = frame_now.copy()
  key = cv2.waitKey(30)
  if key == 27:
    break

save.release()
cv2.destroyAllWindows()
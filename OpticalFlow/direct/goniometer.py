import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


start_row = 6
target_column = 2

target_date = "2022-05-30"
path = "C:/Users/katagi/Desktop/Workspace/Research/UltrasoundImaging/OpticalFlow/direct/dataset/" + target_date + "/gonioMeter/"

gonio_data = ["12-31-26", "12-34-25", "12-39-14", "12-40-51", "12-42-42"]

def read_csv():
  thetas = list()

  for i in range(len(gonio_data)):
    gonio_path = path + gonio_data[i] + ".csv"

    with open(gonio_path) as f:
      reader = csv.reader(f)
      theta = list()
      for t, row in enumerate(reader):
        if t < start_row:
          continue

        theta.append(row[target_column])
      thetas.append(theta)

  thetas = np.array(thetas).astype(float)[:, :, np.newaxis]
  return thetas



t0 = 0    # 初期時間[s]
tf = 30   # 終了時間[s]
dt = 0.1  # 時間刻み[s]
t = np.arange(t0, tf, dt)

sample_num = round(tf / dt)
resample_num = sample_num * 3 - 2   # 898
t_resample = np.linspace(t0, tf - dt, resample_num)

def resample_angle_data(thetas):
  resampled_thetas = list()

  for theta in thetas:
    theta = np.squeeze(theta)
    f = interpolate.interp1d(t, theta, kind="cubic")
    theta_resample = f(t_resample)
    resampled_thetas.append(theta_resample)

    # # グラフ化
    # plt.scatter(t, theta, label="observed")
    # plt.plot(t_resample, theta_resample, c="red", label="fitted")
    # plt.grid()
    # plt.legend()
    # plt.show()
  
  resampled_thetas = np.array(resampled_thetas).astype(float)[:, :, np.newaxis]
  return resampled_thetas


thetas = read_csv()
print("Shape of Thetas: " + str(thetas.shape))
resampled_thetas = resample_angle_data(thetas)
print("Shape of Resampled Thetas: " + str(resampled_thetas.shape))
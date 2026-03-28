# secondary optional preprocessing step, separating activities

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# combine accelerometer and gyroscope data into one IMU file

root = Path('hss_final_proj/3_25_raw_data')

split_folder = "3_1_2"
new_folder = "3_2_2"
filepath_a = root / split_folder / "Accelerometer.csv"
filepath_g = root / split_folder / "Gyroscope.csv" 


acc = pd.read_csv(filepath_a)
gyro = pd.read_csv(filepath_g)

# plot accelerometer data for easiest visual (x/y/z)
acc_cols = ['x', 'y', 'z']

fig, ax = plt.subplots(figsize=(10, 5))

for col in acc_cols:
    ax.plot(imu[col].values, lw=0.8, label=col)  # no time, just index
ax.set_xlabel('Sample Index')
ax.set_ylabel('Acceleration')
ax.set_title('Click START then END for each of the 10 gestures (20 clicks total)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

# get points to segment each gesture from user input
n = 1 # number of split points
plt.show()
pts = plt.ginput(n=n, timeout=-1) #selecting n number of points, -1 never times out
plt.close()

trials_path = Path("final_proj/trials")
print(pts)
print(pts[0])
index = int(pts[0][0]) # rounding to integer of nearest index

segment1a = acc.iloc[:index]
segment2a = acc.iloc[index:]

segment1g = gyro.iloc[:index].reset_index(drop=True)
segment2g = gyro.iloc[index:].reset_index(drop=True)

seg_path1 = root / split_folder
seg_path2 = root / new_folder

# save activity segments as individual files, easier to parse through later
segment1a.to_csv(seg_path1 / "Accelerometer_trim.csv", index=False) # manually rename trimmed files in folder
segment2a.to_csv(seg_path2 / "Accelerometer.csv", index=False)

segment1g.to_csv(seg_path1 / "Gyroscope_trim.csv", index=False)
segment2g.to_csv(seg_path2 / "Gyroscope.csv", index=False)


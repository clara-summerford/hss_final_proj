# secondary optional preprocessing step, adjusting UNIX time 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


root = Path('hss_final_proj/3_25_raw_data')

trial_folder = "1_4_1"
filepath_a = root / trial_folder / "Accelerometer.csv"
filepath_g = root / trial_folder / "Gyroscope.csv" 

acc1 = pd.read_csv(filepath_a)
gyro1 = pd.read_csv(filepath_g)

acc2 = pd.read_csv(root / "1_4_2" / "Accelerometer.csv")
acc3 = pd.read_csv(root / "1_4_3" / "Accelerometer.csv")

# plot accelerometer data for easiest visual (x/y/z)
acc_cols = ['x', 'y', 'z']

fig, axes = plt.subplots(nrows = 3, ncols=1, figsize=(10, 5))

ax_flat = axes.flatten()
acc_list = [acc1, acc2, acc3]

for index, a in enumerate(ax_flat):
        
    for col in acc_cols:
        a.plot(acc_list[index][col].values, lw=0.8, label=col)  # no time, just index

        a.set_xlabel('Sample Index')
        a.set_ylabel('Acceleration')
        a.set_title('Click START then END for each of the 10 gestures (20 clicks total)')
        a.legend()
        a.grid(True, alpha=0.3)
        plt.tight_layout()

# get points to segment each gesture from user input
n = 4 # number of points to match up, allows 2 "dummy" clicks to zoom into graph
pts = plt.ginput(n=n, timeout=-1) #selecting n number of points, -1 never times out
# plt.show()
plt.close()

# trials_path = Path("final_proj/trials")
index1 = int(pts[2][0]) # rounding to integer of nearest index
index2 = int(pts[3][0])


acc1_time = acc1.iloc[index1, 0]
print(f'Acc1 time: {acc1_time}')
acc2_time = acc2.iloc[index2, 0]
print(f'Acc2 time: {acc2_time}')

gyro1_time = gyro1.iloc[index1, 0]

# calculate UNIX time offset
time_diff = acc2_time - acc1_time
print(time_diff)
acc1.iloc[:, 0] += time_diff
gyro1.iloc[:, 0] += time_diff

# save new CSV with fixed time
acc1.to_csv(root / trial_folder / "Accelerometer_fixed.csv", index=False)
gyro1.to_csv(root / trial_folder / "Gyroscope_fixed.csv", index=False)



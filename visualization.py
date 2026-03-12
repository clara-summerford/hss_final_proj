# processing all activities and feature extraction

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd



activity1_path = Path("final_proj/trials/P2/A1/2_1_01.csv")
activity2_path = Path("final_proj/trials/P1/A2/1_2_01.csv")

        
imu = pd.read_csv(activity1_path)
imu2 = pd.read_csv(activity2_path)

# plot wrist accelerometer data for easiest visual (imu3_accel_x/y/z)
acc_cols = ['imu3_accel_x', 'imu3_accel_y', 'imu3_accel_z']
gyro_cols = ['imu3_gyro_x', 'imu3_gyro_y', 'imu3_gyro_z']

# fig, (ax1, ax2, ax3, ax4) = plt.subplot(nrows=4, ncols=2, figsize=(10, 5))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

for a_col, g_col in zip(acc_cols, gyro_cols):
    ax1.plot(imu[a_col].values, lw=0.8, label=a_col)
    ax1.legend() 
    ax1.set_title("Frisbee Accelerometer")
    ax2.plot(imu[g_col].values, lw=0.8, label=g_col) 
    ax2.set_title("Frisbee Gyroscope")
    ax2.legend() 

    ax3.plot(imu2[a_col].values, lw=0.8, label=a_col)
    ax3.legend() 
    ax3.set_title("Pickleball Accelerometer")
    ax4.plot(imu2[g_col].values, lw=0.8, label=g_col) 
    ax4.set_title("Pickleball Gyroscope")
    ax4.legend()


plt.show()


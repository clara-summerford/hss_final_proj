# processing all activities and feature extraction

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


# trials_path = Path("final_proj/trials")


activity_path = Path("final_proj/trials/P2/A1")

# for participant_path in sorted(trials_path.iterdir()): # iterate through all folders in given data path
#     if not participant_path.is_dir(): # skipping hidden files
#         continue 

#     for activity_path in sorted(participant_path.iterdir()):
#         if not activity_path.is_dir():
#             continue
        
for imu_path in sorted(activity_path.iterdir()):
    if not imu_path.is_file():
        continue
    imu = pd.read_csv(imu_path)

    # plot wrist accelerometer data for easiest visual (imu3_accel_x/y/z)
    acc_cols = ['imu3_accel_x', 'imu3_accel_y', 'imu3_accel_z']

    fig, ax = plt.subplots(figsize=(10, 5))

    for col in acc_cols:
        ax.plot(imu[col].values, lw=0.8, label=col) 

    plt.show()


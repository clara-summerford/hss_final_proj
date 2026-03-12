# pre-processing IMU samples for each activity sample

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os

# combine accelerometer and gyroscope data into one IMU file

root = Path('final_proj/raw_data')

def merge_acc_gryo():
    for trial_path in sorted(root.iterdir()):
        if not trial_path.is_dir(): # can change this to leave out previously processed trials later
            continue

        gyro = pd.read_csv(trial_path / 'Gyroscope.csv').sort_values('time').reset_index(drop=True)
        accel = pd.read_csv(trial_path / 'Accelerometer.csv').sort_values('time').reset_index(drop=True)

        merged = pd.merge_asof(
            accel, gyro,
            on='time',
            direction='nearest',
            tolerance=5*(10**6) # 5ms expressed in nanoseconds for UNIX epoch timestamp
        )   

        n_dropped = merged.isnull().any(axis=1).sum()
        if n_dropped:
            print(f"{n_dropped} rows dropped (no gyro match within 5ms)")

        merged = merged.dropna().reset_index(drop=True)
        merged.drop('seconds_elapsed_y', axis = 1, inplace=True) # removing second elapsed time col
    
        # Rename to standard column names
        merged.columns = ['time', 'seconds_elapsed', 'accel_x', 'accel_y', 'accel_z',
                                        'gyro_x',  'gyro_y',  'gyro_z']
        
        # p = trial_path.stem
        output_file = str(trial_path) + "_IMU.csv"
        output_path = trial_path / output_file

        merged.to_csv(output_file, index=False)
        print(f"Saved {len(merged)} rows → {output_path}")

# merge_acc_gryo() 
        
# combine all 3 IMU streams
def combine_IMU_streams():
    # for number participants
    for i in range(1,3): 

        # for number activities
        for j in range(1,4): 

            # for number IMU streams (this range never changes, will always be 3)
            # filenames = [str(i) + "_" + str(j) + "_" + str(k) + "_IMU.csv" for k in range(1,4)] 
            filenames = [f"{i}_{j}_{k}_IMU.csv" for k in range(1, 4)]

            # df1 = pd.read_csv(root/filenames[0]).sort_values('time').reset_index(drop=True)
            # df2 = pd.read_csv(root/filenames[1]).sort_values('time').reset_index(drop=True)
            # df3 = pd.read_csv(root/filenames[2]).sort_values('time').reset_index(drop=True)
            # dfs = [df1, df2, df3]
            dfs = [pd.read_csv(root / f).sort_values('time').reset_index(drop=True) for f in filenames]

            # trimming start/end values that are not included in all IMU streams
            latest_start = max(df['time'].iloc[0] for df in dfs)
            earliest_end = min(df['time'].iloc[-1] for df in dfs)

            trimmed = []
            for df in dfs: # creates a nested list with 3 entries, each entry is a trimmed DF
                mask = (df['time'] >= latest_start) & (df['time'] <= earliest_end)
                trimmed.append(df[mask].sort_values('time').reset_index(drop=True))
                # print(len(trimmed[0]))
                # print(len(trimmed[1]))


            # store timestamp of the shortest of the trimmed dataframes
            ref_df = min(trimmed, key=len)
            ref_timestamps = ref_df['time'].values

            # snap the IMU values to the correct, trimmed reference timestamps
            ref_grid = pd.DataFrame({'time': ref_timestamps}) # initialize df with only timestamps

            aligned = []
            for id, df in enumerate(trimmed): 
                # creating a new dataframe (snapped) attaching individual IMU data to desired timeframe
                snapped = pd.merge_asof(
                    ref_grid, 
                    df.sort_values('time'), 
                    on='time', 
                    direction = 'nearest',
                    tolerance = 5*(10**6)
                    )
                
                n_dropped = snapped.isnull().any(axis=1).sum()
                if n_dropped > 0:
                    print(f"Warning! IMU {id+1}: {n_dropped} reference timestamps had no match "
                    f"within 5ms")

                snapped = snapped.dropna().reset_index(drop=True)

                # combine all IMU channels with corrected timeframes
                aligned.append(snapped)
                # print(f"Iteration {id} snapped size: {snapped.size}")
                # print(f"Snapped columns: {snapped.columns}")

            # equalize lengths of dataframes after dropping mismatched samples
            min_len = min(len(df) for df in aligned)
            aligned = [df.iloc[:min_len].reset_index(drop=True) for df in aligned]
            # print(f"aligned 0 columns: {aligned[0].columns}")
            
            # rename columns before merging
            for k in range(len(aligned)):
                aligned[k] = aligned[k].rename(columns={
                    'accel_x': f'imu{k+1}_accel_x',
                    'accel_y': f'imu{k+1}_accel_y',
                    'accel_z': f'imu{k+1}_accel_z',
                    'gyro_x':  f'imu{k+1}_gyro_x',
                    'gyro_y':  f'imu{k+1}_gyro_y',
                    'gyro_z':  f'imu{k+1}_gyro_z',
                })

            # combining all 3 aligned dataframes into one big dataframe for each participant and activity
            combined = aligned[0] # initializing with first aligned dataframe
            for t in range(1, len(aligned)): # starting with second dataframe
                combined = combined.merge(aligned[t].drop(columns=["time", "seconds_elapsed"]), left_index=True, right_index=True)
                
            output_file = f"{i}_{j}_combined_IMU.csv"
            output_path = root / "combined_IMU"

            combined.to_csv(output_path/output_file, index=False)
            print(f"Saved {len(combined)} rows → {output_path/output_file}")

# combine_IMU_streams()


# analyze data by activity and user 
def extract_windows():

    user = int(input("Enter user number: \nVictoria = 1 \nClara = 2\n").strip())
    sport = int(input("Enter sport: \nFrisbee = 1 \nPickle = 2\nThrowing = 3\n").strip())
    # user = 1
    # sport = 3

    sport_file = f"{user}_{sport}_combined_IMU.csv"
    filepath = root / 'combined_IMU' / sport_file

    imu = pd.read_csv(filepath)
    
    # plot wrist accelerometer data for easiest visual (imu3_accel_x/y/z)
    acc_cols = ['imu3_accel_x', 'imu3_accel_y', 'imu3_accel_z']

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
    pts = plt.ginput(n=20, timeout=-1) # need 2*10 points, -1 never times out
    plt.close()

    # trials_path = Path("final_proj/trials")
    # for i in range(10):
    #     start_in = int(pts[i*2][0]) # rounding to integer of nearest index
    #     end_in = int(pts[(i*2)+1][0])

    #     segment = imu.iloc[start_in:end_in].reset_index(drop=True)

    #     user_folder = f"P{user}"
    #     seg_folder = f"A{sport}"
    #     seg_path = trials_path/user_folder/seg_folder
    #     seg_name = f"{user}_{sport}_{i+1:02d}.csv"

    #     # save activity segments as individual files, easier to parse through later
    #     segment.to_csv(seg_path/seg_name, index=False)
    #     print(f"Saved {len(segment)} rows → {seg_path/seg_name}")

extract_windows()



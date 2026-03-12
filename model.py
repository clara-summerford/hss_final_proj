# processing all activities and feature extraction

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

trials_path = Path("final_proj/trials")

# create feature vector and labels by iterating through each activity iteration
X_vec = []
y_vec = []
participants = []

for participant_path in sorted(trials_path.iterdir()): # iterate through all folders in given data path
    if not participant_path.is_dir(): # skipping hidden files
        continue 

    for activity_path in sorted(participant_path.iterdir()):
        if not activity_path.is_dir():
            continue
        
        for imu_path in sorted(activity_path.iterdir()):
            if not imu_path.is_file():
                continue

            df = pd.read_csv(imu_path)

            # features: peak acceleration, peak angular velocity, axis energy ratios, signal energy (magnitude)
            features = {}
            
            imus = [1, 2, 3]
            sensors = ["accel","gyro"]
            axes = ["x","y","z"]

            for imu in imus:
                for sensor in sensors:

                    cols = [f"imu{imu}_{sensor}_{axis}" for axis in axes]

                    data = df[cols]

                    # peak values
                    features[f"imu{imu}_{sensor}_max"] = data.max().max()
                    features[f"imu{imu}_{sensor}_min"] = data.min().min()

                    # magnitude signal
                    mag = np.sqrt((data**2).sum(axis=1))

                    features[f"imu{imu}_{sensor}_mag_max"] = mag.max()
                    features[f"imu{imu}_{sensor}_mag_mean"] = mag.mean()
                    features[f"imu{imu}_{sensor}_mag_std"] = mag.std()

                    # axis ratios
                    if sensor == "accel":
                        ax = df[f"imu{imu}_accel_x"]
                        ay = df[f"imu{imu}_accel_y"]
                        az = df[f"imu{imu}_accel_z"]

                        features[f"imu{imu}_ay_az_ratio"] = ay.abs().mean() / (az.abs().mean() + 1e-6) # making sure you dom't divide by 0
                        features[f"imu{imu}_ax_az_ratio"] = ax.abs().mean() / (az.abs().mean() + 1e-6)
                        features[f"imu{imu}_ax_ay_ratio"] = ax.abs().mean() / (ay.abs().mean() + 1e-6)

            # generating labels and appending all features
            p = imu_path.stem
            y_vec.append(p.split('_')[1])
            participants.append(p.split('_')[0])
            X_vec.append(features)

print(len(y_vec))

X_vec = pd.DataFrame(X_vec)
# y_vec = pd.Series(y_vec)
y_vec = np.array(y_vec)
participants = np.array(participants)

print(X_vec.shape)
print(X_vec.columns)

# scale data using Z-score normalization
scaler = StandardScaler()
X_vec = scaler.fit_transform(X_vec)

# selecting classifiers
# clf = RandomForestClassifier()
clf = svm.SVC()

# --- LOPO model evaluation ---
LOPO_score = []
logo = LeaveOneGroupOut()
activity_list = ["Frisbee", "Pickleball", "Baseball"]

for train_index, test_index in logo.split(X_vec, y_vec, groups=participants):
    X_train, X_test = X_vec[train_index], X_vec[test_index]
    y_train, y_test = y_vec[train_index], y_vec[test_index]

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    LOPO_score.append(score)

    print(f'Score for individual LOPO iteration: {score}')

    # create a confusion matrix for each participant to identify discrepancies
    # y_pred = clf.predict(X_test)
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=activity_list)
    # plt.title("LOPO Confusion Matrix")
    # plt.show()
    
print(f'Average LOPO score: {np.mean(LOPO_score)}')

# --- k-fold evaluation ---
k = 3 # 10-fold validation for both cases
kf = KFold(n_splits = k, shuffle=True)

kf_score = []
for train_index, test_index in kf.split(X_vec):
    X_train, X_test = X_vec[train_index], X_vec[test_index]
    y_train, y_test = y_vec[train_index], y_vec[test_index]

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    kf_score.append(score)

    print(f'Iteration k-fold score: {score}')
    y_pred = clf.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=activity_list)
    plt.title("K-fold Confusion Matrix")
    plt.show()

print(f'Average k-fold score: {np.mean(kf_score)}')

# --- k-fold evaluation for each participant ---
kf = KFold(n_splits = 6, shuffle=True)
for participant in np.unique(participants):

    index = np.where(participants == participant)
    X_group = X_vec[index]
    y_group = y_vec[index]
    kf_score = []

    for train_index, test_index in kf.split(X_group):
        X_train, X_test = X_group[train_index], X_group[test_index]
        y_train, y_test = y_group[train_index], y_group[test_index]

        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        kf_score.append(score)

    print(f'Average k-fold score for participant {participant}: {np.mean(kf_score)}')



            


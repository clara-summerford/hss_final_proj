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
import sklweka.jvm as jvm
from sklweka.dataset import to_nominal_labels
from sklweka.classifiers import WekaEstimator
from sklweka.preprocessing import WekaTransfomer
from sklearn.model_selection import cross_val_score

# make able to find folder path regardless of folder you are running from
trials_path = Path("hss_final_proj/trials")
BASE_DIR = Path(__file__).resolve().parent
trials_path = BASE_DIR / "trials"

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

            # Features added: variance (for each axis), zero crossing rate (for each axis), signal energy (each axis), total signal energy, mean
            # Other maybes (frequency features): spectral entropy, HHT? , spectral centroid, MFCC
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

                    # total energy
                    features[f"imu{imu}_{sensor}_tot_sigen"] = mag.sum()

                        
                    if sensor == "accel":
                        ax = df[f"imu{imu}_accel_x"]
                        ay = df[f"imu{imu}_accel_y"]
                        az = df[f"imu{imu}_accel_z"]

                        # axis ratios
                        features[f"imu{imu}_ay_az_ratio"] = ay.abs().mean() / (az.abs().mean() + 1e-6) # making sure you don't divide by 0
                        features[f"imu{imu}_ax_az_ratio"] = ax.abs().mean() / (az.abs().mean() + 1e-6)
                        features[f"imu{imu}_ax_ay_ratio"] = ax.abs().mean() / (ay.abs().mean() + 1e-6)


                    for axis in axes:
                        col = [f"imu{imu}_{sensor}_{axis}"]
                        data = df[col]

                        # variance of each axis
                        features[f"imu{imu}_{sensor}_{axis}_var"] = data.var().item()
  
                        # zero crossing rate of each axis
                        signs = np.sign(data)
                        crossings = signs.diff().fillna(0) != 0
                        zcr = crossings.sum() / len(data)
                        features[f"imu{imu}_{sensor}_{axis}_zcr"] = zcr.item()

                        # per axis signal energy
                        mag = np.sqrt((data**2))
                        features[f"imu{imu}_{sensor}_{axis}_sigen"] = mag.sum().item()

                        # mean of each axis
                        features[f"imu{imu}_{sensor}_{axis}_mean"] = data.mean().item()


            # generating labels and appending all features
            p = imu_path.stem
            y_vec.append(p.split('_')[1])
            participants.append(p.split('_')[0])
            X_vec.append(features)


X_vec = pd.DataFrame(X_vec)
# y_vec = pd.Series(y_vec)
y_vec = np.array(y_vec)
participants = np.array(participants)

# uncomment below for troubleshooting/adding more features
# print(X_vec.shape)
# print(X_vec.columns)

# scale data using Z-score normalization
scaler = StandardScaler()
X_vec = scaler.fit_transform(X_vec)

# initialize a df to save all results in and export to spreadsheet
all_scores = pd.DataFrame(columns=['Model, Score LOPO 1', 'Score LOPO 2', 'Score LOPO 3', 'Average LOPO Score', 'K-Fold Score 1', 'K-Fold Score 2', 'K-Fold Score 3', 'K-Fold Score 4', 'Average K-Fold Score', 'Average K-Fold Score Participant 1', 'Average K-Fold Score Participant 2', 'Average K-Fold Score Participant 3' ])

# ---  SVC classifier ---
clf = svm.SVC()

# LOPO model evaluation 
print("Results for SVC classifier:")

# for some reason this isn't actually saving the name but I will work on that 
all_scores.loc[0, 'Model'] = 'SVM'
LOPO_score = []
logo = LeaveOneGroupOut()
activity_list = ["Frisbee", "Pickleball", "Baseball", "Rugby", "Lacrosse", "Tennis"] # for confusion matrix 

for i, (train_index, test_index) in enumerate(logo.split(X_vec, y_vec, groups=participants)):
    X_train, X_test = X_vec[train_index], X_vec[test_index]
    y_train, y_test = y_vec[train_index], y_vec[test_index]

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    LOPO_score.append(score)

    print(f'Score for individual LOPO iteration: {score}')
    all_scores.loc[0, f'Score LOPO {i+1}'] = score

    # create a confusion matrix for each participant to identify discrepancies
    # y_pred = clf.predict(X_test)
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=activity_list)
    # plt.title("LOPO Confusion Matrix")
    # plt.show()
    
print(f'Average LOPO score: {np.mean(LOPO_score)}')
all_scores.loc[0, 'Average LOPO Score'] = np.mean(LOPO_score)

# k-fold evaluation 
k = 4
kf = KFold(n_splits = k, shuffle=True)

kf_score = []
for i, (train_index, test_index) in enumerate(kf.split(X_vec)):
    X_train, X_test = X_vec[train_index], X_vec[test_index]
    y_train, y_test = y_vec[train_index], y_vec[test_index]

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    kf_score.append(score)

    print(f'Iteration k-fold score: {score}')
    y_pred = clf.predict(X_test)
    all_scores.loc[0, f'K-Fold Score {i+1}'] = score
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=activity_list)
    # plt.title("K-fold Confusion Matrix")
    # plt.show()

print(f'Average k-fold score: {np.mean(kf_score)}')
all_scores.loc[0, f'Average K-Fold Score'] = np.mean(kf_score)

# k-fold evaluation for each participant
kf = KFold(n_splits = 6, shuffle=True)
for i, participant in enumerate(np.unique(participants)):

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
    all_scores.loc[0, f'Average K-Fold Score Participant {i+1}'] = np.mean(kf_score)

# print(all_scores)


# --- Random Forest Classifer ---
clf = RandomForestClassifier()
print("Results for random forest classifier:")
all_scores.loc[1, 'Model'] = 'Random Forest'

# LOPO CV
LOPO_score = []
logo = LeaveOneGroupOut()
activity_list = ["Frisbee", "Pickleball", "Baseball", "Rugby", "Lacrosse", "Tennis"]

for i, (train_index, test_index) in enumerate(logo.split(X_vec, y_vec, groups=participants)):
    X_train, X_test = X_vec[train_index], X_vec[test_index]
    y_train, y_test = y_vec[train_index], y_vec[test_index]

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    LOPO_score.append(score)

    print(f'Score for individual LOPO iteration: {score}')
    all_scores.loc[1, f'Score LOPO {i+1}'] = score

print(f'Average LOPO score: {np.mean(LOPO_score)}')
all_scores.loc[1, 'Average LOPO Score'] = np.mean(LOPO_score)

# K-Fold CV
k = 4
kf = KFold(n_splits = k, shuffle=True)

kf_score = []
for i, (train_index, test_index) in enumerate(kf.split(X_vec)):
    X_train, X_test = X_vec[train_index], X_vec[test_index]
    y_train, y_test = y_vec[train_index], y_vec[test_index]

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    kf_score.append(score)

    print(f'Iteration k-fold score: {score}')
    y_pred = clf.predict(X_test)
    all_scores.loc[1, f'K-Fold Score {i+1}'] = score

print(f'Average k-fold score: {np.mean(kf_score)}')
all_scores.loc[1, f'Average K-Fold Score'] = np.mean(kf_score)

# individual participant CV
kf = KFold(n_splits = 6, shuffle=True)
for i, participant in enumerate(np.unique(participants)):

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
    all_scores.loc[1, f'Average K-Fold Score Participant {i+1}'] = np.mean(kf_score)


# -- Logistic Model Tree Model ---
# using sklearn weka plugin
jvm.start(packages=True)
y = to_nominal_labels(y)


# -- Long Short Term Memory Model ---



# export score data frame to excel/csv, important for when we eventually want to compare many different models
all_scores.to_csv("Model Scores.csv", index=False)
# all_scores.to_excel("ModelScore.xlsx", index=False)
            


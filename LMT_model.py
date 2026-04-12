import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import sklweka
import sklweka.jvm as jvm
from sklweka.classifiers import WekaEstimator
from sklweka.dataset import to_nominal_labels
from sklearn.model_selection import cross_val_score
import logging

# my working attempt to implement the LMT model, since I was having trouble am importing the normalized feature vec
# and seeing if I can go from there

# Import X_vec and y_vec, and 'participants'
X_vec = np.load('Normalized_X_vec.npy')
y_vec = np.load('y_vec.npy')
participants = np.load('Participants.npy')

# -- Logistic Model Tree Model ---
# my only troubleshooting idea left is to convert to the proper type before normalizing
# using sklearn weka plugin
logging.basicConfig(level=logging.ERROR) # this gemini suggestion did not seem to work for limiting the Warnings
jvm.start(packages=True)

clf = WekaEstimator(classname="weka.classifiers.trees.LMT") 
#options to play around with for estimator: https://weka.sourceforge.io/doc.dev/weka/classifiers/trees/LMT.html

# LOPO CV
print("Results for LMT classifier:")

# use the basic X_vec, convert to float-64 before normalization. 
# could have done this above in code, but thought it would be easier to toggle here. 
X_vec = np.asarray(X_vec)
X_vec = X_vec.astype(np.float64)
# scale data using Z-score normalization
scaler = StandardScaler()
X_vec = scaler.fit_transform(X_vec)
 
# all_scores.loc[2, 'Model'] = 'LMT'
LOPO_score = []
logo = LeaveOneGroupOut()
activity_list = ["Frisbee", "Pickleball", "Baseball", "Rugby", "Lacrosse", "Tennis"] # for confusion matrix 

# y_vec_new = to_nominal_labels(y_vec)
for i, (train_index, test_index) in enumerate(logo.split(X_vec, y_vec, groups=participants)):
    X_train, X_test = X_vec[train_index], X_vec[test_index]
    y_train, y_test = y_vec[train_index], y_vec[test_index]

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    LOPO_score.append(score)

    print(f'Score for individual LOPO iteration: {score}')
    #all_scores.loc[2, f'Score LOPO {i+1}'] = score

    # create a confusion matrix for each participant to identify discrepancies
    # y_pred = clf.predict(X_test)
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=activity_list)
    # plt.title("LOPO Confusion Matrix")
    # plt.show()
    
print(f'Average LOPO score: {np.mean(LOPO_score)}')
# all_scores.loc[2, 'Average LOPO Score'] = np.mean(LOPO_score)

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
    #all_scores.loc[2, f'K-Fold Score {i+1}'] = score

print(f'Average k-fold score: {np.mean(kf_score)}')
#all_scores.loc[2, f'Average K-Fold Score'] = np.mean(kf_score)

# Per-participant K-Fold CV
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
    #all_scores.loc[1, f'Average K-Fold Score Participant {i+1}'] = np.mean(kf_score)


jvm.stop()

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer

trials_path = Path("hss_final_proj/trials")
BASE_DIR = Path(__file__).resolve().parent
trials_path = BASE_DIR / "trials"

# -- Long Short Term Memory Model ---

# so far this is working very poorly and the code is messy, will hopefully edit later

# need to get the data into 2D form
# need to edit if I can get this to work
data = []
labels = []
w_list = []
for i, participant_path in enumerate(sorted(trials_path.iterdir())): # iterate through all folders in given data path
    if not participant_path.is_dir(): # skipping hidden files
        continue 

    for j, activity_path in enumerate(sorted(participant_path.iterdir())):
        if not activity_path.is_dir():
            continue
        
        for k, imu_path in enumerate(sorted(activity_path.iterdir())):
            if not imu_path.is_file():
                continue
            df = pd.read_csv(imu_path)
            data.append(df.iloc[1:,2:])
            [w, h] = df.shape
            w_list.append(w)
            labels.append(j)
# need to crop to the smallest time series length, so each slice of the array is the same size
# alternative may be to try padding with zeros
np.array(w_list)
size_goal = np.min(w_list) - 1 # account for taking away the column
data = [df.iloc[:size_goal, :] for df in data]

X = np.array([df.values for df in data])
y = np.array(labels)
lb = LabelBinarizer()
y = lb.fit_transform(y)
# Normalize, it seems most common to use this normalization for LSTM model
n_samp, n_time, n_stream = X.shape
scalar = MinMaxScaler()
X_2D = X.reshape(-1, n_stream)
X_2D_scaled = scalar.fit_transform(X_2D)
X = X_2D_scaled.reshape(n_samp, n_time, n_stream)


# evalute model on data
# start with K-Fold evaluation, HW 3 is helpful for copying necessary evaluation code
# K-fold validation for the whole data set
kf = KFold(4, shuffle=True, random_state=42)
accuracy_list = []

# code from homework 3 for K-Fold validation on a nueral network
for fold, (train_index, val_index) in enumerate(kf.split(X)):# Create Model            
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(n_time, n_stream)))
    model.add(LSTM(50))
    #model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    model.fit(X[train_index], y[train_index], epochs=100, batch_size=4, verbose=0)
    val_preds = np.argmax(model.predict(X[val_index]), axis=1)
    val_true = np.argmax(y[val_index], axis=1)
    accuracy = accuracy_score(val_true, val_preds)
    results = model.evaluate(X[val_index], y[val_index])
    print(results[1])
    accuracy_list.append(accuracy)
    print(accuracy)

average_accuracy = np.mean(accuracy_list)
print(f'\nAverage Accuracy Across Folds: {average_accuracy}')

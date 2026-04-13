
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, Masking
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_selection import SelectPercentile
import keras_tuner

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

#code for padding with zeros rather than cropping
size_goal = np.max(w_list) - 1
data_padded = []
for df in data:
    arr = df.to_numpy()
    padding_size = size_goal - len(arr)
    padded_arr = np.pad(arr, ((padding_size, 0), (0, 0)), mode='constant', constant_values=0)
    data_padded.append(padded_arr)
X = np.stack(data_padded)

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
print(np.shape(X))


# evalute model on data
# start with K-Fold evaluation, HW 3 is helpful for copying necessary evaluation code
# K-fold validation for the whole data set
kf = KFold(4, shuffle=True, random_state=42)
accuracy_list = []

# # code from homework 3 for K-Fold validation on a nueral network
# for fold, (train_index, val_index) in enumerate(kf.split(X)):# Create Model            
#     model = Sequential()
#     model.add(Masking(mask_value=0.0, input_shape=(n_time, n_stream))
#     model.add(LSTM(100, return_sequences=True)
#     model.add(LSTM(50))
#     #model.add(Dropout(0.2))
#     model.add(Dense(6, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     #print(model.summary())
#     model.fit(X[train_index], y[train_index], epochs=100, batch_size=4, verbose=0)
#     val_preds = np.argmax(model.predict(X[val_index]), axis=1)
#     val_true = np.argmax(y[val_index], axis=1)
#     accuracy = accuracy_score(val_true, val_preds)
#     results = model.evaluate(X[val_index], y[val_index])
#     print(results[1])
#     accuracy_list.append(accuracy)
#     print(accuracy)

# try implementing Keras tuner to see if the model can perform better
def build_model(hp):
    model = Sequential()
    #model.add(Flatten())
    # Tune number of layers
    for i in range(hp.Int("num_layers", 1, 5)):
        model.add(
            LSTM(
                units = hp.Int(f"units_{i}", min_value=50, max_value = 600, step = 20),
                activation = hp.Choice("activation", ["tanh", "relu", "sigmoid"]),
                use_bias = hp.Boolean("bias"),
                dropout = hp.Float("dropout", min_value=0, max_value=1),
                recurrent_dropout = hp.Float("recurrent dropout", min_value=0, max_value=0),
                return_sequences = True,
                input_shape = (n_time, n_stream),
            )
        )
        if hp.Boolean("dropout"):
            model.add(Dropout(rate=0.25))
        model.add(LSTM(units = 50, return_sequences = False))
        model.add(
            Dense(
                units = hp.Int("dense_units", min_value=3, max_value=25, step=1),
                activation = hp.Choice("dense_activation", ['softmax', 'relu'])
            )
        )
        model.add(Dense(6, activation = 'softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

tuner = keras_tuner.RandomSearch(
   build_model,
   objective='val_loss',
   max_trials=100,
   directory = "optimal_params", 
   project_name = "LSTM_params")

# for troubleshooting
# tuner.search_space_sumamry()


tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
best_model = tuner.get_best_models()[0]

# Get the top 2 models.
models = tuner.get_best_models(num_models=2)
best_model = models[0]
print('Best Model:')
best_model.summary()
tuner.results_summary()

# kind of confused about the 5 and 0 here
best_hps = tuner.get_best_hyperparameters(5)
model = build_model(best_hps[0])
# then do K-Fold validation

# average_accuracy = np.mean(accuracy_list)
# print(f'\nAverage Accuracy Across Folds: {average_accuracy}')


# LOPO evaluation on the model

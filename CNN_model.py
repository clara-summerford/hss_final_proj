from pathlib import Path
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import EarlyStopping
import keras_tuner
from keras.callbacks import EarlyStopping

# -- Try a different 2D CNN Model (time versus 6 sensor columns), use football paper as inspiration ---
trials_path = Path("hss_final_proj/trials")
BASE_DIR = Path(__file__).resolve().parent
trials_path = BASE_DIR / "trials"

# Import X_vec and y_vec, and 'participants'
# X_vec = np.load('Normalized_X_vec.npy')
# y_vec = np.load('y_vec.npy')
# participants = np.load('Participants.npy')

# Load data in the same way as for LSTM

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
# size_goal = np.min(w_list) - 1 # account for taking away the column
# data = [df.iloc[:size_goal, :] for df in data]

#code for padding with zeros rather than cropping
size_goal = np.max(w_list) - 1
data_padded = []
for df in data:
    arr = df.to_numpy()
    padding_size = size_goal - len(arr)
    padded_arr = np.pad(arr, ((padding_size, 0), (0, 0)), mode='constant', constant_values=0)
    data_padded.append(padded_arr)
X = np.stack(data_padded)

# X = np.array([df.values for df in data])
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


def build_model(hp):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(n_time, n_stream)))
    #model.add(Flatten())
    model.add(
        Conv1D( filters = hp.Int('filter1', min_value=100, max_value=300), 
                kernel_size = hp.Int('kernel_size1', min_value=1, max_value=5))
        )
    model.add(
        MaxPooling1D( pool_size = hp.Int('pool_size1', min_value=1, max_value = 5))
    )
    model.add(
        Conv1D( filters = hp.Int('filter2', min_value=100, max_value=300), 
                kernel_size = hp.Int('kernel_size2', min_value=1, max_value=5))
        )
    model.add(Dropout(rate=0.3))
    model.add(
        MaxPooling1D( pool_size = hp.Int('pool_size2', min_value=1, max_value = 5))
        )
    model.add(
        Conv1D( filters = hp.Int('filter3', min_value=100, max_value=300), 
                kernel_size = hp.Int('kernel_size3', min_value=1, max_value=5))
        )
    model.add(Dropout(rate=0.03))
    model.add(
        MaxPooling1D( pool_size = hp.Int('pool_size3', min_value=1, max_value = 5))
        )
    model.add(Dropout(rate=0.4))
    model.add(Flatten())
    model.add(Dense(6, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

tuner = keras_tuner.RandomSearch(
   build_model,
   objective='val_loss',
   max_trials=150,
   directory = "optimal_params_CNN", 
   project_name = "CNN_params")

# for troubleshooting
# tuner.search_space_sumamry()

tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
best_model = tuner.get_best_models()[0]

# Get the top 2 models.
models = tuner.get_best_models(num_models=2)
best_model = models[0]
print('Best Model:')
best_model.summary()
tuner.results_summary()
second_best = models[1]
print('Second Best Model:')
second_best.summary()

# # kind of confused about the 5 and 0 here
# best_hps = tuner.get_best_hyperparameters(5)
# model = build_model(best_hps[0])

# evalute model on data
# start with K-Fold evaluation
# K-fold validation for the whole data set
# kf = KFold(4, shuffle=True, random_state=42)
# accuracy_list = []

# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# for fold, (train_index, val_index) in enumerate(kf.split(X)):          
#     model = Sequential()
#     #model.add(Masking(mask_value=0.0, input_shape=(n_time, n_stream)))
        # model.add(Conv1D( filter = hp.Int('filter1', min_value=100, max_value=300), 
        #            kernel_size = np.Int('kernel_size1', min_value=1, max_value=5))
        #     )
        # model.add(
        #     MaxPooling1D( poolsize = np.Int('pool_size1', min_value=1, max_value = 5))
        # )
        # model.add(
        #     Conv1D( filter = hp.Int('filter2', min_value=100, max_value=300), 
        #            kernel_size = np.Int('kernel_size2', min_value=1, max_value=5))
        #     )
        # model.add(Dropout(rate=0.3))
        # model.add(
        #     MaxPooling1D( poolsize = np.Int('pool_size2', min_value=1, max_value = 5))
        # )
        # model.add(
        #     Conv1D( filter = hp.Int('filter3', min_value=100, max_value=300), 
        #            kernel_size = np.Int('kernel_size3', min_value=1, max_value=5))
        #     )
        # model.add(Dropout(rate=0.03))
        # model.add(
        #     MaxPooling1D( poolsize = np.Int('pool_size3', min_value=1, max_value = 5))
        # )
        # model.add(Dropout(rate=0.4))
        # model.add(Dense(6, activation = 'softmax'))
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.fit(X[train_index], y[train_index], epochs=30, batch_size=16, verbose=0, callbacks=[early_stopping])
#     val_preds = np.argmax(model.predict(X[val_index]), axis=1)
#     val_true = np.argmax(y[val_index], axis=1)
#     accuracy = accuracy_score(val_true, val_preds)
#     results = model.evaluate(X[val_index], y[val_index])
#     accuracy_list.append(accuracy)
#     print(accuracy)

# average_accuracy = np.mean(accuracy_list)
# print(f'\nAverage Accuracy Across Folds: {average_accuracy}')
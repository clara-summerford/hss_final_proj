# ECE832V - Human Signals and Systems Final Project: Team Sports Classifier

## Preprocessing
- All raw data processing is done in preprocess.py
- As of this commit (3/27), this script does not do any signal processing or filtering. This script just formats the data and extracts individual activities.
- Each activity trial is in the "trials" with the format subject_activity_trial where each CSV contains the combined accelerometer and gyroscope data across all 3 IMUs.

## Secondary processing
- For cases in which multiple activities got combined into one SensorLogger recording (i.e. forgot to press stop in between a tennis and rugby trial) the split_activities.py script separates these activities to fit into the data structure to be analyzed by preprocess.py
- For cases where the UNIX time was offset significantly (the date was incorrect and offset by a week for one trial) the time_adjust.py script uses user input to visually line up readings to correct this major offset. 

## Model training
- As of this commit (3/27), this script builds and trains an SVM model that is analyzed using LOPO, k-fold validation across all trials, and k-fold validations specific to each user. 
- This is a simple proof-of-concept model with limited success.

## Visualization
- To create data visuals for report and presentations, the visualization.py script generates IMU plots of individual trials. Alternatively, raw data could also be plotted if needed. 
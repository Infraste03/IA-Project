# The code is importing the necessary libraries and modules for data manipulation, machine learning,
# and data visualization.
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

# This code is reading three CSV files ('MLMED_Dataset_train.csv', 'ML_MED_Dataset_test.csv',
# 'ML_MED_Dataset_validation.csv') and storing the data in three separate pandas DataFrames: dfTrain,
# dfTest, and dfValidation. These DataFrames will be used for training, testing, and validation
# purposes in the subsequent code.
dfTrain = pd.read_csv('MLMED_Dataset_train.csv')
dfTest = pd.read_csv('ML_MED_Dataset_test.csv')
dfValidation= pd.read_csv('ML_MED_Dataset_validation.csv')

# The above code is using the LabelEncoder class from the scikit-learn library to encode categorical
# variables in three different dataframes: dfTrain, dfTest, and dfValidation. It loops through each
# column in each dataframe and checks if the column's data type is 'object', indicating that it is a
# categorical variable. If it is, the code applies the label encoding transformation using the
# fit_transform() method of the LabelEncoder object. This replaces the categorical values with
# numerical labels in each dataframe.
le = LabelEncoder()
for col in dfTrain.columns:
    if dfTrain[col].dtype == 'object':
        dfTrain[col] = le.fit_transform(dfTrain[col])

le = LabelEncoder()
for col in dfTest.columns:
    if dfTest[col].dtype == 'object':
        dfTest[col] = le.fit_transform(dfTest[col])
        
le = LabelEncoder()
for col in dfValidation.columns:
    if dfValidation[col].dtype == 'object':
        dfValidation[col] = le.fit_transform(dfValidation[col])


# The above code is splitting the dataset into training, validation, and test sets.
x_train = dfTrain.drop(['BLE_tot_OR_time'], axis=1)
y_train = dfTrain['BLE_tot_OR_time']
x_val = dfValidation.drop('BLE_tot_OR_time', axis=1)
y_val = dfValidation['BLE_tot_OR_time']
x_test = dfTest.drop('BLE_tot_OR_time', axis=1)
y_test = dfTest['BLE_tot_OR_time']


##############################################MODELLO XGBOOST####################################### 
print("MODELLO XGBOOST")

# The above code is creating an instance of the XGBRegressor class from the XGBoost library in Python.
# It is specifying the parameters for the XGBoost model, including the number of estimators (500), the
# maximum depth of each tree (5), the fraction of columns to be randomly sampled for each tree (1.0),
# the fraction of samples to be randomly sampled for each tree (0.5), the learning rate (0.01), and
# the number of parallel threads to be used for training (4).
xgb_model = XGBRegressor(n_estimators=500, max_depth = 5, colsample_bytree=1.0, 
                     subsample=0.5, learning_rate=0.01, n_jobs=4)

# The above code is fitting an XGBoost model to the training data (x_train and y_train). It uses early
# stopping with a maximum of 10 rounds without improvement in performance on the validation set (x_val
# and y_val). The model's performance is evaluated on the validation set during training. The verbose
# parameter is set to False, so no progress will be printed during training.
xgpred = xgb_model.fit(x_train, y_train,
                early_stopping_rounds=10,
                eval_set=[(x_val, y_val)],
                verbose=False)

# The above code is calculating the best iteration of an XGBoost model, making predictions on a
# validation dataset, and calculating the mean absolute error (MAE) between the predictions and the
# actual values.
best_iteration = xgb_model.best_iteration
predictions_1 = xgb_model.predict(x_val)
mae_1 = mean_absolute_error(predictions_1, y_val)
print("Mean Absolute Error:" , mae_1)


#IMPORTANT: The hyperparament search code is commented out for computational reasons,
#has been run and the following best parameters were found:
#n_estimators=500, max_depth = 5, colsample_bytree=1.0, subsample=0.5, learning_rate=0.01
#which were already included in the previous xgb_model.

""" from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold, cross_val_score

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.8, 1.0]
}
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# Create a grid search object
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1,  verbose=2)

# Perform k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(grid_search, x_train, y_train, cv=kfold, scoring='neg_mean_absolute_error')

# Print the mean MAE and standard deviation
print("Mean MAE: ", -np.mean(scores))
print("Standard deviation: ", np.std(scores)) """


############RANDOM FOREST######################

print("MODELLO Random Forest")

# The above code is creating a random forest regression model using the RandomForestRegressor class
# from the scikit-learn library in Python. The model is being initialized with the following
# parameters:
rf_model = RandomForestRegressor(bootstrap= True, 
                                 max_depth= 90, 
                                 max_features=None,
                                 min_samples_leaf= 3,
                                 min_samples_split= 2, 
                                 n_estimators= 100)

#best parameters:  {'bootstrap': True, 'max_depth': 90, 'max_features': None, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100}
# Fit on training data
rf_model.fit(x_train, y_train)


# The above code is calculating the mean absolute error (MAE) of the predictions made by
# forest model (rf_model) on the training data (x_train) compared to the actual target values
# (y_train). The calculated MAE is then printed to the console.
rf_predictions = rf_model.predict(x_train)
mae_rf = mean_absolute_error(y_train, rf_predictions)
print("Mean Absolute Error:", mae_rf)
# The above code is calculating the mean absolute error (MAE) between the predicted values (y_pred)
# and the actual values (y_test) using a random forest model (rf_model). The calculated MAE is then
# printed to the console.
y_pred = rf_model.predict(x_test)
mae_rf2 = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae_rf2)


#IMPORTANT: The hyperparament search code is commented out for computational reasons,
#has been run and the following best parameters were found:
#{'bootstrap': True, 'max_depth': 90, 'max_features': None, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100}
#which were already included in the previous xgb_model.

""""
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, None],
    'max_features': ["sqrt", "log2", None],
    'min_samples_leaf': [1, 3, 5],
    'min_samples_split': [2, 3, 4],
    'n_estimators': [10, 25, 50, 75, 100]
}

rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, verbose =2, scoring='r2',  n_jobs = -1)
best_grid = grid_search.fit(x_train, y_train)
print("Best parameters: ", best_grid.best_params_)

#best parameters:  {'bootstrap': True, 'max_depth': 90, 'max_features': None, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100}"""


#######################################MODELLO NEURAL NETWORK#################################
print("MODELLO NEURAL NETWORK")
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping,ReduceLROnPlateau

# The above code is creating a neural network model using the Keras library in Python.
nn_model = Sequential()
nn_model.add(Dense(100, input_shape=(62,), activation='relu'))
nn_model.add(Dense(100, activation='relu'))
nn_model.add(Dense(1, activation='linear'))


# The above code is compiling a neural network model in Python. It is specifying the loss function as
# mean absolute error, the optimizer as Adam, and the metric to evaluate the model as mean absolute
# error.
nn_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
# The above code is converting the data arrays `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, and
# `y_test` into TensorFlow tensors with a data type of `float32`.
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)


# The above code is creating an instance of the EarlyStopping callback class in Python. This callback
# is commonly used in machine learning models to stop training early if a certain condition is met.
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
# The above code is creating a learning rate scheduler object using the ReduceLROnPlateau class from
# the Keras library. This scheduler is used to dynamically adjust the learning rate during training
# based on the validation loss.
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min', min_lr=1e-6)
# The above code is training a neural network model using the fit() function. It takes in the training
# data (x_train and y_train) and validation data (x_val and y_val) as inputs. The model is trained for
# 100 epochs with a batch size of 50. It also includes callbacks, such as early stopping and learning
# rate scheduler, which are used to monitor the training process and make adjustments accordingly.
history = nn_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=50, callbacks=[early_stopping, lr_scheduler])
# Evaluate the model
scores = nn_model.evaluate(x_test, y_test)
print("Mean Absolute Error: ", scores[1])

###kfold cross validation###

from sklearn.model_selection import  KFold
from sklearn.metrics import mean_absolute_error


# The above code is creating a KFold object for performing k-fold cross-validation. The number of
# splits is set to 5, meaning that the data will be divided into 5 equal-sized folds. The shuffle
# parameter is set to True, which means that the data will be randomly shuffled before splitting into
# folds. The random_state parameter is set to 1, which ensures that the random shuffling is
# reproducible.
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
# The above code is converting the `x_train` and `y_train` tensors into pandas DataFrames.
x_train = pd.DataFrame(x_train.numpy())
y_train = pd.DataFrame(y_train.numpy())
# Define the models
models = [xgb_model, rf_model, nn_model]

# Loop over the models
for model in models:
    model_name = model.__class__.__name__  # Get the name of the model

    # List to save the results of cross-validation
    mae_scores = []

    # Loop over the folds
    for train_index, test_index in kf.split(x_train):
        x_train_fold, x_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Train the model
        model.fit(x_train_fold, y_train_fold)

        # Make predictions
        predictions = model.predict(x_test_fold)

        # Calculate the MAE
        mae = mean_absolute_error(y_test_fold, predictions)
        mae_scores.append(mae)

    # Calculate the mean MAE over all folds
    avg_mae = np.mean(mae_scores)
    print(f"{model_name} - Mean Absolute Error: {avg_mae}")
    

    
################################feature selection ###################
#IMPRTANT: the following function is used only for random forest and xgboost models.
#for reasons of feature selection correctness of the neural network.
def train_and_evaluate(model, x_train, y_train, x_val, y_val, feature_indices=None):
    
    # The code is checking if the variable `feature_indices` is not `None`. If it is not `None`, it
    # selects only the columns specified by `feature_indices` from the `x_train` and `x_val` arrays.
    # Then, it fits a model using the modified `x_train` and `y_train` arrays.
    if feature_indices is not None:
        x_train = x_train[:, feature_indices]
        x_val = x_val[:, feature_indices]
    model.fit(x_train, y_train)

    
    # The above code is predicting the target variable values (y_train_pred) using a trained model on
    # the training data (x_train). It then calculates the mean absolute error (mean_score) between the
    # actual target variable values (y_train) and the predicted values (y_train_pred).
    y_train_pred = model.predict(x_train)
    mean_score = mean_absolute_error(y_train, y_train_pred)

    # Evaluate the model on the validation data
    y_val_pred = model.predict(x_val)
    test_score = mean_absolute_error(y_val, y_val_pred)

    return mean_score, test_score


import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

# Select the top 5 features using mutual information
selector = SelectKBest(score_func=mutual_info_regression, k=5)
selector.fit(x_train, y_train)
# Get the indices of the selected features
selected_indices = np.where(selector.get_support())[0]
# The above code is selecting specific columns from the `x_train` and `x_val` datasets. It is using
# the `selected_indices` variable to specify which columns to select. The selected columns are then
# assigned to the `x_train_selected` and `x_val_selected` variables, respectively.
x_train_selected = x_train.values[:, selected_indices]
x_val_selected = np.array(x_val)[:, selected_indices]

models = [xgb_model, rf_model]
scores_all = []
scores_selected = []

# The above code is iterating over a list of models and for each model, it trains and evaluates the
# model using all features and selected features. It then prints the mean score and test score for
# both cases.
for model in models:
    # Train and evaluate the model using all features
    mean_score_all, test_score_all = train_and_evaluate(model, x_train.values, y_train, np.array(x_val), y_val)

    # Train and evaluate the model using selected features
    mean_score, test_score = train_and_evaluate(model, x_train_selected, y_train, x_val_selected, y_val)
    
    

    # Print the results
    print("Model:", type(model).__name__)
    print("All features:")
    print("Mean score: {:.4f}".format(mean_score_all))
    print("Test score: {:.4f}".format(test_score_all))

    print("Selected features:")
    print("Mean score: {:.4f}".format(mean_score))
    print("Test score: {:.4f}".format(test_score))
    


#######PER LA RETE NEURALE#########


# The above code is selecting specific columns from the input tensors `x_train`, `x_val`, and `x_test`
# based on the `selected_indices` array. It uses the `tf.gather` function to gather the columns
# specified by `selected_indices` along the specified axis (axis=1 in this case). The resulting
# tensors `x_train_selected`, `x_val_selected`, and `x_test_selected` will contain only the selected
# columns from the original tensors.
x_train_selected = tf.gather(x_train, selected_indices, axis=1)
x_val_selected = tf.gather(x_val, selected_indices, axis=1)
x_test_selected = tf.gather(x_test, selected_indices, axis=1)

# CNew model with selected features
nn_model_selected = Sequential()
nn_model_selected.add(Dense(100, input_shape=(len(selected_indices),), activation='relu'))
nn_model_selected.add(Dense(100, activation='relu'))
nn_model_selected.add(Dense(1, activation='linear'))


nn_model_selected.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


# The above code is converting the numpy arrays `x_train_selected`, `x_val_selected`, and
# `x_test_selected` into TensorFlow tensors of type `tf.float32`.
x_train_selected_tensor = tf.convert_to_tensor(x_train_selected, dtype=tf.float32)
x_val_selected_tensor = tf.convert_to_tensor(x_val_selected, dtype=tf.float32)
x_test_selected_tensor = tf.convert_to_tensor(x_test_selected, dtype=tf.float32)

# The above code is training a neural network model called `nn_model_selected` using the `fit` method.
# It is using the `x_train_selected_tensor` as the input data and `y_train` as the target labels. It
# also includes a validation set using `x_val_selected_tensor` and `y_val` for validation during
# training. The model is trained for 100 epochs with a batch size of 50. It also includes callbacks
# such as `early_stopping` and `lr_scheduler` for early stopping and learning rate scheduling
# respectively.
history_selected = nn_model_selected.fit(x_train_selected_tensor, y_train, validation_data=(x_val_selected_tensor, y_val), epochs=100, batch_size=50, callbacks=[early_stopping, lr_scheduler])


# The above code is evaluating the performance of a neural network model on a test dataset. It uses
# the `evaluate` method of the `nn_model_selected` object to calculate the model's performance
# metrics. The code then prints the mean absolute error (MAE) of the model's predictions using the new
# features.
scores_selected = nn_model_selected.evaluate(x_test_selected_tensor, y_test)
print("Mean Absolute Error con le nuove features: ", scores_selected[1])

# The above code is plotting the training and validation loss for two different sets of features. It
# is comparing the loss for the original set of features and a new set of features. The code uses the
# `plt.plot()` function to plot the loss values from the `history` and `history_selected` objects. The
# labels are provided for each line in the plot. The title, x-axis label, y-axis label, and legend are
# also set before displaying the plot using `plt.show()`.
plt.plot(history.history['loss'], label='Train Loss (tutte le features)')
plt.plot(history.history['val_loss'], label='Validation Loss (tutte le features)')
plt.plot(history_selected.history['loss'], label='Train Loss (nuove features)')
plt.plot(history_selected.history['val_loss'], label='Validation Loss (nuove features)')
plt.title('Training/Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



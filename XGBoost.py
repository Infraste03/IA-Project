#file per modello XGBoost 

import pandas as pd

from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, roc_curve
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from xgboost import XGBRegressor

# Importo i dataset
dfTrain = pd.read_csv('MLMED_Dataset_train.csv')
dfTest = pd.read_csv('ML_MED_Dataset_test.csv')
dfValidation= pd.read_csv('ML_MED_Dataset_validation.csv')


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

#split fra x_train e y_train 
#split fra x_test e y_test
#split fra x_validation e y_validation

x_train = dfTrain.drop(['BLE_tot_OR_time'], axis=1)
y_train = dfTrain['BLE_tot_OR_time']
x_val = dfValidation.drop('BLE_tot_OR_time', axis=1)
y_val = dfValidation['BLE_tot_OR_time']
x_test = dfTest.drop('BLE_tot_OR_time', axis=1)
y_test = dfTest['BLE_tot_OR_time']

#MODELLO XGBOOST 
print("MODELLO XGBOOST")

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
xgb_model = XGBRegressor(n_estimators=500, max_depth = 5, colsample_bytree=1.0, 
                     subsample=0.5, learning_rate=0.01, n_jobs=4)

xgpred = xgb_model.fit(x_train, y_train,
                early_stopping_rounds=10,
                eval_set=[(x_val, y_val)],
                verbose=False)

best_iteration = xgb_model.best_iteration
# Get predictions
predictions_1 = xgb_model.predict(x_val)

# Calculate MAE
mae_1 = mean_absolute_error(predictions_1, y_val)

print("Mean Absolute Error:" , mae_1)

""" from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

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


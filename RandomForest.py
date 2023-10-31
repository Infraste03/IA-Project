#file per random forest

import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import warnings

from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore")


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

print("MODELLO Random Forest")
from sklearn.metrics import mean_absolute_error, r2_score

rf_model = RandomForestRegressor(bootstrap= True, 
                                 max_depth= 90, 
                                 max_features=None,
                                 min_samples_leaf= 3,
                                 min_samples_split= 2, 
                                 n_estimators= 100)
# Fit on training data
rf_model.fit(x_train, y_train)

# Actual class predictions
#training accuracy 
rf_predictions = rf_model.predict(x_train)
mae_rf = mean_absolute_error(y_train, rf_predictions)
print("Mean Absolute Error:", mae_rf)

#testing accuracy

y_pred = rf_model.predict(x_test)
mae_rf2 = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae_rf2)

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








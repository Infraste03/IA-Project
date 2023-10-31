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


##FRANCESCA RICORDATI CHE DEVI FATE LA GRIDH SEARCH, QUANDO HAI TROVATO I PAAMENTRI MIGLIORI ALLORA òI SOSTITUOSCI NEL MODELLI
#POI FAI LA KFOLD VALIDATION E E CONFRONTI I TRE DIFFERENTI MODELLI PER VEDERE QUALE SIA MIGLIROE 
#MAGARI FALLO IN UN FOR IN QUESTO MODO CONTRONTIT IN UN UNICA BOTTA 
#DEVI FARTI STAMPERE IL VALORE PREDETTO TIPO O IN UN QUALCHE MODO FARE COSì

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from xgboost import XGBRegressor


# Importing the dataset
dfTrain = pd.read_csv('MLMED_Dataset_train.csv')
dfTest = pd.read_csv('ML_MED_Dataset_test.csv')
dfValidation= pd.read_csv('ML_MED_Dataset_validation.csv')

#per rimuovere le righe che hanno valori mancanti 
#puoi usare isNull e vedere se ce ne sono 
dfTrain.dropna(inplace=True)
dfTest.dropna(inplace=True)
dfValidation.dropna(inplace=True)
######
#PER LA VISUALIZZAZIONE
#####

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

#split the data into indipendent and dependent variables
x_train = dfTrain.drop('BLE_tot_OR_time', axis=1)
y_train = dfTrain['BLE_tot_OR_time']
x_val = dfValidation.drop('BLE_tot_OR_time', axis=1)
y_val = dfValidation['BLE_tot_OR_time']
x_test = dfTest.drop('BLE_tot_OR_time', axis=1)
y_test = dfTest['BLE_tot_OR_time']



#XGBOOST
print("###################XGBOOST###################")
#create an xgboost regression model
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
xgb_model = XGBRegressor(n_estimators=1000, max_depth = 7, eta=0.1, 
                     subsample=0.7, colsample_bytree=0.8, learning_rate=0.05, n_jobs=4)

#for evaluation of the regression model we use  cross validation
#cv is the number of cross validation sets we want to use
#mae is the mean absolute error
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(xgb_model, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )

xgpred = xgb_model.fit(x_train, y_train,
                early_stopping_rounds=5,
                eval_set=[(x_val, y_val)],
                verbose=False)

# Get predictions
predictions_1 = xgb_model.predict(x_val)

# Calculate MAE
mae_1 = mean_absolute_error(predictions_1, y_val)

print("Mean Absolute Error:" , mae_1)

""" from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.8, 1.0]
}

# Create a grid search object
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1,  verbose=2)

# Fit the grid search object to the training data
grid_search.fit(x_train, y_train)

# Print the best hyperparameters and mean MAE
print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Mean MAE: ", -grid_search.best_score_)  """


#RANDOM FOREST
print("###################RANDOM FOREST###################")
# Create the model with 100 trees
rf_model = RandomForestRegressor(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
rf_model.fit(x_train, y_train)

# Actual class predictions
rf_predictions = rf_model.predict(x_val)


mae_value = mean_absolute_error(y_val, rf_predictions)
print("Mean Absolute Error: ", mae_value)


"""

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(rf_model, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# Print the mean and standard deviation of the scores
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))
 """

""" from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Create a random forest regression model
rf_model = RandomForestRegressor()

# Create a grid search object with verbose output
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search object to the training data
grid_search.fit(x_train, y_train)

# Print the best hyperparameters and mean MAE
print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Mean Absolute Error: ", -grid_search.best_score_) """


print("###################NEURAL NETWORK ###################")

#Import packages
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping,ReduceLROnPlateau


# Define the model architecture
nn_model = Sequential()
nn_model.add(Dense(100, input_shape=(62,), activation='relu'))
nn_model.add(Dense(100, activation='relu'))
nn_model.add(Dense(1, activation='linear'))

# Compile the model
nn_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

# Convert the data to tensors
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Define the callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min', min_lr=1e-6)

# Fit the model with callbacks
history = nn_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=50, callbacks=[early_stopping, lr_scheduler])

# Evaluate the model
scores = nn_model.evaluate(x_test, y_test)
print("Mean Absolute Error: ", scores[1])

# Make predictions on the test data
predictions = nn_model.predict(x_test)
"""
# Plot the training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.legend()
plt.show()

# Plot the training and validation MAE
mae = history.history['mean_absolute_error']
val_mae = history.history['val_mean_absolute_error']
plt.plot(mae, label='Training MAE')
plt.plot(val_mae, label='Validation MAE')
plt.legend()
plt.show()
 """
####kfold cross validation per il confronto delle prestazione dei modelli 
from sklearn.model_selection import KFold
import numpy as np
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

def train_and_evaluate(model, x_train, y_train, x_val, y_val, feature_indices=None):
    if feature_indices is not None:
        x_train = x_train[:, feature_indices]
        x_val = x_val[:, feature_indices]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_index, val_index in kf.split(x_train):
        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        model.fit(x_train_fold, y_train_fold)
        y_pred = model.predict(x_val_fold)
        score = mean_absolute_error(y_val_fold, y_pred)
        scores.append(score)
    mean_score = np.mean(scores)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    test_score = mean_absolute_error(y_val, y_pred)
    return mean_score, test_score



#feture selection
from sklearn.metrics import mean_squared_error


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

print("###################FEATURE SELECTION###################")
k = 7
#mutual_info = mutual_info_regression(x_train, y_train)
mi = mutual_info_regression(x_train, y_train.numpy())
selector = SelectKBest(score_func=mutual_info_regression, k=k)

selector.fit(x_train, y_train.numpy())

selected_indices = selector.get_support()

x_train_numpy= x_train.numpy()
x_val_numpy = x_val.numpy()

x_train_selected = x_train_numpy[:, selected_indices]
x_val_selected = x_val_numpy[:, selected_indices]

import matplotlib.pyplot as plt
import numpy as np

# Train and evaluate models using all features
rf_mean_score_all, rf_test_score_all = train_and_evaluate(rf_model, x_train, y_train, x_val, y_val)
rf_mean_score, rf_test_score = train_and_evaluate(rf_model, x_train_selected, y_train, x_val_selected, y_val)
print("Random Forest:")
print("Mean score: {:.4f}".format(rf_mean_score_all))
print("Mean score new: {:.4f}".format(rf_mean_score))
print("Test score: {:.4f}".format(rf_test_score_all))
print("Test score new : {:.4f}".format(rf_test_score))

xgb_mean_score_all, xgb_test_score_all = train_and_evaluate(xgb_model, x_train, y_train, x_val, y_val)
xgb_mean_score, xgb_test_score = train_and_evaluate(xgb_model, x_train_selected, y_train, x_val_selected, y_val)
print("XGBoost:")
print("Mean score: {:.4f}".format(xgb_mean_score_all))
print("Mean score new: {:.4f}".format(xgb_mean_score))
print("Test score: {:.4f}".format(xgb_test_score_all))
print("Test score: {:.4f}".format(xgb_test_score))



nn_mean_score_all, nn_test_score_all = train_and_evaluate(nn_model, x_train, y_train, x_val, y_val)
nn_mean_score, nn_test_score = train_and_evaluate(nn_model, x_train_selected, y_train, x_val_selected, y_val,feature_indices=selected_features)




print("Neural Network:")
print("Mean score: {:.4f}".format(nn_mean_score_all))
print("Test score: {:.4f}".format(nn_test_score_all))



""" 
# Train and evaluate models using selected features
rf_mean_score, rf_test_score = train_and_evaluate(rf_model, x_train_selected, y_train, x_val_selected, y_val)
xgb_mean_score, xgb_test_score = train_and_evaluate(xgb_model, x_train_selected, y_train, x_val_selected, y_val)
nn_mean_score, nn_test_score = train_and_evaluate(nn_model, x_train_selected, y_train, x_val_selected, y_val)



# Create a bar chart to compare the performance of the models
models = ['Random Forest', 'XGBoost', 'Neural Network']
mean_scores_all = [rf_mean_score_all, xgb_mean_score_all, nn_mean_score_all]
test_scores_all = [rf_test_score_all, xgb_test_score_all, nn_test_score_all]
mean_scores = [rf_mean_score, xgb_mean_score, nn_mean_score]
test_scores = [rf_test_score, xgb_test_score, nn_test_score]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mean_scores_all, width, label='Mean Score (All Features)')
rects2 = ax.bar(x + width/2, mean_scores, width, label='Mean Score (Selected Features)')
rects3 = ax.bar(x - width/2, test_scores_all, width, label='Test Score (All Features)', alpha=0.5)
rects4 = ax.bar(x + width/2, test_scores, width, label='Test Score (Selected Features)', alpha=0.5)

ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

fig.tight_layout()

plt.show() 
 """















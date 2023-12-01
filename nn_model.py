#file per la rete neurale 
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
dfTrain = dfTrain.drop(['BLE_tot_BO_time', 'BLE_tot_RR_time', 'Tempo Tot BO Ormaweb', 'Tempo Tot. SO OrmaWeb','Tempo Tot. RR', 'LATERALITA_BILATERALE','MALLAMPATI_4.0', 'TORACOSCOPIA','MIVAR', 'PERCUTANEO','OSAS', 'ENDOSCOPIA', 
                        'Antipertensivi', 'Insulina', 'Reintervento', 'Fumo', 'CVC', 'BPCO', 'feasible', 'TIGO', 'Anticoagulanti', 'Pregresso TIA', 'Codice alfa numerico'], axis=1)

dfTest = dfTest.drop(['BLE_tot_BO_time', 'BLE_tot_RR_time', 'Tempo Tot BO Ormaweb', 'Tempo Tot. SO OrmaWeb','Tempo Tot. RR', 'LATERALITA_BILATERALE', 'MALLAMPATI_4.0', 
                      'TORACOSCOPIA','MIVAR','PERCUTANEO','OSAS','ENDOSCOPIA', 'Antipertensivi','Insulina', 'Reintervento','Fumo', 'CVC', 'BPCO', 'feasible', 'TIGO'
                      , 'Anticoagulanti', 'Pregresso TIA', 'Codice alfa numerico'], axis=1)


dfValidation = dfValidation.drop(['BLE_tot_BO_time', 'BLE_tot_RR_time', 'Tempo Tot BO Ormaweb', 'Tempo Tot. SO OrmaWeb','Tempo Tot. RR',
                                  'LATERALITA_BILATERALE','MALLAMPATI_4.0', 'TORACOSCOPIA','MIVAR', 'PERCUTANEO','OSAS', 'ENDOSCOPIA', 
                                  'Antipertensivi', 'Insulina', 'Reintervento', 'Fumo', 'CVC', 'BPCO', 'feasible',
                                  'TIGO', 'Anticoagulanti', 'Pregresso TIA', 'Codice alfa numerico'], axis=1)

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

print("MODELLO NEURAL NETWORK")
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras import backend as K
from keras.layers import Dropout


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

# Define the model
nn_model = Sequential()
nn_model.add(Dense(100, input_shape=(39,), activation='relu'))
nn_model.add(Dense(100, activation='relu'))
nn_model.add(Dense(100, activation='relu'))  # Added layer
nn_model.add(Dense(100, activation='relu'))  # Added layer
nn_model.add(Dense(1, activation='linear'))

# Compile the model
nn_model.compile(loss=root_mean_squared_error, optimizer='adam', metrics=[root_mean_squared_error])

# Convert data to tensors
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min', min_lr=1e-6)

# Train the model
history = nn_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=50, callbacks=[early_stopping, lr_scheduler])

# Evaluate the model
scores = nn_model.evaluate(x_test, y_test)
print("RMSE:", scores[1])

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
dfTrain = pd.read_csv('MLMED_Dataset_train.csv')
from sklearn.calibration import LabelEncoder

# Assuming that dfTrain is your DataFrame and 'BLE_tot_OR_time' is your target variable

le = LabelEncoder()
for col in dfTrain.columns:
    if dfTrain[col].dtype == 'object':
        dfTrain[col] = le.fit_transform(dfTrain[col])
X_train = dfTrain.drop(['BLE_tot_OR_time'], axis=1)
y_train = dfTrain['BLE_tot_OR_time']
# Split the data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for visualization
feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': importances})

# Sort the DataFrame by importance score
feature_importances = feature_importances.sort_values('importance', ascending=False)

# Print the DataFrame
print(feature_importances)
import matplotlib.pyplot as plt

# Create a DataFrame for visualization
feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': importances})

# Sort the DataFrame by importance score
feature_importances = feature_importances.sort_values('importance', ascending=False)

# Create a bar plot
plt.figure(figsize=(10, 8))
plt.barh(feature_importances['feature'], feature_importances['importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature importances')
plt.gca().invert_yaxis()  # Invert the y-axis to show the feature with the highest importance at the top
plt.show()
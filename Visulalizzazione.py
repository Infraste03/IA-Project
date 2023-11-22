import pandas as pd
import matplotlib.pyplot as plt

dfTrain = pd.read_csv('MLMED_Dataset_train.csv')

#print( "info dataset train " , dfTrain.info())
#print(dfTrain.value_counts('BLE_tot_OR_time'))
#print(dfTrain.isnull().sum())

import pandas as pd
import matplotlib.pyplot as plt

# Print the column names
print(dfTrain.columns)

# Print the first few rows of the DataFrame
print(dfTrain.head())

# Print the summary statistics of the DataFrame
print(dfTrain.describe())

# Plot histograms for all numeric columns
# List of column names you want to plot
columns_to_plot = ['Codice alfa numerico', 'Altezza', 'BLE_tot_BO_time', 'Numero chirurghi', 'Pregresso SCC', 'ASA_2.0']

# Select the columns and plot
dfTrain[columns_to_plot].hist(bins=50, figsize=(20,15))
plt.show()
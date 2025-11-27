"""
Data Normalize/Standardize Assignment
"""

"""
Load libraries & set-up environment
"""
# Load libraries
import pandas as pd
import numpy as numpy

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Set up Python enviornment
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

"""
Load file & set up dataframe
"""
# Load in file
FILE = "HMEQ_Loss.csv"

df = pd.read_csv(FILE)

# Target variables
TARGET_F = "TARGET_BAD_FLAG"
TARGET_A = "TARGET_LOSS_AMT"

# Get data types
dt = df.dtypes
objList = [] # Initialize empty list
numList = [] # Initialize empty list
for i in dt.index:
    if i in ([TARGET_F, TARGET_A]):
        continue
    if dt[i] in (["object"]):
        objList.append(i)
    if dt[i] in (["float64", "int64"]):
        numList.append(i)

# Create copy
X = df.copy()
X = X.drop(objList, axis=1)
X = X.drop([TARGET_F, TARGET_A], axis=1)
varNames = X.columns
print(X.head())

"""
MIN/MAX SCALER - Automatic
"""
# Create & trainer scaler
theScaler = MinMaxScaler()
theScaler.fit(X) # Train scaler

# Transform data
X_MINMAX = theScaler.transform(X)

# Create dataframe
X_MINMAX = pd.DataFrame(X_MINMAX)
print("\n=======================\n")
print("MIN/MAX NORMALIZED DATA\n")
print(X_MINMAX.head())

# Edit variable names to indicated transformed
varNames_minmax = [] # Initialize empty list
for i in varNames:
    newName = "nor_" + i
    varNames_minmax.append(newName)
print("\n=======================\n")
print("MIN/MAX Edited Variable Names\n")
print(varNames_minmax)

# Append new column names
X_MINMAX.columns = varNames_minmax

# Append categorical variables
X_MINMAX[objList] = df[objList]
print("\n=======================\n")
print("Appending Categorical Variables\n")
print(X_MINMAX.head().T)

# Append TARGET variables
X_MINMAX[TARGET_F] = df.TARGET_BAD_FLAG
X_MINMAX[TARGET_A] = df.TARGET_LOSS_AMT
print("\n=======================\n")
print("MIN/MAX Final dataset\n")
print(X_MINMAX.head().T)

# Combine dataframes
X_NEW = pd.concat([X, X_MINMAX], axis=1)
print("\n=======================\n")
print("MIN/MAX Combined dataset\n")
print(X_NEW.head().T)

"""
MIN/MAX SCALER - Manual
"""
# Isolate 1 numeric variable
X_TEST = X_NEW[["MORTDUE", "nor_MORTDUE"]]
print("\n=======================\n")
print("MORTDUE Variable\n")
print(X_TEST.head())

# Determine min/max
desc_stats = X_TEST["MORTDUE"].describe()
min_val = desc_stats.loc["min"]
max_val = desc_stats.loc["max"]
print("\n=======================\n")
print("Descriptive Statistics\n")
print(desc_stats)

# Manually calculate and transform
TEMP = (X_TEST["MORTDUE"] - min_val) / (max_val - min_val)
X_TEST = X_TEST.assign(calc_MORTDUE = TEMP.values) # Add calc value
print("\n=======================\n")
print("MIN/MAX Manual Calculation Comparison\n")
print(X_TEST.head())

"""
STANDARD SCALER - Automatic
"""
# Create & trainer scaler
theScaler = StandardScaler()
theScaler.fit(X) # Train scaler

# Transform data
Y_STD = theScaler.transform(X)

# Create dataframe
Y_STD = pd.DataFrame(Y_STD)
print("\n=======================\n")
print("STANDARDIZED NORMALIZED DATA\n")
print(Y_STD.head())

# Edit variable names to indicated transformed
varNames_std = [] # Initialize empty list
for i in varNames:
    newName = "std_" + i
    varNames_std.append(newName)
print("\n=======================\n")
print("STANDARDIZED Edited Variable Names\n")
print(varNames_std)

# Append new column names
Y_STD.columns = varNames_std

# Append categorical variables
Y_STD[objList] = df[objList]
print("\n=======================\n")
print("Appending Categorical Variables\n")
print(Y_STD.head().T)

# Append TARGET variables
Y_STD[TARGET_F] = df.TARGET_BAD_FLAG
Y_STD[TARGET_A] = df.TARGET_LOSS_AMT
print("\n=======================\n")
print("STANDARDIZED Final Dataset\n")
print(Y_STD.head().T)

# Combine dataframes
Y_NEW = pd.concat([X, Y_STD], axis=1)
print("\n=======================\n")
print("STANDARDIZED Combined dataset\n")
print(Y_NEW.head().T)

"""
STANDRD SCALER - Manual
"""
# Isolate 1 numeric variable
Y_TEST = Y_NEW[["MORTDUE", "std_MORTDUE"]]
print("\n=======================\n")
print("MORTDUE Variable\n")
print(Y_TEST.head())

# Determine min/max
desc_stats = Y_TEST["MORTDUE"].describe()
min_val = desc_stats.loc["min"]
max_val = desc_stats.loc["max"]
print("\n=======================\n")
print("Descriptive Statistics\n")
print(desc_stats)

# Manually calculate and transform
TEMP = (Y_TEST["MORTDUE"] - min_val) / (max_val - min_val)
Y_TEST = Y_TEST.assign(calc_MORTDUE = TEMP.values) # Add calc value
print("\n=======================\n")
print("STANDARDIZED Manual Calculation Comparison\n")
print(Y_TEST.head())

"""
BINGO BONUS - Remove outliers
"""
# Determine outliers
print("\n=============\n")
print("Removing outliers")
print("df shape:", df.shape)
print("\ndf stats:\n", df.describe().T)

# Remove outliers
Q1 = df[TARGET_A].quantile(0.25)
Q3 = df[TARGET_A].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

outlier_mask =(df[TARGET_A] > lower) & (df[TARGET_A] < upper)

df = df.loc[outlier_mask]
df = df.loc[outlier_mask]

# Print results
print("\nFinal df shape (w/o outliers):", df.shape)
print("\nFinal df stats (w/o outliers):\n", df.describe().T)

# Create copy
X = df.copy()
X = X.drop(objList, axis=1)
X = X.drop([TARGET_F, TARGET_A], axis=1)
varNames = X.columns
print("\n", X.head())

"""
MIN/MAX SCALER - Automatic (w/o outliers)
"""
# Create & trainer scaler
theScaler = MinMaxScaler()
theScaler.fit(X) # Train scaler

# Transform data
X_MINMAX = theScaler.transform(X)

# Create dataframe
X_MINMAX = pd.DataFrame(X_MINMAX)
print("\n=======================\n")
print("MIN/MAX NORMALIZED DATA (w/o outliers)\n")
print(X_MINMAX.head())

# Edit variable names to indicated transformed
varNames_minmax = [] # Initialize empty list
for i in varNames:
    newName = "nor_" + i
    varNames_minmax.append(newName)
print("\n=======================\n")
print("MIN/MAX Edited Variable Names\n")
print(varNames_minmax)

# Append new column names
X_MINMAX.columns = varNames_minmax

# Append categorical variables
X_MINMAX[objList] = df[objList]
print("\n=======================\n")
print("Appending Categorical Variables (w/o outliers)\n")
print(X_MINMAX.head().T)

# Append TARGET variables
X_MINMAX[TARGET_F] = df.TARGET_BAD_FLAG
X_MINMAX[TARGET_A] = df.TARGET_LOSS_AMT
print("\n=======================\n")
print("MIN/MAX Final dataset (w/o outliers)\n")
print(X_MINMAX.head().T)

# Combine dataframes
X_NEW = pd.concat([X, X_MINMAX], axis=1)
print("\n=======================\n")
print("MIN/MAX Combined dataset (w/o outliers)\n")
print(X_NEW.head().T)

"""
MIN/MAX SCALER - Manual (w/o outliers)
"""
# Isolate 1 numeric variable
X_TEST = X_NEW[["MORTDUE", "nor_MORTDUE"]]
print("\n=======================\n")
print("MORTDUE Variable\n")
print(X_TEST.head())

# Determine min/max
desc_stats = X_TEST["MORTDUE"].describe()
min_val = desc_stats.loc["min"]
max_val = desc_stats.loc["max"]
print("\n=======================\n")
print("Descriptive Statistics (w/o outliers)\n")
print(desc_stats)

# Manually calculate and transform
TEMP = (X_TEST["MORTDUE"] - min_val) / (max_val - min_val)
X_TEST = X_TEST.assign(calc_MORTDUE = TEMP.values) # Add calc value
print("\n=======================\n")
print("MIN/MAX Manual Calculation Comparison (w/o outliers)\n")
print(X_TEST.head())

"""
STANDARD SCALER - Automatic
"""
# Create & trainer scaler
theScaler = StandardScaler()
theScaler.fit(X) # Train scaler

# Transform data
Y_STD = theScaler.transform(X)

# Create dataframe
Y_STD = pd.DataFrame(Y_STD)
print("\n=======================\n")
print("STANDARDIZED NORMALIZED DATA (w/outliers)\n")
print(Y_STD.head())

# Edit variable names to indicated transformed
varNames_std = [] # Initialize empty list
for i in varNames:
    newName = "std_" + i
    varNames_std.append(newName)
print("\n=======================\n")
print("STANDARDIZED Edited Variable Names (w/o outliers)\n")
print(varNames_std)

# Append new column names
Y_STD.columns = varNames_std

# Append categorical variables
Y_STD[objList] = df[objList]
print("\n=======================\n")
print("Appending Categorical Variables (w/o outliers)\n")
print(Y_STD.head().T)

# Append TARGET variables
Y_STD[TARGET_F] = df.TARGET_BAD_FLAG
Y_STD[TARGET_A] = df.TARGET_LOSS_AMT
print("\n=======================\n")
print("STANDARDIZED Final Dataset (w/o outliers)\n")
print(Y_STD.head().T)

# Combine dataframes
Y_NEW = pd.concat([X, Y_STD], axis=1)
print("\n=======================\n")
print("STANDARDIZED Combined dataset (w/outliers)\n")
print(Y_NEW.head().T)

"""
STANDRD SCALER - Manual (w/o outliers)
"""
# Isolate 1 numeric variable
Y_TEST = Y_NEW[["MORTDUE", "std_MORTDUE"]]
print("\n=======================\n")
print("MORTDUE Variable (w/o outliers)\n")
print(Y_TEST.head())

# Determine min/max
desc_stats = Y_TEST["MORTDUE"].describe()
min_val = desc_stats.loc["min"]
max_val = desc_stats.loc["max"]
print("\n=======================\n")
print("Descriptive Statistics (w/o outliers)\n")
print(desc_stats)

# Manually calculate and transform
TEMP = (Y_TEST["MORTDUE"] - min_val) / (max_val - min_val)
Y_TEST = Y_TEST.assign(calc_MORTDUE = TEMP.values) # Add calc value
print("\n=======================\n")
print("STANDARDIZED Manual Calculation Comparison (w/o outliers)\n")
print(Y_TEST.head())

"""
KMeans Clustering Assignment
"""

"""
Set-up enviornment
"""
# Import libraries
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

# Modify output options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

sns.set_theme(style="whitegrid")

# Import data
FILE = "HMEQ_Loss.csv"

df = pd.read_csv(FILE)

"""
Set up Data
"""
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

"""
Fill in numeric missing with median
"""
for i in numList:
    if df[i].isna().sum() == 0:
        continue
    IMP = "IMP_" + i
    df[IMP] = df[i]
    df.loc[df[IMP].isna(), IMP] = df[i].median()
    df = df.drop(i, axis=1)

"""
Standardize data
"""
# Create copy
X = df.copy()
X = X.drop(objList, axis=1)
X = X.drop([TARGET_F, TARGET_A], axis=1)
X = X[["LOAN", "IMP_MORTDUE", "IMP_VALUE"]] # Variable selection
varNames = X.columns

# Create standard scaler
Scaler = StandardScaler()
Scaler.fit(X) # Train scaler

# Transform data
X_STD = Scaler.transform(X)
X_STD = pd.DataFrame(X_STD) # Create dataframe

"""
Set-up new dataframe
"""
# Initialize empty list
varNames_trn = []

# Add variables names back
for i in varNames:
    newName = "trn_" + i
    varNames_trn.append(newName)

X_STD.columns = varNames_trn

# Print output
print("Standardized Data:")
print(X_STD.head())

"""
KMeans Clustering
"""
# Initialize empty lists
K_LIST = []
I_LIST = []
S_LIST = []
C_LIST = []

# Develop clusters
for K in range(3,12):
    km = KMeans(n_clusters=K, random_state=1)
    km.fit(X_STD)
    K_LIST.append(K)
    I_LIST.append(km.inertia_)
    S_LIST.append(silhouette_score(X_STD, km.labels_))
    C_LIST.append(calinski_harabasz_score(X_STD, km.labels_))

# Define function for plots
def drawPlot(K, SCORE, LABEL):
    plt.plot(K, SCORE, "ro-", linewidth=2)
    plt.title(LABEL)
    plt.xlabel("Clusters")
    plt.ylabel("Score")
    plt.savefig(f"{LABEL}_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

# Make plots
drawPlot(K_LIST, I_LIST, "Inertia")
drawPlot(K_LIST, S_LIST, "Silhouette")
drawPlot(K_LIST, C_LIST, "Calinski")

"""
Determine and Predicting Clusters
"""
# Define user function to predict target variable
def clusterData(DATA, TRN_DATA, K, TARGET):
    print("\n")
    print("K = ", K)
    print("========")
    km = KMeans(n_clusters=K, random_state=1) # Make kmeans
    km.fit(TRN_DATA) # Fit data
    Y = km.predict(TRN_DATA) # Predict data
    DATA["CLUSTER"] = Y
    print(DATA.head())

    G = DATA.groupby("CLUSTER") # Group data
    print(G.mean(numeric_only=True))
    print("\n")
    print(G[TARGET].value_counts()) # Get clustering of targets

# Print clusters
clusterData(df, X_STD, 3, TARGET_F)
clusterData(df, X_STD, 4, TARGET_F)
clusterData(df, X_STD, 5, TARGET_F)

# Print clusters
clusterData(df, X_STD, 3, TARGET_A)
clusterData(df, X_STD, 4, TARGET_A)
clusterData(df, X_STD, 5, TARGET_A)

"""
KMeans Clustering w/PCA
"""
# Create copy
Z = df.copy()
Z = Z.drop(objList, axis=1)
Z = Z.drop([TARGET_F, TARGET_A], axis=1)
varNames = Z.columns

# Create standard scaler
Scaler = StandardScaler()
Scaler.fit(Z) # Train scaler

# Transform data
Z_STD = Scaler.transform(Z)
Z_STD = pd.DataFrame(Z_STD) # Create dataframe

# Create PCA
pca = PCA()
pca.fit(Z_STD) # Train data

# Transform data
Z_STD = pca.transform(Z_STD)
Z_STD = pd.DataFrame(Z_STD) # Create dataframe

"""
Set-up new PCA dataframe
"""
# Initialize empty list
varNames_trn = []

# Add variables names back
for i in range(Z_STD.shape[1]):
    index = str(i+1)
    newName = "PC_" + index
    varNames_trn.append(newName)

Z_STD.columns = varNames_trn

# Print output
print("Principal Component Standardized data:")
print(Z_STD.head(n=10))

"""
KMeans Clustering
"""
# Initialize empty lists
K_LIST = []
I_LIST = []
S_LIST = []
C_LIST = []

# Develop clusters
for K in range(3,12):
    km = KMeans(n_clusters=K, random_state=1)
    km.fit(Z_STD)
    K_LIST.append(K)
    I_LIST.append(km.inertia_)
    S_LIST.append(silhouette_score(X_STD, km.labels_))
    C_LIST.append(calinski_harabasz_score(X_STD, km.labels_))

# Define function for plots
def drawPlot(K, SCORE, LABEL):
    plt.plot(K, SCORE, "ro-", linewidth=2)
    plt.title(LABEL)
    plt.xlabel("Clusters")
    plt.ylabel("Score")
    plt.savefig(f"{LABEL}_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

# Make plots
drawPlot(K_LIST, I_LIST, "Inertia_PCA")
drawPlot(K_LIST, S_LIST, "Silhouette_PCA")
drawPlot(K_LIST, C_LIST, "Calinski_PCA")

"""
Determine and Predicting Clusters
"""
# Define user function to predict target variable
def clusterData(DATA, TRN_DATA, K, TARGET):
    print("\n")
    print("K = ", K)
    print("========")
    km = KMeans(n_clusters=K, random_state=1) # Make kmeans
    km.fit(TRN_DATA) # Fit data
    Y = km.predict(TRN_DATA) # Predict data
    DATA["CLUSTER"] = Y
    print(DATA.head())

    G = DATA.groupby("CLUSTER") # Group data
    print(G.mean(numeric_only=True))
    print("\n")
    print(G[TARGET].value_counts()) # Get clustering of targets

# Print clusters
clusterData(df, Z_STD, 3, TARGET_F)
clusterData(df, Z_STD, 4, TARGET_F)
clusterData(df, Z_STD, 5, TARGET_F)

# Print clusters
clusterData(df, Z_STD, 3, TARGET_A)
clusterData(df, Z_STD, 4, TARGET_A)
clusterData(df, Z_STD, 5, TARGET_A)

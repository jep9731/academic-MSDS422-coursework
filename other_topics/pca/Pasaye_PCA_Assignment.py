"""
PCA Assignment
"""

"""
Set-up enviornment
"""
# Import libraries
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
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
varNames = X.columns

# Create standard scaler
Scaler = StandardScaler()
Scaler.fit(X) # Train scaler

# Transform data
X_STD = Scaler.transform(X)

"""
PCA
"""
# Get PCs
MAX_N = X_STD.shape[1]

# Create PCA
pca = PCA(n_components=MAX_N)
pca.fit(X_STD)

# Get eignevalues
ev = pca.explained_variance_
print("Egien Values")
print(ev)
print("\n")

# Determine variations explained
VarPCT = []
TotPCT = []
total = 0
for i in ev:
    total += 1
    VAR = int(i / len(ev) * 100)
    PCT = int(total / len(ev) * 100)
    VarPCT.append(VAR)
    TotPCT.append(PCT)
    print(round(i, 2), "variation=", VAR,"%", " total=", PCT, "%")
print("\n")

# Print SCREE Plots
PC_NUM = np.arange(MAX_N) + 1
plt.plot(PC_NUM , ev, "ro-", linewidth=2)
plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue")
plt.savefig(f"Scree_plot_Eigen.png", dpi=300, bbox_inches="tight")
plt.show()

plt.plot(PC_NUM , VarPCT, "ro-", linewidth=2)
plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.savefig(f"Scree_plot_VarExp.png", dpi=300, bbox_inches="tight")
plt.show()


plt.plot(PC_NUM , TotPCT, "ro-", linewidth=2)
plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Total Variance Explained")
plt.savefig(f"Scree_plot_TotVarExp.png", dpi=300, bbox_inches="tight")
plt.show()

# Create dataframe for principal components
dfc = pd.DataFrame(pca.components_)
dfc.columns = varNames
print(dfc)
print("\n")

"""
Re-add columns to dataframe
"""
X_PCA = pca.transform(X_STD)
X_PCA = pd.DataFrame(X_PCA)
X_PCA = X_PCA.iloc[:, 0:2]

colNames = X_PCA.columns
pcaNames = []
for i in colNames:
    index = int(i) + 1
    Name = "PC_" + str(index)
    pcaNames.append(Name)

# Append column names
X_PCA.columns = pcaNames
print("Principal Component dataframe\n")
print(X_PCA.head(n=10))
print("\n")

# Print first eigenvalue
print("Principal Component 1")
print(X_PCA["PC_1"])
print("\n")

# Add categorical variables
X_PCA[objList] = df[objList]
print("Appending Categorical Variables\n")
print(X_PCA.head())
print("\n")

# Add target variables
X_PCA[TARGET_F] = df.TARGET_BAD_FLAG
X_PCA[TARGET_A] = df.TARGET_LOSS_AMT
print("PCA Final dataset\n")
print(X_PCA.head())
print("\n")

"""
Group analysis
"""
# Group by TARGET_F
for Name, Group in X_PCA.groupby(TARGET_F):
    label = "default" if Name == 1 else "non-default"
    print(f"{label}")
    print(Group.head())
    print("\n")


# Create scatter plot
for Name, Group in X_PCA.groupby(TARGET_F):
    plt.scatter(Group.PC_1, Group.PC_2)
plt.xlabel("PC_1")
plt.ylabel("PC_2")
plt.title("Scatter Plot With First Two Principal Components")
plt.legend(["Non-Default", "Default"])
plt.savefig(f"Scatter_plot.png", dpi=300, bbox_inches="tight")
plt.show()

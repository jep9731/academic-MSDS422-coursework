"""
Assignment 04: Neural Netorks
"""

"""
Import libraries & Set-up Enviornment
"""
# Import libraries
import math
import pandas as pd
import numpy as np
from operator import itemgetter

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from sklearn import tree
from sklearn.tree import _tree

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

import tensorflow as tf
tf.keras.utils.disable_interactive_logging()

from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

# Set up enviornment

sns.set(style="whitegrid")
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

"""
Set-up Data
"""
INFILE = "HMEQ_Loss.csv"

# Read file
df = pd.read_csv(INFILE)

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
Fill in missing with category 'MISSING'
"""
for i in objList:
    if df[i].isna().sum() == 0:
        continue
    NAME = "IMP_" + i
    df[NAME] = df[i]
    df[NAME] = df[NAME].fillna("MISSING")
    g = df.groupby(NAME)
    df = df.drop(i, axis=1)

# Convert all categorical variables into numeric
dt = df.dtypes
objList = []
for i in dt.index:
    if i in ([TARGET_F, TARGET_A]):
        continue
    if dt[i] in (["object"]):
        objList.append(i)

for i in objList:
    thePrefix = "z_" + i
    y = pd.get_dummies(df[i], prefix=thePrefix, drop_first=True, dtype=int)
    df = pd.concat([df, y], axis=1)

"""
Fill in numeric missing with median
"""
for i in numList:
    if df[i].isna().sum() == 0:
        continue
    FLAG = "M_" + i
    IMP = "IMP_" + i
    df[FLAG] = df[i].isna() + 0
    df[IMP] = df[i]
    df.loc[df[IMP].isna(), IMP] = df[i].median()
    df = df.drop(i, axis=1)

"""
Remove outliers
"""
dt = df.dtypes
numList = [] # Initialize empty list
for i in dt.index:
    if i in ([TARGET_F, TARGET_A]):
        continue
    if dt[i] in (["float64", "int64"]):
        numList.append(i)

# Find cutoff & remove outliers
for i in numList:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 - 1.5*IQR
    if IQR == 0:
        continue
    FLAG = "O_" + i # Create flag
    TRUNC = "TRUNC_" + i # Update column
    df[FLAG] = ((df[i] < lower) | (df[i] > upper)) + 0 # Add numeric flag
    df[TRUNC] = df[i].clip(lower=lower, upper=upper) # Update values
    df = df.drop(i, axis=1) # Remove original column names

for i in objList:
    df = df.drop( i, axis=1 )

"""
Split data
"""
# Create copy of dataframe
X = df.copy()

# Remove target variables
X = X.drop([TARGET_F, TARGET_A], axis=1)

# Create separate dataframe with only targets
Y = df[[TARGET_F, TARGET_A]]

# Split dataframes
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8,
                                                    test_size=0.2, random_state=2)

# Split by continuous target variables
F = ~ Y_train[TARGET_A].isna()
W_train = X_train[F].copy()
Z_train = Y_train[F].copy()

F = ~ Y_test[TARGET_A].isna()
W_test = X_test[F].copy()
Z_test = Y_test[F].copy()

"""
Remove outliers from target variables
"""
# Train data
Q1 = Z_train[TARGET_A].quantile(0.25)
Q3 = Z_train[TARGET_A].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

outlier_mask =(Z_train[TARGET_A] > lower) & (Z_train[TARGET_A] < upper)

Z_train = Z_train.loc[outlier_mask]
W_train = W_train.loc[outlier_mask]

# Test data
Q1 = Z_test[TARGET_A].quantile(0.25)
Q3 = Z_test[TARGET_A].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

upper_array = np.where(Z_test[TARGET_A] >= upper)[0]
lower_array = np.where(Z_test[TARGET_A] <= lower)[0]

outlier_mask = (Z_test[TARGET_A] > lower) & (Z_test[TARGET_A] < upper)

Z_test = Z_test.loc[outlier_mask]
W_test = W_test.loc[outlier_mask]

"""
Model accuracy metrics
"""
# Probability score function
def getProbAccuracyScores(NAME, MODEL, X, Y):
    pred = MODEL.predict(X)
    probs = MODEL.predict_proba(X)
    acc_score = metrics.accuracy_score(Y, pred)
    p1 = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(Y, p1)
    auc = metrics.auc(fpr,tpr)
    return [NAME, acc_score, fpr, tpr, auc]

# ROC curve
def print_ROC_Curve(TITLE, LIST):
    fig = plt.figure(figsize=(6,4))
    plt.title(TITLE)
    for theResults in LIST:
        NAME = theResults[0]
        fpr = theResults[2]
        tpr = theResults[3]
        auc = theResults[4]
        theLabel = "AUC " + NAME + f" {auc:.2f}"
        plt.plot(fpr, tpr, label = theLabel )
    plt.legend(loc = "lower right")
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(f"{TITLE}_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

# Classification accuracy
def print_Accuracy(TITLE, LIST):
    print(TITLE)
    print("======")
    for theResults in LIST:
        NAME = theResults[0]
        ACC = theResults[1]
        print(NAME, " = ", ACC)
    print("------\n")

# Regression accuracy
def getAmtAccuracyScores(NAME, MODEL, X, Y):
    pred = MODEL.predict(X)
    MEAN = Y.mean()
    RMSE = math.sqrt(metrics.mean_squared_error(Y, pred))
    return [NAME, RMSE, MEAN]

"""
Decision Tree
"""
#Get tree variables
def getTreeVars(TREE, varNames):
    tree_ = TREE.tree_
    varName = [varNames[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

    nameSet = set()
    for i in tree_.feature:
        if i != _tree.TREE_UNDEFINED:
            nameSet.add(i)
    nameList = list(nameSet)
    parameter_list = list()
    for i in nameList :
        parameter_list.append(varNames[i])
    return parameter_list

# Default Loan Probability
WHO = "TREE"
depth = 4

# Create decision tree classifier
FLG = tree.DecisionTreeClassifier(max_depth=depth)
FLG = FLG.fit(X_train, Y_train[TARGET_F])

TRAIN_FLG = getProbAccuracyScores(WHO + "_Train", FLG, X_train, Y_train[TARGET_F])
TEST_FLG = getProbAccuracyScores(WHO + "_Test", FLG, X_test, Y_test[TARGET_F])

# Print metrics
print_ROC_Curve(WHO, [TRAIN_FLG, TEST_FLG])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_FLG, TEST_FLG])

# Get variables
feature_cols = list(X.columns.values)
vars_tree_flag = getTreeVars(FLG, feature_cols)
tree.export_graphviz(FLG, out_file="tree_f.txt", filled=True, rounded=True,
                     feature_names=feature_cols, impurity=False, class_names=["Good", "Bad"])

# Loss Amount Porbability
WHO = "TREE"
depth = 4

# Create decision tree regressor
AMT = tree.DecisionTreeRegressor(max_depth=depth)
AMT = AMT.fit(W_train, Z_train[TARGET_A])

TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO + "_Test", AMT, W_test, Z_test[TARGET_A])

# Print metrics
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Get variables
feature_cols = list(X.columns.values)
vars_tree_amt = getTreeVars(AMT, feature_cols)
tree.export_graphviz(AMT, out_file="tree_a.txt", filled=True, rounded=True,
                     feature_names=feature_cols, impurity=False, precision=0)

# Copy of metrics
TREE_FLG = TEST_FLG.copy()
TREE_AMT = TEST_AMT.copy()

"""
Random Forest
"""
# Get ensemble tree variables
def getEnsembleTreeVars(ENSTREE, varNames):
    importance = ENSTREE.feature_importances_
    index = np.argsort(importance)
    theList = []
    for i in index:
        imp_val = importance[i]
        if imp_val > np.average(ENSTREE.feature_importances_):
            v = int(imp_val / np.max(ENSTREE.feature_importances_) * 100)
            theList.append((varNames[i], v))
    theList = sorted(theList, key=itemgetter(1), reverse=True)
    return theList

# Default Loan Probability
WHO = "RF"
n_estimator = 25

# Create RF classifier
FLG = RandomForestClassifier(n_estimators=n_estimator, random_state=1)
FLG = FLG.fit(X_train, Y_train[TARGET_F])

TRAIN_FLG = getProbAccuracyScores(WHO + "_Train", FLG, X_train, Y_train[TARGET_F])
TEST_FLG = getProbAccuracyScores(WHO + "_Test", FLG, X_test, Y_test[TARGET_F])

# Print metrics
print_ROC_Curve(WHO, [TRAIN_FLG, TEST_FLG])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_FLG, TEST_FLG])

# Get variables
feature_cols = list(X.columns.values)
vars_RF_flag = getEnsembleTreeVars(FLG, feature_cols)

# Default Loss Probability
WHO = "RF"
n_estimator = 100

# Create RF Regressor
AMT = RandomForestRegressor(n_estimators=n_estimator, random_state=1)
AMT = AMT.fit(W_train, Z_train[TARGET_A])

TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO + "_Test", AMT, W_test, Z_test[TARGET_A])

# Print metrics
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Get variables
feature_cols = list(X.columns.values)
vars_RF_amt = getEnsembleTreeVars(AMT, feature_cols)

# Copy metrics
RF_FLG = TEST_FLG.copy()
RF_AMT = TEST_AMT.copy()

"""
Gradient Boosting
"""
# Default Loan Probability
WHO = "GB"

# Create RG classifier
FLG = GradientBoostingClassifier(random_state=1)
FLG = FLG.fit(X_train, Y_train[TARGET_F])

TRAIN_FLG = getProbAccuracyScores(WHO + "_Train", FLG, X_train, Y_train[TARGET_F])
TEST_FLG = getProbAccuracyScores(WHO + "_Test", FLG, X_test, Y_test[TARGET_F])

# Print metrics
print_ROC_Curve(WHO, [TRAIN_FLG, TEST_FLG])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_FLG, TEST_FLG])

# Get variables
feature_cols = list(X.columns.values)
vars_GB_flag = getEnsembleTreeVars(FLG, feature_cols)

# Default Loss Probability
WHO = "GB"

# Create GB Regressor
AMT = GradientBoostingRegressor(random_state=1)
AMT.fit(W_train, Z_train[TARGET_A])

TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO + "_Test", AMT, W_test, Z_test[TARGET_A])

# Print metrics
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Get GB variables
feature_cols = list(X.columns.values)
vars_GB_amt = getEnsembleTreeVars(AMT, feature_cols)

# Copy metrics
GB_FLG = TEST_FLG.copy()
GB_AMT = TEST_AMT.copy()

"""
Regression Coefficient Functions
"""
# Logistic
def getCoefLogit(MODEL, TRAIN_DATA):
    varNames = list(TRAIN_DATA.columns.values)
    coef_dict = {}
    coef_dict["INTERCEPT"] = MODEL.intercept_[0]
    for coef, feat in zip(MODEL.coef_[0],varNames):
        coef_dict[feat] = coef
    print(f"{WHO} DEFAULT LOAN")
    print("---------")
    print("Total Variables: ", len(coef_dict))
    for i in coef_dict:
        print(i, " = ", coef_dict[i])
    print("\n")

# Linear
def getCoefLinear(MODEL, TRAIN_DATA):
    varNames = list(TRAIN_DATA.columns.values)
    coef_dict = {}
    coef_dict["INTERCEPT"] = MODEL.intercept_
    for coef, feat in zip(MODEL.coef_, varNames):
        coef_dict[feat] = coef
    print(f"{WHO} DEFAULT LOSS")
    print("--------")
    print("Total Variables: ", len(coef_dict))
    for i in coef_dict:
        print(i, " = ", coef_dict[i])
    print("\n")

"""
Regression - all variables
"""
# Create logistic regression
WHO = "REG_ALL"
solver = "newton-cg"
max_iter = 1000

# Default Loan Probability
FLG = LogisticRegression(solver=solver, max_iter=max_iter)
FLG = FLG.fit(X_train, Y_train[TARGET_F])

TRAIN_FLG = getProbAccuracyScores(WHO + "_Train", FLG, X_train, Y_train[TARGET_F])
TEST_FLG = getProbAccuracyScores(WHO + "_Test", FLG, X_test, Y_test[TARGET_F])

# Print metrics
print_ROC_Curve(WHO, [TRAIN_FLG, TEST_FLG])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_FLG, TEST_FLG])

# Create Linear Regression
AMT = LinearRegression()
AMT = AMT.fit(W_train, Z_train[TARGET_A])

# Default Loss Probability
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO + "_Test", AMT, W_test, Z_test[TARGET_A])

# Print metrics
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Get variables names
varNames =  list(X_train.columns.values)

# Coefficients
REG_ALL_FLG_COEF = getCoefLogit(FLG, X_train )
REG_ALL_AMT_COEF = getCoefLinear(AMT, X_train )

# Copy metrics
REG_ALL_FLG = TEST_FLG.copy()
REG_ALL_AMT = TEST_AMT.copy()

"""
Regression - Decision Tree
"""
# Create logistic regression
WHO = "REG_TREE"
solver = "newton-cg"
max_iter = 1000

# Default Loan Probability
FLG = LogisticRegression(solver=solver, max_iter=max_iter)
FLG = FLG.fit(X_train[vars_tree_flag], Y_train[TARGET_F])

TRAIN_FLG = getProbAccuracyScores(WHO + "_Train", FLG, X_train[vars_tree_flag], Y_train[TARGET_F])
TEST_FLG = getProbAccuracyScores(WHO + "_Test", FLG, X_test[vars_tree_flag], Y_test[TARGET_F])

# Print metrics
print_ROC_Curve(WHO, [TRAIN_FLG, TEST_FLG])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_FLG, TEST_FLG])

# Create linear regression
AMT = LinearRegression()
AMT = AMT.fit(W_train[vars_tree_amt], Z_train[TARGET_A])

# Default Loss Probability
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train[vars_tree_amt], Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO + "_Test", AMT, W_test[vars_tree_amt], Z_test[TARGET_A])

# Print metrics
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Get variables names
varNames = list(X_train.columns.values)

# Coefficients
REG_TREE_FLG_COEF = getCoefLogit(FLG, X_train[vars_tree_flag])
REG_TREE_AMT_COEF = getCoefLinear(AMT, X_train[vars_tree_amt])

# Copy metrics
REG_TREE_FLG = TEST_FLG.copy()
REG_TREE_AMT = TEST_AMT.copy()

"""
Regression - Random Forest
"""
# Create logistic regression
WHO = "REG_RF"
solver = "newton-cg"
max_iter = 1000

# Get RF variables
RF_flag = [] # initialize empty list
for i in vars_RF_flag:
    theVar = i[0]
    RF_flag.append(theVar)

RF_amt = [] # initialize empty list
for i in vars_RF_amt:
    theVar = i[0]
    RF_amt.append(theVar)

# Default Loan Probability
FLG = LogisticRegression(solver=solver, max_iter=max_iter)
FLG = FLG.fit(X_train[RF_flag], Y_train[TARGET_F])

TRAIN_FLG = getProbAccuracyScores(WHO + "_Train", FLG, X_train[RF_flag], Y_train[TARGET_F])
TEST_FLG = getProbAccuracyScores(WHO + "_Test", FLG, X_test[RF_flag], Y_test[TARGET_F])

# Print metrics
print_ROC_Curve(WHO, [TRAIN_FLG, TEST_FLG])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_FLG, TEST_FLG])

# Create linear regression
AMT = LinearRegression()
AMT = AMT.fit(W_train[RF_amt], Z_train[TARGET_A])

# Default Loss Probability
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train[RF_amt], Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO + "_Test", AMT, W_test[RF_amt], Z_test[TARGET_A])

# Print metrics
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Coefficients
REG_RF_FLG_COEF = getCoefLogit(FLG, X_train[RF_flag])
REG_RF_AMT_COEF = getCoefLinear(AMT, X_train[RF_amt])

# Copy metrics
REG_RF_FLG = TEST_FLG.copy()
REG_RF_AMT = TEST_AMT.copy()

"""
Regression - Gradient Boosting
"""
# Create logistic regression
WHO = "REG_GB"
solver = "newton-cg"
max_iter = 1000

# Get GB variables
GB_flag = [] # Initialize empty list
for i in vars_GB_flag:
    theVar=i[0]
    GB_flag.append(theVar)

GB_amt = [] # Initialize empty list
for i in vars_GB_amt:
    theVar=i[0]
    GB_amt.append(theVar)

# Default Loan Probability
FLG = LogisticRegression(solver=solver, max_iter=max_iter)
FLG = FLG.fit(X_train[GB_flag], Y_train[TARGET_F])

TRAIN_FLG = getProbAccuracyScores(WHO + "_Train", FLG, X_train[GB_flag], Y_train[TARGET_F])
TEST_FLG = getProbAccuracyScores(WHO + "_Test", FLG, X_test[GB_flag], Y_test[TARGET_F])

# Print metrics
print_ROC_Curve(WHO, [TRAIN_FLG, TEST_FLG])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_FLG, TEST_FLG])

# Create linear regression
AMT = LinearRegression()
AMT = AMT.fit(W_train[GB_amt], Z_train[TARGET_A])

# Default Loss Probability
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train[GB_amt], Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO + "_Test", AMT, W_test[GB_amt], Z_test[TARGET_A])

# Print metrics
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Coefficients
REG_GB_FLG_COEF = getCoefLogit(FLG, X_train[GB_flag])
REG_GB_AMT_COEF = getCoefLinear(AMT, X_train[GB_amt])

# Copy metrics
REG_GB_FLG = TEST_FLG.copy()
REG_GB_AMT = TEST_AMT.copy()

"""
Regression Stepwise
"""
# Logistical Regression
U_train = X_train[vars_tree_flag]
stepVarNames = list(U_train.columns.values)
maxCols = U_train.shape[1]

sfs = SFS(LogisticRegression(solver='newton-cg', max_iter=100),
          k_features=(1, maxCols),
          forward=True,
          floating=False,
          cv=3
          )
sfs.fit(U_train.values, Y_train[TARGET_F].values)

theFigure = plot_sfs(sfs.get_metric_dict(), kind=None)
plt.title('DEFAULT LOAN Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()

dfm = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
dfm = dfm[['feature_names', 'avg_score']]
dfm.avg_score = dfm.avg_score.astype(float)

print(" ................... ")
maxIndex = dfm.avg_score.argmax()
print("argmax")
print(dfm.iloc[maxIndex, ])
print(" ................... ")

stepVars = dfm.iloc[maxIndex, ]
stepVars = stepVars.feature_names
print(stepVars)

finalStepVars = []
for i in stepVars:
    index = int(i)
    try:
        theName = stepVarNames[index]
        finalStepVars.append(theName)
    except:
        pass

for i in finalStepVars:
    print(i)

U_train = X_train[finalStepVars]
U_test = X_test[finalStepVars]

# Linear Regression
V_train = W_train[GB_amt]
stepVarNames = list(V_train.columns.values)
maxCols = V_train.shape[1]

sfs = SFS(LinearRegression(),
          k_features=(1, maxCols),
          forward=True,
          floating=False,
          scoring = 'r2',
          cv=5
          )
sfs.fit(V_train.values, Z_train[TARGET_A].values)

theFigure = plot_sfs(sfs.get_metric_dict(), kind=None )
plt.title('LOSS Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()

dfm = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
dfm = dfm[['feature_names', 'avg_score']]
dfm.avg_score = dfm.avg_score.astype(float)

print(" ................... ")
maxIndex = dfm.avg_score.argmax()
print("argmax")
print( dfm.iloc[ maxIndex, ] )
print(" ................... ")

stepVars = dfm.iloc[ maxIndex,]
stepVars = stepVars.feature_names
print(stepVars)

finalStepVars = []
for i in stepVars:
    index = int(i)
    try:
        theName = stepVarNames[index]
        finalStepVars.append(theName)
    except:
        pass

for i in finalStepVars:
    print(i)

V_train = W_train[finalStepVars]
V_test = W_test[finalStepVars]

"""
Regression
"""
# Create logistical regression
WHO = "REG_STEPWISE"

FLG = LogisticRegression(solver='newton-cg', max_iter=1000 )
FLG = FLG.fit(U_train, Y_train[ TARGET_F ])

TRAIN_FLG = getProbAccuracyScores(WHO + "_Train", FLG, U_train, Y_train[ TARGET_F])
TEST_FLG = getProbAccuracyScores(WHO + "_Test", FLG, U_test, Y_test[TARGET_F])

# Create linear regression
AMT = LinearRegression()
AMT = AMT.fit(V_train, Z_train[TARGET_A])

TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, V_train, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO + "_Test", AMT, V_test, Z_test[TARGET_A])

# Get coefficientes
REG_STEP_FLG_COEF = getCoefLogit(FLG, U_train)
REG_STEP_AMT_COEF = getCoefLinear(AMT, V_train)

# Copy metrics
REG_STEP_FLG = TEST_FLG.copy()
REG_STEP_AMT = TEST_AMT.copy()

"""
TENSORFLOW - Loan Default
"""
# Define metrics function
def get_TF_ProbAccuracyScores(NAME, MODEL, X, Y):
    probs = MODEL.predict(X)
    pred_list = []
    for p in probs:
        pred_list.append(np.argmax(p))
    pred = np.array(pred_list)
    acc_score = metrics.accuracy_score(Y, pred)
    p1 = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(Y, p1)
    auc = metrics.auc(fpr,tpr)
    return [NAME, acc_score, fpr, tpr, auc]

# Transform data
theScaler = MinMaxScaler()
theScaler.fit(X_train)

T_train = theScaler.transform(X_train) 
T_test = theScaler.transform(X_test)

# Create dataframe
T_train = pd.DataFrame(T_train)
T_test = pd.DataFrame(T_test)

T_train.columns = list(X_train.columns.values)
T_test.columns = list(X_train.columns.values)

# Determine variables
T_train = T_train[GB_flag]
T_test = T_test[GB_flag]

# Determine model parameters
WHO = "TF"
F_ShapeSize = T_train.shape[1]
F_Activation = tf.keras.activations.relu
F_lossMetric = tf.keras.losses.SparseCategoricalCrossentropy()
F_Optimizer = tf.keras.optimizers.Adam()
F_Epochs = 100
F_Units = int(2*F_ShapeSize/3)

# Create TF model
F_LAYER_01 = tf.keras.layers.Dense(units=F_Units, activation=F_Activation, input_dim=F_ShapeSize)
F_LAYER_DROP = tf.keras.layers.Dropout(0.2)
F_LAYER_02 = tf.keras.layers.Dense(units=F_Units, activation=F_Activation)
F_LAYER_OUTPUT = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)

FLG = tf.keras.Sequential()
FLG.add(F_LAYER_01)
FLG.add(F_LAYER_02)
FLG.add(F_LAYER_OUTPUT)
FLG.compile(loss=F_lossMetric, optimizer=F_Optimizer)
FLG.fit(T_train, Y_train[TARGET_F], epochs=F_Epochs, verbose=0)

TRAIN_FLG = get_TF_ProbAccuracyScores(WHO + "_Train", FLG, T_train, Y_train[TARGET_F])
TEST_FLG = get_TF_ProbAccuracyScores(WHO + "_Test", FLG, T_test, Y_test[TARGET_F])

# Print accuracy metrics
print_ROC_Curve(WHO, [TRAIN_FLG, TEST_FLG]) 
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_FLG, TEST_FLG])

"""
TENSORFLOW - Loss Amount
"""
# Transform data
F_train = theScaler.transform(W_train)
F_test = theScaler.transform(W_test)

# Create dataframe
F_train = pd.DataFrame(F_train)
F_test = pd.DataFrame(F_test)

F_train.columns = list(X_train.columns.values)
F_test.columns = list(X_train.columns.values)

# Determine variables
F_train = F_train[GB_amt]
F_test = F_test[GB_amt]

# Determine model parameters
A_ShapeSize = F_train.shape[1]
A_Activation = tf.keras.activations.relu
A_lossMetric = tf.keras.losses.MeanSquaredError()
A_Optimizer = tf.keras.optimizers.Adam()
A_Epochs = 100
A_Units = int(2*A_ShapeSize/3)

# Create TF model
A_LAYER_01 = tf.keras.layers.Dense(units=A_Units, activation=A_Activation, input_dim=A_ShapeSize)
A_LAYER_DROP = tf.keras.layers.Dropout(0.2)
A_LAYER_02 = tf.keras.layers.Dense(units=A_Units, activation=A_Activation)
A_LAYER_OUTPUT = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear)

AMT = tf.keras.Sequential()
AMT.add(A_LAYER_01)
AMT.add(A_LAYER_02)
AMT.add(A_LAYER_OUTPUT)
AMT.compile(loss=A_lossMetric, optimizer=A_Optimizer)
AMT.fit(F_train, Z_train[TARGET_A], epochs=A_Epochs, verbose=0)

TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, F_train[GB_amt], Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO + "_Test", AMT, F_test[GB_amt], Z_test[TARGET_A])

# Print metrics
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Copy metrics
TF_FLG = TEST_FLG.copy()
TF_AMT = TEST_AMT.copy()

"""
Model Comparison
"""
WHO = "ALL MODELS"
# Default probabability
ALL_FLG = [TREE_FLG, RF_FLG, GB_FLG, REG_ALL_FLG, REG_TREE_FLG,
            REG_RF_FLG, REG_GB_FLG, REG_STEP_FLG, TF_FLG]

ALL_FLG = sorted(ALL_FLG, key = lambda x: x[4], reverse=True)
print_ROC_Curve(WHO, ALL_FLG) 

# Default accuracy
ALL_FLG = sorted(ALL_FLG, key = lambda x: x[1], reverse=True)
print_Accuracy("ALL CLASSIFICATION ACCURACY", ALL_FLG)

# Loss amount
ALL_AMT = [TREE_AMT, RF_AMT, GB_AMT, REG_ALL_AMT, REG_TREE_AMT,
            REG_RF_AMT, REG_GB_AMT, REG_STEP_AMT, TF_AMT]
ALL_AMT = sorted( ALL_AMT, key = lambda x: x[1])

# Loss accuracy
print_Accuracy("ALL LOSS MODEL ACCURACY", ALL_AMT)

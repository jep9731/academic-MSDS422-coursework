'''
Assignment 01: Data Preparation
'''

'''
Import libraries & program set-up
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

'''
Load file
'''
INFILE = "HMEQ_Loss.csv"

# Read file
df = pd.read_csv(INFILE)

# Explore database
print("----------------------------------- \nDataframe Preview:\n", df.head().T)
print("\n")
print("----------------------------------- \nShape:",df.shape)
print("\n")
print("----------------------------------- \nDataframe info:\n")
df.info()
print("\n")
print("----------------------------------- \nMissing values:\n", df.isna().sum())
print("\n")

'''
Set up target & input variables based on data dictionary
'''
# Target variables
TARGET_F = "TARGET_BAD_FLAG" # Flag variable
TARGET_A = "TARGET_LOSS_AMT" # Amount variable

# Numeric input variables
NUM_INPUT = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG",
             "DELINQ", "CLAGE", "NINQ","CLNO","DEBTINC"
             ]

# Categorical inputs
CAT_INPUT = [
    "REASON", "JOB"
    ]

'''
Pull data types
'''
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

print("OBJECTS")
print("-------")
for i in objList :
    print(i)
print("\n")

print("NUMBER")
print("-------")
for i in numList :
    print(i)
print("\n")

'''
Explore input and target variables - statistical
'''
# Summary statistics
stats_sum = round(df.describe().T, 2)
print("-----------------------------------\nSummary Statistics:\n",
      stats_sum
      )
print("\n")
stats_sum.to_csv("stas_sum.csv") # Export statistics summary

# Frequency distribution
job_df = df[objList].value_counts(subset=["JOB"], ascending=False)
print("-----------------------------------\nFrequency Distrubtion for JOB:\n",
      job_df
      )
print("\n")
job_df.to_csv("freq_dist_job.csv") # Export JOB frequence distribution

reason_df = df[objList].value_counts(subset=["REASON"],ascending=False)
print("-----------------------------------\nFrequency Distrubtion for REASON:\n",
      reason_df
      )
print("\n")
reason_df.to_csv("freq_dist_reason.csv") # Export REASON frequence distribution

# Correlation
full_corr_matrix = round(df.select_dtypes(include=["float64", "int64"]).corr(), 2)
print("-----------------------------------\nFull Correlation matrix:\n",
      full_corr_matrix
      )
print("\n")
full_corr_matrix.to_csv("full_corr_matrix.csv") #Export full correlation matrix

'''
Explore input and target variables - graphical
'''
# Target variables
plt.figure(figsize=(8, 6))
sns.countplot(x=TARGET_F, data=df)
plt.title(f"{TARGET_F} Distribution")
plt.xlabel(f"{TARGET_F}")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(x=TARGET_A, data=df, kde=True)
plt.title(f"{TARGET_A} Distribution")
plt.xlabel(f"{TARGET_A}")
plt.ylabel("Frequency")
plt.show()

# Numeric variables
df_NUM = df[numList]

fig, axes = plt.subplots(4, 3, figsize=(18, 18), constrained_layout=True)
axes = axes.flatten()  # flatten in case of multiple rows

for idx, i in enumerate(df_NUM.columns):
    sns.histplot(df_NUM[i], kde=True, ax=axes[idx])
    axes[idx].set_title(f"{i} Distribution")
    axes[idx].set_ylabel("Frequency")

# Remove any unused subplots
for j in range(idx+1, len(axes)):
    fig.delaxes(axes[j])

plt.show()

# Categical variables
df_CAT = df[objList]
for i in df_CAT:
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=i, data=df_CAT, hue = i, palette="Set2", legend=False)
    plt.title(f"{i} Distribution")
    plt.xlabel(i)
    plt.ylabel("Count")
    for p in ax.patches: # Add data labels
        ax.text(
            p.get_x() + p.get_width() / 2., # x-coordinate of the label
            p.get_height(), # y-coordinate of the label (top of the bar)
            "{:.0f}".format(p.get_height()), # The label text (count value)
            ha="center", # Horizontal alignment
            va="bottom" # Vertical alignment
            )
    plt.show()

'''
Explore relationships between input and target variables
'''

"""1. Undestanding which inputs are associated with a default loan"""
# Numeric inputs
default_loan_stats = round(df.groupby(TARGET_F)[NUM_INPUT].describe().T, 2)
print("-----------------------------------\nSummary Statistics:\n",
      default_loan_stats
      )
print("\n")
default_loan_stats.to_csv("default_loan_stats.csv") # Export default loan stats

# Boxplots for numeric variables
fig, axes = plt.subplots(4, 3, figsize=(18, 18), constrained_layout=True)
axes = axes.flatten()  # flatten in case of multiple rows

for idx, i in enumerate(df[NUM_INPUT].columns):
    sns.boxplot(x=TARGET_F, y=i, data=df, ax=axes[idx])
    sns.stripplot(x=TARGET_F, y=i, data=df, color=".25", jitter=True, size=3, ax=axes[idx])
    axes[idx].set_title(f"Boxplot: {i} vs {TARGET_F}")
    axes[idx].set_xlabel(f"Default ({TARGET_F})")
    axes[idx].set_ylabel(i)
    axes[idx].set_xticks(ticks=[0,1], labels=["No","Yes"])

# Remove any unused subplots
for j in range(idx+1, len(axes)):
    fig.delaxes(axes[j])

plt.show()

# Correlation matrix of defaulted loans
corr_matrix = df[NUM_INPUT + [TARGET_F]].corr()
target_corr = round(corr_matrix[TARGET_F].sort_values(ascending=False), 2)
print("-----------------------------------\nDefault Loan Correlation Matrix:\n",
      target_corr
      )
print("\n")
target_corr.to_csv("default_loan_corr_matrix.csv") # Export default loan corr matrix

# Heatmap of correlation matrix
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, center=0,
            fmt=".2f", vmax=1.0, vmin=-1.0, annot_kws={"size":10})
plt.title("Correlation Matrix Heatmap")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.show()

# Categorical Inputs
print("-----------------------------------\nCross Tab:\n")
for i in CAT_INPUT:
    ct = pd.crosstab(df[i], df[TARGET_F]) # Create cross tab for each CAT
    print(ct) # Print corss tab for each CAT
    filename = f"crosstab_{i}.csv" # Save cross tab for each CAT
    ct.to_csv(filename) # Export cross tab for each CAT
print("\n")

# Countplot for categorical variables
for i in CAT_INPUT:
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=i, hue=TARGET_F, palette="Dark2", data=df)
    plt.title(f"Countplot: {i} vs {TARGET_F}")
    plt.xlabel(i)
    plt.ylabel(f"Count")
    plt.legend(title=f"Default ({TARGET_F})", labels=["No", "Yes"])
    plt.show()

"""2. Determining factors that influence loss amount"""
# Filter dataframe to only contain default loans
df_loss = df[df[TARGET_F] == 1] 

# Numeric inputs
# Summary stats
amount_loss_stats = round(df_loss[NUM_INPUT].describe().T, 2)
print("-----------------------------------\nSummary Statistics of Defaulted Loan:\n",
      amount_loss_stats
      )
print("\n")
amount_loss_stats.to_csv("amount_loss_stats.csv") # Export amount loss stats

# Correlation with loss amount
corr_loss_amount = round(df_loss[NUM_INPUT + [TARGET_A]].corr()[TARGET_A]
                         .sort_values(ascending=False), 2)
print("-----------------------------------\nCorrelation with Loss Amount:\n",
      corr_loss_amount
      )
print("\n")
corr_loss_amount.to_csv("corr_loss_amount.csv")

# Regression plot of numeric variables
fig, axes = plt.subplots(4, 3, figsize=(18, 18), constrained_layout=True)
axes = axes.flatten()  # flatten in case of multiple rows

for idx, i in enumerate(df_loss[NUM_INPUT].columns):
    sns.regplot(x=i, y=TARGET_A, data=df_loss, ax=axes[idx])
    axes[idx].set_title(f"Regression plot: {i} vs {TARGET_A}")

# Remove any unused subplots
for j in range(idx+1, len(axes)):
    fig.delaxes(axes[j])

plt.show()

# Categorical Inputs
# Boxplots of categorical variables
for i in CAT_INPUT:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=i, y=TARGET_A, data=df_loss)
    plt.title(f"Boxplot: {i} vs {TARGET_A}")
    plt.show()

'''
Fix (impute) all missing data
'''
# Categorical - Impute with "Missing"
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
for i in dt.index :
    if i in ([TARGET_F, TARGET_A]):
        continue
    if i in (["object"]):
        objList.append(i)

for i in objList:
    thePrefix = "z_" + i
    y = pd.get_dummies(df[i], prefix=thePrefix, dummy_na=False, drop_first=True, dtype=int)
    df = pd.concat([df, y], axis=1)
    df = df.drop(i, axis=1)

print("-----------------------------------")
print(df.head(4).T)
print("\n")

# Numeric Data - Impute with Median
for i in numList:
    if df[i].isna().sum == 0:
        continue
    FLAG = "M_" + i
    IMP = "IMP_" + i
    df[FLAG] = df[i].isna() + 0
    df[IMP] = df[i]
    df.loc[df[IMP].isna(), IMP] = df[i].median()
    df = df.drop(i, axis=1)

print("-----------------------------------")
print(df.head(4).T)
print("\n")

'''
Final Dataset
'''
df_final = df.copy() # Create final dataframe
df_final.to_csv("HMEQ_Loss_final.csv", index=False) # Export final dataframe
print("-----------------------------------\nFinal shape:", df_final.shape)
print("\n")
print("-----------------------------------\nFinal dataset:\n", df_final.head(4).T)

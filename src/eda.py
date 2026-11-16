
# ==========================================
# LOAN APPROVAL PREDICTION USING MACHINE LEARNING
# MODULE 1: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visualization style
sns.set_style("whitegrid")

# Load dataset
df = pd.read_csv("data/train_u6lujuX_CVtuZ9i.csv")

print("\n========== LOAN APPROVAL DATASET ==========")

# 1. Display first five records
print("\n1. FIRST FIVE RECORDS")
print(df.head())

# 2. Dataset dimensions
print("\n2. DATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 3. Dataset information
print("\n3. DATASET INFORMATION")
df.info()

# 4. Statistical summary
print("\n4. STATISTICAL SUMMARY")
print(df.describe())

# 5. Missing values
print("\n5. MISSING VALUES")
print(df.isnull().sum())

# 6. Duplicate records
print("\n6. DUPLICATE RECORDS")
print(df.duplicated().sum())

# 7. Loan approval distribution
print("\n7. LOAN STATUS DISTRIBUTION")
print(df["Loan_Status"].value_counts())

# Create output folder
os.makedirs("reports/figures", exist_ok=True)

# 8. Loan approval distribution graph
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Loan_Status")
plt.title("Loan Approval Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")
plt.savefig("reports/figures/loan_distribution.png")
plt.show()

# 9. Applicant income distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["ApplicantIncome"], kde=True)
plt.title("Applicant Income Distribution")
plt.xlabel("Applicant Income")
plt.savefig("reports/figures/income_distribution.png")
plt.show()

# 10. Education vs Loan Status
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Education", hue="Loan_Status")
plt.title("Education vs Loan Approval")
plt.savefig("reports/figures/education_analysis.png")
plt.show()

# 11. Credit History vs Loan Status
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Credit_History", hue="Loan_Status")
plt.title("Credit History vs Loan Approval")
plt.savefig("reports/figures/credit_history_analysis.png")
plt.show()

# 12. Correlation heatmap
plt.figure(figsize=(10, 6))
numeric_data = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")
plt.savefig("reports/figures/correlation_heatmap.png")
plt.show()

print("\n==========================================")
print("EXPLORATORY DATA ANALYSIS COMPLETED!")
print("==========================================")

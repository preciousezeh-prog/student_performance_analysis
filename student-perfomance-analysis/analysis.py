# analysis.py
import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load the dataset
# -----------------------------
# Dataset is in the 'data' folder
file_path = "data/student_data.csv"

# Read CSV
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumns:", list(df.columns))

# -----------------------------
# 2. Basic cleaning
# -----------------------------
df = df.drop_duplicates()

# Fill missing values
for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        df[col] = df[col].fillna(df[col].median())
    else:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after cleaning:")
print(df.isna().sum())

# -----------------------------
# 3. Summary statistics
# -----------------------------
summary = df.describe(include="all")
print("\nSummary statistics:")
print(summary)

# Make sure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Save summary to file
summary.to_csv("outputs/summary_statistics.csv")

# -----------------------------
# 4. Plot numeric distributions
# -----------------------------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_cols:
    plt.figure()
    plt.hist(df[col], bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"outputs/{col}_distribution.png")
    plt.close()

# -----------------------------
# 5. Correlation plot
# -----------------------------
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation="nearest", cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha="right")
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("outputs/correlation_matrix.png")
    plt.close()

# -----------------------------
# 6. Simple insight section
# -----------------------------
print("\nTop insights:")
for col in numeric_cols:
    print(f"- {col}: mean = {df[col].mean():.2f}, median = {df[col].median():.2f}")

print("\nDone! Check the 'outputs' folder for all generated graphs and summary statistics.")
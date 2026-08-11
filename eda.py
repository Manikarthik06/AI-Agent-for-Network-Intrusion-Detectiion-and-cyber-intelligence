#!/usr/bin/env python3
"""
Exploratory Data Analysis for CIC-IDS2017 Friday DDos Dataset
Run this from the project root directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Define paths relative to current working directory
DATA_FILE = Path('data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
OUTPUT_DIR = Path('reports/figures/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Looking for file: {DATA_FILE.absolute()}")

# Check if file exists
if not DATA_FILE.exists():
    print(f"ERROR: File not found!")
    print(f"Please check: {DATA_FILE.absolute()}")
    exit(1)

print("File found! Loading dataset...")

# Load the dataset
df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()
print(f"Cleaned column names: {df.columns.tolist()[:5]}...")

# 1. Check label distribution
print("\n" + "="*60)
print("LABEL DISTRIBUTION")
print("="*60)
label_counts = df['Label'].value_counts()
print(label_counts)
print(f"\nPercentage:")
print(df['Label'].value_counts(normalize=True) * 100)

# 2. Remove duplicates
print("\n" + "="*60)
print("REMOVING DUPLICATES")
print("="*60)
initial_shape = df.shape
df = df.drop_duplicates()
print(f"Removed {initial_shape[0] - df.shape[0]:,} duplicate rows")
print(f"New shape: {df.shape[0]:,} rows")

# 3. Remove constant columns
print("\n" + "="*60)
print("REMOVING CONSTANT COLUMNS")
print("="*60)
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
print(f"Constant columns found: {len(constant_cols)}")
if constant_cols:
    print(f"  {constant_cols[:5]}...")
df = df.drop(columns=constant_cols)
print(f"Removed {len(constant_cols)} constant columns")
print(f"New shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

# 4. Handle infinite values
print("\n" + "="*60)
print("HANDLING INFINITE VALUES")
print("="*60)
numeric_cols = df.select_dtypes(include=[np.number]).columns
inf_count_total = 0
for col in numeric_cols:
    inf_count = (df[col].isin([np.inf, -np.inf])).sum()
    if inf_count > 0:
        inf_count_total += inf_count
        print(f"  {col}: {inf_count} infinite values")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())
print(f"Total infinite values handled: {inf_count_total}")

# 5. Handle missing values
print("\n" + "="*60)
print("HANDLING MISSING VALUES")
print("="*60)
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    print(f"Missing values found in {len(missing_cols)} columns:")
    for col, count in missing_cols.items():
        print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    # Fill with median for numeric
    for col in missing_cols.index:
        if col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
            print(f"  Filled {col} with median")
else:
    print("No missing values found")

# 6. Check for identifier columns
print("\n" + "="*60)
print("IDENTIFIER COLUMNS")
print("="*60)
id_patterns = ['Flow ID', 'Flow_ID', 'Src IP', 'Dst IP', 'Source IP', 'Destination IP', 
               'Timestamp', 'Time', 'id', 'ID']
id_cols = [col for col in df.columns if any(pattern in col for pattern in id_patterns)]
print(f"Identifier columns found: {len(id_cols)}")
if id_cols:
    print(f"  {id_cols}")
    print("These will be removed before modeling to prevent data leakage")
    df = df.drop(columns=id_cols)
    print(f"Removed {len(id_cols)} identifier columns")

# 7. Separate features and label
print("\n" + "="*60)
print("FEATURES AND LABEL")
print("="*60)
X = df.drop('Label', axis=1)
y = df['Label']
print(f"Features: {X.shape[1]} columns")
print(f"Label: {y.nunique()} classes")
print(f"Label values: {y.unique()}")

# 8. Create visualizations
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Figure 1: Class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar plot
colors = ['#2ecc71' if label == 'BENIGN' else '#e74c3c' for label in label_counts.index]
axes[0].bar(label_counts.index, label_counts.values, color=colors)
axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 1000, f'{v:,}', ha='center', fontsize=10)

# Pie chart
axes[1].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', 
            colors=['#2ecc71', '#e74c3c'], explode=(0.05, 0.05))
axes[1].set_title('Class Distribution (%)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'class_distribution.png', dpi=300, bbox_inches='tight')
print(f"  Saved: class_distribution.png")

# Figure 2: Feature distributions (first 6 features)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
feature_cols = X.columns[:min(6, len(X.columns))]
for idx, col in enumerate(feature_cols):
    row = idx // 3
    col_idx = idx % 3
    axes[row, col_idx].hist(X[col], bins=50, alpha=0.7, color='steelblue')
    axes[row, col_idx].set_title(col[:30], fontsize=10)  # Truncate long names
    axes[row, col_idx].set_xlabel('Value')
    axes[row, col_idx].set_ylabel('Frequency')
    axes[row, col_idx].tick_params(axis='x', rotation=45)
plt.suptitle('Feature Distributions (First 6 Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_distributions.png', dpi=300, bbox_inches='tight')
print(f"  Saved: feature_distributions.png")

# Figure 3: Correlation heatmap (only first 15 features for readability)
print("  Computing correlation matrix...")
corr_matrix = X.iloc[:, :min(15, len(X.columns))].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix (First 15 Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
print(f"  Saved: correlation_matrix.png")

plt.close('all')
print("\nVisualizations saved to:", OUTPUT_DIR)

# 9. Data types and memory
print("\n" + "="*60)
print("DATA TYPES AND MEMORY")
print("="*60)
print(f"Data types:")
print(X.dtypes.value_counts())
print(f"\nMemory usage: {X.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

# 10. Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total samples: {len(df):,}")
print(f"Features after cleaning: {X.shape[1]}")
print(f"Classes: {y.nunique()}")
print(f"Class distribution: {dict(label_counts)}")
if len(label_counts) == 2:
    imbalance_ratio = label_counts.min() / label_counts.max()
    print(f"Class imbalance ratio: {imbalance_ratio:.4f}")
    if imbalance_ratio < 0.1:
        print("⚠️  Severe class imbalance detected!")

# Save processed data
print("\n" + "="*60)
print("SAVING PROCESSED DATA")
print("="*60)
processed_dir = Path('data/processed/')
processed_dir.mkdir(parents=True, exist_ok=True)

# Save features and labels
X.to_csv(processed_dir / 'X_features.csv', index=False)
y.to_csv(processed_dir / 'y_labels.csv', index=False)

# Save feature names
with open(processed_dir / 'feature_names.txt', 'w') as f:
    for col in X.columns:
        f.write(f"{col}\n")

print(f"Saved features to: {processed_dir / 'X_features.csv'}")
print(f"Saved labels to: {processed_dir / 'y_labels.csv'}")
print(f"Saved feature names to: {processed_dir / 'feature_names.txt'}")

print("\n" + "="*60)
print("EDA COMPLETE!")
print("="*60)

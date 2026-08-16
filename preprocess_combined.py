#!/usr/bin/env python3
"""
Preprocess the COMBINED CIC-IDS2017 dataset for multi-class training.
Loads combined_cicids2017.csv and prepares data for multi-class models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

print("="*60)
print("PREPROCESSING COMBINED DATASET (MULTI-CLASS)")
print("="*60)

# --- LOAD THE COMBINED DATASET ---
combined_path = Path('data/processed/combined_cicids2017.csv')
if not combined_path.exists():
    print("❌ Combined dataset not found!")
    print(f"   Expected at: {combined_path.absolute()}")
    print("   Please run combine_datasets.py first.")
    exit(1)

df = pd.read_csv(combined_path)
print(f"\n✅ Loaded combined dataset: {len(df):,} rows, {len(df.columns)} columns")

# Clean column names
df.columns = df.columns.str.strip()

# --- DATA CLEANING ---
# 1. Remove duplicates
initial_rows = len(df)
df = df.drop_duplicates()
print(f"Removed {initial_rows - len(df):,} duplicates")

# 2. Remove constant columns (except Label)
constant_cols = [col for col in df.columns if col != 'Label' and df[col].nunique() == 1]
df = df.drop(columns=constant_cols)
print(f"Removed {len(constant_cols)} constant columns")

# 3. Handle infinite values
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(df[col].median())

# --- SEPARATE FEATURES AND LABELS ---
X = df.drop('Label', axis=1)
y = df['Label']

# --- ENCODE LABELS (MULTI-CLASS) ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
print(f"\n✅ Encoded {num_classes} classes:")
for i, cls in enumerate(label_encoder.classes_):
    count = (y == cls).sum()
    print(f"  {i}: {cls} ({count:,} samples)")

# --- TRAIN/VAL/TEST SPLIT (STRATIFIED) ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nSplit sizes:")
print(f"  Training: {len(X_train):,}")
print(f"  Validation: {len(X_val):,}")
print(f"  Test: {len(X_test):,}")

# --- FEATURE SCALING ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --- SAVE PROCESSED DATA ---
processed_dir = Path('data/processed')
processed_dir.mkdir(parents=True, exist_ok=True)

np.save(processed_dir / 'X_train.npy', X_train_scaled)
np.save(processed_dir / 'X_val.npy', X_val_scaled)
np.save(processed_dir / 'X_test.npy', X_test_scaled)
np.save(processed_dir / 'y_train.npy', y_train)
np.save(processed_dir / 'y_val.npy', y_val)
np.save(processed_dir / 'y_test.npy', y_test)
joblib.dump(scaler, processed_dir / 'scaler.pkl')
joblib.dump(label_encoder, processed_dir / 'label_encoder.pkl')
# Save feature names
feature_names = X.columns.tolist()
with open(processed_dir / 'feature_names.txt', 'w') as f:
    for name in feature_names:
        f.write(name + '\n')
print(f"\n✅ Preprocessed data saved to {processed_dir}")
print(f"   Features: {X_train_scaled.shape[1]}")
print(f"   Classes: {num_classes}")
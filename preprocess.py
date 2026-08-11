#!/usr/bin/env python3
"""
Preprocessing Pipeline for CIC-IDS2017
- Load cleaned data
- Encode labels
- Scale features
- Split data (train/val/test)
- Handle class imbalance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("PREPROCESSING PIPELINE")
print("="*60)

# Load data
print("\nLoading cleaned data...")
X = pd.read_csv('data/processed/X_features.csv')
y = pd.read_csv('data/processed/y_labels.csv').squeeze()

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]:,}")
print(f"Classes: {y.nunique()}")

# 1. Encode labels
print("\n" + "="*60)
print("ENCODING LABELS")
print("="*60)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
print(f"Class distribution (encoded):")
print(pd.Series(y_encoded).value_counts())

# 2. Train/Validation/Test Split
print("\n" + "="*60)
print("TRAIN/VALIDATION/TEST SPLIT")
print("="*60)

# First split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42,
    stratify=y_encoded
)

# Second split: 80% of train -> train, 20% -> validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Validation set: {X_val.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")

# Check class distribution in splits
print(f"\nTraining class distribution:")
print(pd.Series(y_train).value_counts())
print(f"\nValidation class distribution:")
print(pd.Series(y_val).value_counts())
print(f"\nTest class distribution:")
print(pd.Series(y_test).value_counts())

# 3. Scale features
print("\n" + "="*60)
print("FEATURE SCALING")
print("="*60)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Scaled training data: {X_train_scaled.shape}")
print(f"Scaled validation data: {X_val_scaled.shape}")
print(f"Scaled test data: {X_test_scaled.shape}")

# 4. Handle class imbalance - compute class weights
print("\n" + "="*60)
print("CLASS IMBALANCE HANDLING")
print("="*60)
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
print(f"Class weights: {class_weight_dict}")

# 5. Save preprocessed data
print("\n" + "="*60)
print("SAVING PREPROCESSED DATA")
print("="*60)

# Create directories
processed_dir = Path('data/processed')
processed_dir.mkdir(parents=True, exist_ok=True)

# Save arrays
np.save(processed_dir / 'X_train.npy', X_train_scaled)
np.save(processed_dir / 'X_val.npy', X_val_scaled)
np.save(processed_dir / 'X_test.npy', X_test_scaled)
np.save(processed_dir / 'y_train.npy', y_train)
np.save(processed_dir / 'y_val.npy', y_val)
np.save(processed_dir / 'y_test.npy', y_test)

# Save scaler and encoder
joblib.dump(scaler, processed_dir / 'scaler.pkl')
joblib.dump(label_encoder, processed_dir / 'label_encoder.pkl')

# Save feature names
with open(processed_dir / 'feature_names_processed.txt', 'w') as f:
    for col in X.columns:
        f.write(f"{col}\n")

# Save split info
with open(processed_dir / 'split_info.txt', 'w') as f:
    f.write(f"Training samples: {X_train.shape[0]:,}\n")
    f.write(f"Validation samples: {X_val.shape[0]:,}\n")
    f.write(f"Test samples: {X_test.shape[0]:,}\n")
    f.write(f"Features: {X.shape[1]}\n")
    f.write(f"Classes: {label_encoder.classes_.tolist()}\n")
    f.write(f"Class weights: {class_weight_dict}\n")

print(f"Saved preprocessed data to: {processed_dir}")
print(f"  - X_train.npy, X_val.npy, X_test.npy")
print(f"  - y_train.npy, y_val.npy, y_test.npy")
print(f"  - scaler.pkl")
print(f"  - label_encoder.pkl")
print(f"  - feature_names_processed.txt")
print(f"  - split_info.txt")

# 6. Verify shapes
print("\n" + "="*60)
print("VERIFICATION")
print("="*60)
print(f"X_train shape: {X_train_scaled.shape}")
print(f"X_val shape: {X_val_scaled.shape}")
print(f"X_test shape: {X_test_scaled.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"y_test shape: {y_test.shape}")

print("\n" + "="*60)
print("PREPROCESSING COMPLETE!")
print("="*60)

#!/usr/bin/env python3
"""
Dataset Analysis Script for CIC-IDS2017
Analyzes all CSV files in the dataset directory
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path('data/raw/')
OUTPUT_FILE = 'dataset_analysis_results.txt'

def analyze_csv_file(filepath):
    """Analyze a single CSV file"""
    
    print(f"\n{'='*80}")
    print(f"FILE: {filepath.name}")
    print('='*80)
    
    # Get file size
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")
    
    # Read the file
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return None
    
    # Column information
    print(f"\nCOLUMN INFORMATION:")
    print(f"Total columns: {len(df.columns)}")
    print(f"First 10 columns: {df.columns[:10].tolist()}")
    
    # Data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    print(f"\nData types:")
    print(f"  Numeric columns: {len(numeric_cols)}")
    print(f"  Categorical columns: {len(categorical_cols)}")
    
    # Check for potential identifier columns
    id_patterns = ['Flow ID', 'Flow_ID', 'id', 'ID', 'Src IP', 'Dst IP', 
                   'Source IP', 'Destination IP', 'Timestamp', 'Time']
    id_cols = [col for col in df.columns if any(pattern in col for pattern in id_patterns)]
    if id_cols:
        print(f"\nPotential identifier columns: {id_cols}")
    
    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"\nMISSING VALUES:")
        for col, count in missing_cols.items():
            pct = count / len(df) * 100
            print(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        print(f"\nMISSING VALUES: None found")
    
    # Infinite values
    inf_count = 0
    for col in numeric_cols:
        inf_count += (df[col].isin([np.inf, -np.inf])).sum()
    if inf_count > 0:
        print(f"\nINFINITE VALUES: {inf_count:,}")
    else:
        print(f"\nINFINITE VALUES: None found")
    
    # Duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"\nDUPLICATE ROWS: {dup_count:,} ({dup_count/len(df)*100:.2f}%)")
    else:
        print(f"\nDUPLICATE ROWS: None found")
    
    # Constant columns
    constant_cols = []
    for col in df.columns:
        try:
            if df[col].nunique() == 1:
                constant_cols.append(col)
        except:
            pass
    if constant_cols:
        print(f"\nCONSTANT COLUMNS: {constant_cols}")
    else:
        print(f"\nCONSTANT COLUMNS: None found")
    
    # Find label column
    label_candidates = ['Label', 'label', 'Attack', 'attack', 'Class', 'class', 
                        'Attack Type', 'Attack_Type', 'Target']
    label_col = None
    for col in label_candidates:
        if col in df.columns:
            label_col = col
            break
    
    # If not found, look for columns with few unique values
    if not label_col:
        print("\nLooking for label column...")
        for col in df.columns:
            try:
                unique_vals = df[col].nunique()
                if 2 <= unique_vals <= 20:  # Likely a label column
                    print(f"  {col}: {unique_vals} unique values")
            except:
                pass
    else:
        print(f"\nLABEL COLUMN: '{label_col}'")
        print(f"Unique values: {df[label_col].nunique()}")
        print(f"\nClass distribution:")
        class_dist = df[label_col].value_counts()
        for cls, count in class_dist.items():
            pct = count / len(df) * 100
            print(f"  {cls}: {count:,} ({pct:.2f}%)")
    
    # Statistical summary for numeric columns
    print(f"\nNUMERIC FEATURE STATISTICS:")
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe()
        # Show only first 10 stats for brevity
        print(stats.iloc[:, :min(5, len(numeric_cols))])
    
    # Memory usage
    mem_usage = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"\nMemory usage: {mem_usage:.2f} MB")
    
    return {
        'file': filepath.name,
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'label_col': label_col,
        'class_dist': class_dist if label_col else None,
        'has_missing': len(missing_cols) > 0,
        'has_infinite': inf_count > 0,
        'duplicates': dup_count,
        'constant_cols': constant_cols,
        'numeric_cols': len(numeric_cols),
        'categorical_cols': len(categorical_cols)
    }

def main():
    """Main analysis function"""
    
    print("="*80)
    print("CIC-IDS2017 DATASET ANALYSIS")
    print("="*80)
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {DATA_DIR.absolute()}")
    print()
    
    # Check if directory exists
    if not DATA_DIR.exists():
        print(f"ERROR: Directory {DATA_DIR} does not exist!")
        return
    
    # Find all CSV files
    csv_files = sorted(DATA_DIR.glob('*.csv'))
    
    if len(csv_files) == 0:
        print("No CSV files found in data/raw/")
        print("Please download the dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        size_mb = f.stat().st_size / (1024**2)
        print(f"  • {f.name} ({size_mb:.2f} MB)")
    
    print("\n" + "="*80)
    
    # Analyze each file
    all_results = {}
    for csv_file in csv_files:
        result = analyze_csv_file(csv_file)
        if result:
            all_results[csv_file.name] = result
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total files: {len(csv_files)}")
    print(f"Total rows across all files: {sum(r['shape'][0] for r in all_results.values()):,}")
    
    # Check if all files have the same columns
    if all_results:
        first_columns = set(all_results[csv_files[0].name]['columns'])
        column_match = True
        for fname, result in all_results.items():
            if set(result['columns']) != first_columns:
                column_match = False
                print(f"\nWARNING: {fname} has different columns!")
                break
        
        if column_match:
            print(f"All files have consistent columns: {len(first_columns)} columns")
        else:
            print("Files have inconsistent column structures - need to handle carefully")
    
    # Data leakage concerns
    print("\n" + "="*80)
    print("DATA LEAKAGE CONCERNS")
    print("="*80)
    
    # Check for time-based files
    time_patterns = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    has_time_files = any(any(pattern in f.name for pattern in time_patterns) for f in csv_files)
    if has_time_files:
        print("✓ Time-based files detected (Monday-Friday)")
        print("  → Recommended: Use day-based split (e.g., train on Mon-Thu, test on Fri)")
    else:
        print("✗ No explicit time-based files detected")
        print("  → Consider: Random stratified split")
    
    # Check for identifier columns
    has_identifiers = False
    for fname, result in all_results.items():
        if any('IP' in col or 'id' in col.lower() or 'flow' in col.lower() for col in result['columns']):
            has_identifiers = True
            break
    if has_identifiers:
        print("✓ IP addresses or flow identifiers detected")
        print("  → WARNING: These can cause data leakage")
        print("  → Recommended: Remove before training or handle carefully")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("1. Check if all files have the same schema")
    print("2. Identify the label column (likely 'Label' or similar)")
    print("3. Handle class imbalance (attack classes are usually minority)")
    print("4. Remove constant columns")
    print("5. Remove identifier columns (IP addresses, flow IDs)")
    print("6. Plan train/validation/test split (day-based if possible)")
    print("7. Scale numeric features")
    print("8. Encode labels")
    
    print(f"\nAnalysis complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Verify Dataset Upload - Check if data is correct
Handles column names with spaces
"""

import pandas as pd
import sys
import os

def verify_dataset(file_path):
    """Verify dataset contents"""
    
    print("="*60)
    print("🔍 DATASET VERIFICATION")
    print("="*60)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"\n❌ File not found: {file_path}")
        return None
    
    # Load dataset
    print(f"\n📂 Loading: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"\n❌ Error loading file: {e}")
        return None
    
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    # Basic info
    print(f"\n📊 Basic Statistics:")
    print(f"  Total Rows: {len(df):,}")
    print(f"  Total Columns: {len(df.columns)}")
    
    # File size
    file_size = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  File Size: {file_size:.2f} MB")
    
    # Check for label column (case insensitive, with spaces trimmed)
    label_cols = ['Label', 'label', 'attack', 'Attack', 'class', 'Class']
    found_label = None
    for col in df.columns:
        col_clean = col.strip()
        if col_clean in label_cols:
            found_label = col
            break
    
    if found_label:
        print(f"\n✅ Label Column Found: '{found_label}'")
        print(f"\n📊 Class Distribution:")
        counts = df[found_label].value_counts()
        for label, count in counts.items():
            print(f"  {label}: {count:,} ({count/len(df)*100:.1f}%)")
    else:
        print("\n⚠️ No label column found")
        print("📋 First 10 columns:", df.columns[:10].tolist())
        print("📋 Last 10 columns:", df.columns[-10:].tolist())
    
    # Check for expected features
    print(f"\n📋 Column Types:")
    print(f"  Numeric: {len(df.select_dtypes(include=['number']).columns)}")
    print(f"  Categorical: {len(df.select_dtypes(include=['object']).columns)}")
    
    # Sample data
    print(f"\n📄 Sample Data (first 5 rows):")
    print(df.head(5).to_string())
    
    # Verification checklist
    print("\n" + "="*60)
    print("✅ VERIFICATION CHECKLIST")
    print("="*60)
    
    # Check 1: Rows match
    print(f"  ✓ Rows: {len(df):,}")
    
    # Check 2: Has Label column
    if found_label:
        print(f"  ✓ Label column: '{found_label}'")
        # Show class distribution summary
        counts = df[found_label].value_counts()
        for label, count in counts.items():
            print(f"    - {label}: {count:,} ({count/len(df)*100:.1f}%)")
    else:
        print(f"  ⚠️ Label column: Not found")
        print(f"  📋 Available columns: {df.columns[:5].tolist()}...")
    
    # Check 3: No empty rows
    empty_rows = df.isnull().all(axis=1).sum()
    if empty_rows == 0:
        print(f"  ✓ No empty rows")
    else:
        print(f"  ⚠️ {empty_rows} empty rows found")
    
    print("\n" + "="*60)
    print("📋 VERIFICATION COMPLETE")
    print("="*60)
    
    return df

def main():
    if len(sys.argv) < 2:
        print("="*60)
        print("📂 DATASET VERIFICATION TOOL")
        print("="*60)
        print("\nUsage: python3 verify_dataset.py <dataset.csv>")
        print("\nExamples:")
        print("  python3 verify_dataset.py data/raw/Monday-WorkingHours.pcap_ISCX.csv")
        print("  python3 verify_dataset.py data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
        print("\n📂 Available datasets:")
        if os.path.exists("data/raw"):
            for f in os.listdir("data/raw"):
                if f.endswith('.csv'):
                    size = os.path.getsize(f"data/raw/{f}") / (1024*1024)
                    print(f"  - {f} ({size:.1f} MB)")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Try to find the file
    if not os.path.exists(file_path):
        # Try with data/raw prefix
        alt_path = f"data/raw/{file_path}"
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            print(f"\n❌ File not found: {file_path}")
            sys.exit(1)
    
    verify_dataset(file_path)

if __name__ == "__main__":
    main()

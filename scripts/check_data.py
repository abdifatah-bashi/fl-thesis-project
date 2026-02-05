"""
Check what data we actually have
"""

import pandas as pd
from pathlib import Path

print("=" * 60)
print("DATA INVESTIGATION")
print("=" * 60)

# Check CSV file
csv_path = 'data/heart_disease/heart_disease_full.csv'
if Path(csv_path).exists():
    df_csv = pd.read_csv(csv_path)
    print(f"\nCSV File: {csv_path}")
    print(f"Shape: {df_csv.shape}")
    print(f"Columns: {list(df_csv.columns)}")
    print(f"\nFirst 3 rows:")
    print(df_csv.head(3))
else:
    print(f"\nCSV file not found: {csv_path}")

# Check raw .data files
raw_dir = Path('data/heart_disease/raw')
if raw_dir.exists():
    print("\n" + "=" * 60)
    print("RAW .DATA FILES")
    print("=" * 60)
    
    files = {
        'Cleveland': 'processed.cleveland.data',
        'Hungary': 'processed.hungarian.data',
        'Switzerland': 'processed.switzerland.data',
        'VA Long Beach': 'processed.va.data'
    }
    
    total = 0
    for hospital, filename in files.items():
        filepath = raw_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath, header=None)
            total += len(df)
            print(f"\n{hospital:15} : {len(df):3} rows")
        else:
            print(f"\n{hospital:15} : NOT FOUND")
    
    print(f"\n{'TOTAL':15} : {total:3} rows")
else:
    print(f"\nRaw data directory not found: {raw_dir}")

print("\n" + "=" * 60)
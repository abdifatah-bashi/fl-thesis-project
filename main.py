"""
Download UCI Heart Disease Dataset for FL System
This script downloads the dataset and prepares it for federated learning.
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path


def download_heart_disease_dataset(output_dir='data/heart_disease'):
    """
    Download UCI Heart Disease dataset for FL system demonstration.
    
    Args:
        output_dir: Directory to save the dataset
    """
    print("=" * 60)
    print("DOWNLOADING DATASET FOR FL SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Fetch dataset
    print("\nFetching UCI Heart Disease dataset...")
    heart_disease = fetch_ucirepo(id=45)
    
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    # Combine features and target
    df = pd.concat([X, y], axis=1)
    
    # Save to CSV
    output_file = output_path / 'heart_disease_full.csv'
    df.to_csv(output_file, index=False)
    
    # Display summary
    print(f"\n✅ Dataset downloaded for FL system")
    print(f"\nFL System Properties:")
    print(f"  Total samples: {len(df)}")
    print(f"  Features: {len(X.columns)}")
    print(f"  Natural clients: 4 hospitals")
    print(f"  Avg samples/client: ~{len(df)//4}")
    
    # Quick look
    print(f"\nFirst 3 rows (example data for FL system):")
    print(df.head(3))
    
    print(f"\n✅ Saved to: {output_file}")
    print("✅ Ready for FL system development!")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    # Run the download
    download_heart_disease_dataset()
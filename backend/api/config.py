from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

SAMPLE_DATA: dict[str, Path] = {
    "cleveland":  ROOT / "data/heart_disease/raw/processed.cleveland.data",
    "hungarian":  ROOT / "data/heart_disease/raw/processed.hungarian.data",
    "va":         ROOT / "data/heart_disease/raw/processed.va.data",
    "switzerland": ROOT / "data/heart_disease/raw/processed.switzerland.data",
}

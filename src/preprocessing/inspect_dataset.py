
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

FILES = ["CEAS_08.csv", "Nazario.csv"]

for filename in FILES:
    path = DATA_DIR / filename

    print(f"\n{'=' * 60}")
    print(f"DATASET: {filename}")
    print("=" * 60)

    if not path.exists():
        print("ERROR: File not found:", path)
        continue

    try:
        df = pd.read_csv(path, low_memory=False)

        print("\n1. Shape:", df.shape)
        print("\n2. Columns:")
        print(df.columns.tolist())

        print("\n3. Data types:")
        print(df.dtypes.to_string())

        print("\n4. Missing values:")
        print(df.isnull().sum().to_string())

        print("\n5. Duplicate rows:", df.duplicated().sum())

        for col in df.columns:
            if col.lower() in {"label", "class", "target", "type"}:
                print(f"\n6. Label distribution ({col}):")
                print(df[col].value_counts(dropna=False).to_string())

        print("\n7. Sample structure:")
        for col in df.columns:
            value = df[col].iloc[0] if len(df) else ""
            if col.lower() in {"body", "text", "content", "message"}:
                print(f"{col}: [TEXT CONTENT HIDDEN]")
            else:
                print(f"{col}: {str(value)[:100]}")

    except Exception as e:
        print(f"ERROR processing {filename}: {e}")

print("\nInspection completed.")

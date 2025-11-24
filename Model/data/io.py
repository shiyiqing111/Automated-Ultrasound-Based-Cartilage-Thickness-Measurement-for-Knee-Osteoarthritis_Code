import pandas as pd
from pathlib import Path

def load_labeled_data(labeled_excel: Path, image_dir: Path):
    df = pd.read_excel(labeled_excel)
    records = []
    for _, row in df.iterrows():
        fname = str(row["Filename"]).strip()
        img_path = image_dir / f"{fname}.jpg"
        if not img_path.exists():
            continue
        coords = row[["x1","y1","x2","y2","x3","y3","x4","y4","x5","y5","x6","y6"]].values.astype(float)
        records.append({"Filename": fname, "image": img_path, "coords": coords})
    print(f"✅ Loaded {len(records)} labeled samples.")
    return records

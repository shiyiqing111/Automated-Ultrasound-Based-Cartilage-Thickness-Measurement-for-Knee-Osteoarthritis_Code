import pandas as pd
from pathlib import Path


RAW_ANNOT_FILE = Path(r"C:\Users\Charlotte\Desktop\dissertation\US_new\annotation_points_summary2.xlsx")
MERGED_OUTPUT = Path(__file__).resolve().parent / "merged_annotation.xlsx"

# =======================================
def _region_order(label: str) -> int:
    label = str(label).lower().strip()
    if "lateral" in label:
        return 0
    elif "femoral" in label or "condyle" in label:
        return 1
    elif "medial" in label:
        return 2
    else:
        return 99 

def merge_annotation(df: pd.DataFrame) -> pd.DataFrame:
    df["region_order"] = df["Label"].apply(_region_order)

    merged_rows = []
    for fname, group in df.groupby("Filename"):
        group = group.sort_values("region_order")
        row = {"Filename": fname}
        for i, (_, seg) in enumerate(group.iterrows(), start=1):
            row[f"x{2*i-1}"], row[f"y{2*i-1}"] = seg["x1"], seg["y1"]
            row[f"x{2*i}"], row[f"y{2*i}"] = seg["x2"], seg["y2"]
            row[f"Pixel_Distance_{i}"] = seg.get("Pixel_Distance", None)
        merged_rows.append(row)
    return pd.DataFrame(merged_rows)


if __name__ == "__main__":
    if not RAW_ANNOT_FILE.exists():
        raise FileNotFoundError(f"The original annotation file was not found: {RAW_ANNOT_FILE}")

    df_raw = pd.read_excel(RAW_ANNOT_FILE)
    print(f"Read the original annotation file, with a total of {len(df_raw)} records.")

    df_merged = merge_annotation(df_raw)
    df_merged.to_excel(MERGED_OUTPUT, index=False)
    print(f"The merger has been completed：{len(df_merged)} images (each image contains three line segments)")
    print(f"Saved to: {MERGED_OUTPUT}")

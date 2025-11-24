from config import BOOK_EXCEL, PIXEL_TO_CM
from utils.metrics import summarize_length_metrics
from utils.geometry import seg_lengths_from_coords, apply_linear_correction
import pandas as pd

def evaluate_vs_doctor(pred_coords_path, save_path="results/book_eval_summary.xlsx"):
    pred_df = pd.read_excel(pred_coords_path)
    pred_len = seg_lengths_from_coords(pred_df.rename(columns={
        "x1":"seg1_x1","y1":"seg1_y1","x2":"seg1_x2","y2":"seg1_y2",
        "x3":"seg2_x1","y3":"seg2_y1","x4":"seg2_x2","y4":"seg2_y2",
        "x5":"seg3_x1","y5":"seg3_y1","x6":"seg3_x2","y6":"seg3_y2"
    }))
    pred_len_mm = apply_linear_correction(pred_len * PIXEL_TO_CM)

    book = pd.read_excel(BOOK_EXCEL)
    book_mean = book.filter(like="US_").mean(axis=1)

    metrics = summarize_length_metrics(book_mean.to_frame("seg1_length"), pred_len_mm)
    metrics.to_excel(save_path, index=False)
    print(f"Saved evaluation summary: {save_path}")

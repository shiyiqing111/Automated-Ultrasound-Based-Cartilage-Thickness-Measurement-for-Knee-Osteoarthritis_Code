import os
import sys

# 当前文件：US_new/Model/optimization/seg_length_regression_resnet50_physio.py
# 项目根目录（有 config.py 的那个）：US_new/Model
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import *
from pathlib import Path
import numpy as np
import pandas as pd
import re

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

print(">>> running seg_length_regression_resnet50_physio.py")

# --------------------------
# 小工具：group id / stage
# --------------------------
def remove_last_index(name: str) -> str:
    return re.sub(r"(_\d+)$", "", name)

def match_stage(name: str) -> str:
    for stage in ["_base", "_3", "_4", "_5"]:
        if stage in name:
            return stage
    return ""

# --------------------------
# 计算 3 段长度（像素）的小函数
# --------------------------
def compute_seg_lengths_from_coords(coord_df: pd.DataFrame) -> pd.DataFrame:
    """
    输入列：
      seg1_x1, seg1_y1, seg1_x2, seg1_y2,
      seg2_x1, seg2_y1, seg2_x2, seg2_y2,
      seg3_x1, seg3_y1, seg3_x2, seg3_y2
    输出：
      seg1_length, seg2_length, seg3_length（像素）
    """
    out = {}
    for i in [1, 2, 3]:
        x1 = coord_df[f"seg{i}_x1"].values
        y1 = coord_df[f"seg{i}_y1"].values
        x2 = coord_df[f"seg{i}_x2"].values
        y2 = coord_df[f"seg{i}_y2"].values
        out[f"seg{i}_length"] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return pd.DataFrame(out)

# --------------------------
# 文献基础上的“生理合理范围”（单位：cm）
# --------------------------
physio_ranges_cm = {
    "lateral": (0.05, 0.40),   # 0.5–4.0 mm
    "femoral": (0.05, 0.45),   # 0.5–4.5 mm
    "medial":  (0.05, 0.45),
}
print("Physiological ranges (cm):", physio_ranges_cm)

# =====================================================
# 1. 读取 ResNet50 特征（已 PCA）
# =====================================================
feat_path = Path(RESULT_DIR) / "resnet50_features_pca.npz"
feat_data = np.load(feat_path, allow_pickle=True)
X_train, X_test = feat_data["X_train"], feat_data["X_test"]
train_files, test_files = feat_data["train_filenames"], feat_data["test_filenames"]

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# =====================================================
# 2. 从标注坐标算出 3 段长度（cm）
#    约定：seg1 = lateral, seg2 = femoral, seg3 = medial
# =====================================================
df_label = pd.read_excel(LABELED_EXCEL)

coord_cols = [
    "x1","y1","x2","y2",
    "x3","y3","x4","y4",
    "x5","y5","x6","y6",
]
coord_df = df_label[coord_cols].rename(columns={
    "x1":"seg1_x1","y1":"seg1_y1","x2":"seg1_x2","y2":"seg1_y2",
    "x3":"seg2_x1","y3":"seg2_y1","x4":"seg2_x2","y4":"seg2_y2",
    "x5":"seg3_x1","y5":"seg3_y1","x6":"seg3_x2","y6":"seg3_y2",
})

# 像素长度
seg_len_px = compute_seg_lengths_from_coords(coord_df)

# 转 cm（用全局像素→cm 系数）
seg_len_cm = seg_len_px * PIXEL_TO_CM
seg_len_cm.columns = ["lateral", "femoral", "medial"]

print("Segment length (cm) head:")
print(seg_len_cm.head())

# =====================================================
# 3. 定义模型 & y 标准化函数
# =====================================================
def get_models():
    models = {
        "ridge": Ridge(),
        "rf": RandomForestRegressor(random_state=42, n_jobs=-1),
        "xgb": xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
    }
    param_grids = {
        "ridge": {
            "alpha": [0.1, 1.0, 10.0],
        },
        "rf": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10],
            "max_features": ["sqrt"],
            "min_samples_split": [2, 5],
        },
        "xgb": {
            "n_estimators": [200, 300],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
    }
    return models, param_grids

def zscore_fit(y):
    mean = y.mean()
    std = y.std() + 1e-6
    return (y - mean) / std, mean, std

def zscore_inv(y_norm, mean, std):
    return y_norm * std + mean

# =====================================================
# 4. 逐 segment 建模 & 预测（带生理范围 clip）
# =====================================================
book = pd.read_excel(BOOK_EXCEL)
summary_records = []

PRED_DIR_OPT = PRED_DIR / "seg_length_models_physio_resnet50"
PRED_DIR_OPT.mkdir(parents=True, exist_ok=True)

segment_order = ["lateral", "femoral", "medial"]
models, param_grids = get_models()

for seg in segment_order:
    print(f"\n=== Segment: {seg.upper()} ===")

    y_train_raw = seg_len_cm[seg].values  # cm
    y_train_norm, y_mean, y_std = zscore_fit(y_train_raw)

    for model_name, base_model in models.items():
        print(f"\n--- Model: {model_name.upper()} ---")

        param_grid = param_grids[model_name]

        grid = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train_norm)
        best_model = grid.best_estimator_
        print(f"Best params for {seg} - {model_name}: {grid.best_params_}")

        best_model.fit(X_train, y_train_norm)

        # 预测（z-score 空间）
        y_pred_norm = best_model.predict(X_test)
        y_pred_cm = zscore_inv(y_pred_norm, y_mean, y_std)

        # ★ 文献约束：限制在生理合理范围内（单位：cm）
        lo, hi = physio_ranges_cm[seg]
        y_pred_cm_clipped = np.clip(y_pred_cm, lo, hi)

        # 保存逐图像预测
        df_pred = pd.DataFrame({
            "Filename": test_files,
            f"US_{seg}_pred_cm": y_pred_cm_clipped,
        })
        out_pred_path = PRED_DIR_OPT / f"{model_name}_resnet50_{seg}_length_pred_physio.xlsx"
        df_pred.to_excel(out_pred_path, index=False)
        print(f"Saved per-image predictions: {out_pred_path}")

        # -----------------------
        # 5. 与医生测量对齐并计算 MAE/RMSE/R2
        # -----------------------
        pred_all = df_pred.copy()
        pred_all["GroupID"] = pred_all["Filename"].apply(remove_last_index)
        pred_group = pred_all.groupby("GroupID")[f"US_{seg}_pred_cm"].mean().reset_index()
        pred_group["Patient"] = pred_group["GroupID"].str.extract(r"(Abbey_\d+)")[0].str.replace("_"," ")
        pred_group["Stage"] = pred_group["GroupID"].apply(match_stage)

        seg_pred_list, seg_true_list = [], []

        for _, row in pred_group.iterrows():
            stage, patient = row["Stage"], row["Patient"]
            colname = f"US_{seg}_treat{stage}"
            if colname not in book.columns:
                colname = f"US_{seg}_contralateral{stage}"
            if colname not in book.columns:
                continue
            val_true = book.loc[book["Patient"] == patient, colname]
            if len(val_true) == 0 or pd.isna(val_true.values[0]):
                continue
            seg_pred_list.append(row[f"US_{seg}_pred_cm"])
            seg_true_list.append(val_true.values[0])

        if not seg_pred_list:
            print(f"[WARN] No matched records for {seg} - {model_name}")
            continue

        df_seg = pd.DataFrame({"pred": seg_pred_list, "true": seg_true_list})
        mae = np.mean(np.abs(df_seg["pred"] - df_seg["true"]))
        rmse = np.sqrt(np.mean((df_seg["pred"] - df_seg["true"]) ** 2))
        r2 = np.corrcoef(df_seg["pred"], df_seg["true"])[0, 1] ** 2

        print(f"[METRICS] {seg} - {model_name}: N={len(df_seg)}, MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

        summary_records.append({
            "Backbone": "resnet50",
            "Segment": seg,
            "Model": model_name,
            "N": len(df_seg),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

# =====================================================
# 6. 汇总表格
# =====================================================
summary_df = pd.DataFrame(summary_records)
out_summary = Path(RESULT_DIR) / "segment_length_models_resnet50_physio.xlsx"
summary_df.to_excel(out_summary, index=False)
print(f"\nSaved summary: {out_summary}")

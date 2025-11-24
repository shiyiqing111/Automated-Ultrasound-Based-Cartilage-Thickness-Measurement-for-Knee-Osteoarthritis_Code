import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, ConstantInputWarning
print(">>> metrics module loaded (auto segment version)")

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def r2(y_true, y_pred):
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float('nan')

def pearson(y_true, y_pred):
    # 常量输入/样本过少时返回 NaN，不抛 Warning
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConstantInputWarning)
            r, _ = pearsonr(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel())
        return float(r)
    except Exception:
        return float('nan')

def _common_seg_length_cols(df_true_len: pd.DataFrame, df_pred_len: pd.DataFrame):
    t_cols = {c for c in df_true_len.columns if c.startswith("seg") and c.endswith("_length")}
    p_cols = {c for c in df_pred_len.columns if c.startswith("seg") and c.endswith("_length")}
    commons = sorted(
        t_cols.intersection(p_cols),
        key=lambda x: int(x[3:].split("_")[0])  # "seg2_length" -> 2
    )
    return commons

def summarize_length_metrics(df_true_len: pd.DataFrame, df_pred_len: pd.DataFrame):
    """
    只对双方都存在的 segK_length 计算指标；自适应段数（1 段或多段都可）。
    """
    seg_cols = _common_seg_length_cols(df_true_len, df_pred_len)
    if not seg_cols:
        raise ValueError(
            "未找到共同的 *_length 列。\n"
            f"true 列: {list(df_true_len.columns)}\n"
            f"pred 列: {list(df_pred_len.columns)}"
        )
    rows = []
    for col in seg_cols:
        t = df_true_len[col].values
        p = df_pred_len[col].values
        rows.append({
            "segment": col.replace("_length", ""),
            "RMSE": rmse(t, p),
            "MAE":  mae(t, p),
            "R2":   r2(t, p),
            "Pearson": pearson(t, p),
        })
    return pd.DataFrame(rows)

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def seg_lengths_from_coords(df: pd.DataFrame, prefix="seg", n_segments=3):
    """
    根据 seg{i}_x1 ... seg{i}_y2 计算每一段的长度
    默认 3 段：seg1, seg2, seg3
    """
    out = {}
    for i in range(1, n_segments + 1):
        x1 = df[f"{prefix}{i}_x1"]
        y1 = df[f"{prefix}{i}_y1"]
        x2 = df[f"{prefix}{i}_x2"]
        y2 = df[f"{prefix}{i}_y2"]
        out[f"{prefix}{i}_length"] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return pd.DataFrame(out)

def apply_linear_correction(length_mm, slope=1.0, intercept=0.0):
    return length_mm * slope + intercept

# 下面的 apply_geom_post 保持不动


def apply_geom_post(Ypred, df_labeled=None, z_th=2.0):
    """
    自动适配点数量的几何约束：
    - 若 df_labeled 只有 x1,y1,x2,y2，则仅一段线
    - 若包含更多点，则计算多段均值
    """
    Yadj = Ypred.copy()
    if df_labeled is None or len(df_labeled) == 0:
        return Yadj

    # 自动检测有多少对点
    xy_cols = [c for c in df_labeled.columns if c.lower().startswith("x")]
    num_points = len(xy_cols)
    num_pairs = num_points // 2
    if num_pairs == 0:
        return Yadj

    seg_means, seg_stds = [], []
    for i in range(num_pairs):
        try:
            x1 = df_labeled[f"x{2*i+1}"].values
            y1 = df_labeled[f"y{2*i+1}"].values
            x2 = df_labeled[f"x{2*i+2}"].values
            y2 = df_labeled[f"y{2*i+2}"].values
        except KeyError:
            break
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        seg_means.append(L.mean())
        seg_stds.append(L.std() + 1e-6)

    # 逐样本调整
    for n in range(Yadj.shape[0]):
        for i in range(len(seg_means)):
            x1_idx, y1_idx = 2*i, 2*i+1
            x2_idx, y2_idx = 2*i+2, 2*i+3
            if x2_idx >= Yadj.shape[1]:  # 防止越界
                break
            dx = Yadj[n, x2_idx] - Yadj[n, x1_idx]
            dy = Yadj[n, y2_idx] - Yadj[n, y1_idx]
            L = np.sqrt(dx*dx + dy*dy) + 1e-6
            if abs(L - seg_means[i]) > z_th * seg_stds[i]:
                scale = seg_means[i] / L
                cx = (Yadj[n, x1_idx] + Yadj[n, x2_idx]) / 2
                cy = (Yadj[n, y1_idx] + Yadj[n, y2_idx]) / 2
                Yadj[n, x1_idx] = cx - (Yadj[n, x2_idx] - cx) * scale
                Yadj[n, y1_idx] = cy - (Yadj[n, y2_idx] - cy) * scale
                Yadj[n, x2_idx] = cx + (Yadj[n, x2_idx] - cx) * scale
                Yadj[n, y2_idx] = cy + (Yadj[n, y2_idx] - cy) * scale
    return Yadj

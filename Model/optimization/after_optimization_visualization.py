import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import *
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_physio_summaries():

    res18_path = Path(RESULT_DIR) / "segment_length_models_resnet18_physio.xlsx"
    res50_path = Path(RESULT_DIR) / "segment_length_models_resnet50_physio.xlsx"

    df18 = pd.read_excel(res18_path)
    df50 = pd.read_excel(res50_path)

    for df in (df18, df50):
        if "Segment" in df.columns:
            df["Segment"] = df["Segment"].astype(str).str.lower().str.strip()
        if "Model" in df.columns:
            df["Model"] = df["Model"].astype(str).str.lower().str.strip()

        for cand in ["R2", "r2", "R²", "r²"]:
            if cand in df.columns:
                df.rename(columns={cand: "R2"}, inplace=True)
                break

    return {"resnet18": df18, "resnet50": df50}


def _plot_backbone_horizontal_ranked(df: pd.DataFrame,
                                     backbone_name: str,
                                     metric: str = "MAE",
                                     out_dir: Path = None):

    if out_dir is None:
        out_dir = Path(RESULT_DIR) / "comparison_physio" / "hbar_ranked"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 设置论文风格的matplotlib参数
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 10,
        'figure.dpi': 350,
        'savefig.dpi': 350,
        'figure.figsize': (7, 4.5),
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.4,
        'grid.alpha': 0.3
    })

    desired_models = ["xgb", "ridge", "rf"]
    df = df.copy()
    df = df[df["Model"].isin(desired_models)]

    if df.empty:
        print(f"[WARN] {backbone_name}: no rows for models {desired_models}")
        return

    segments = ["lateral", "femoral", "medial"]
    segment_display = {"lateral": "Lateral", "femoral": "Femoral", "medial": "Medial"}

    if metric.lower() in ["r2", "r²"]:
        ascending = False
    else:
        ascending = True

    # 使用更美观的色系 - 柔和的蓝色调
    colors = {
        "xgb": "#699FDA",    # 柔和的蓝色
        "ridge": "#F9A857",  # 柔和的橙色
        "rf": "#6EB764",     # 柔和的绿色
    }

    model_display = {
        "xgb": "XGBoost",
        "ridge": "Ridge",
        "rf": "Random Forest",
    }

    n_models = len(desired_models)
    cluster_gap = n_models + 1.0
    centers = np.arange(len(segments)) * cluster_gap 

    ys = []
    xs = []
    bar_colors = []
    bar_labels_for_legend = []
    legend_used = set()

    if n_models > 1:
        max_offset = 0.5
        offsets = np.linspace(-max_offset, max_offset, n_models)
    else:
        offsets = np.array([0.0])

    # 创建图形
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # 首先收集所有数据
    segment_data = {}
    for i, seg in enumerate(segments):
        center = centers[i]
        seg_df = df[df["Segment"] == seg]
        if not seg_df.empty:
            seg_df = seg_df.set_index("Model").reindex(desired_models).dropna(subset=[metric])
            if not seg_df.empty:
                seg_df = seg_df.reset_index()
                seg_df = seg_df.sort_values(by=metric, ascending=ascending).reset_index(drop=True)
                segment_data[seg] = (center, seg_df)

    # 绘制条形图
    bars = []
    for seg, (center, seg_df) in segment_data.items():
        for idx, row in seg_df.iterrows():
            m = row["Model"]
            val = float(row[metric])
            y = center + offsets[idx]

            ys.append(y)
            xs.append(val)
            bar_colors.append(colors.get(m, "#999999"))

            if m not in legend_used:
                bar_labels_for_legend.append(model_display.get(m, m.upper()))
                legend_used.add(m)

            # 绘制条形
            b = ax.barh(y, val, height=0.5, color=colors.get(m), 
                       edgecolor="white", linewidth=1.0, alpha=0.85)
            bars.append(b)

            # 在条形右侧紧挨着添加黑色数值标签
            text_x = val + 0.001  # 紧挨着柱子右侧，稍微偏移一点
            ax.text(text_x, y, f"{val:.3f}",
                    va="center", ha="left", fontsize=8, color="black", 
                    fontweight='bold')

    # 设置y轴标签和刻度
    ax.set_yticks(centers)
    ax.set_yticklabels([segment_display[s] for s in segments], fontweight='bold')

    ax.invert_yaxis()
    
    # 设置x轴标签
    if metric == "MAE":
        xlabel = "Mean Absolute Error (mm)"
    elif metric == "R2":
        xlabel = "R² Score"
    else:
        xlabel = metric
        
    ax.set_xlabel(xlabel, fontweight='bold')
    
    # 设置标题
    backbone_display = "ResNet-18" if "18" in backbone_name else "ResNet-50"
    ax.set_title(f"{backbone_display} - CartillaThickness Estimation", 
                 fontsize=12, fontweight='bold', pad=15)

    # 添加图例 - 放在图表外部
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # 去重
        unique_handles = []
        unique_labels = []
        seen_labels = set()
        for handle, label in zip(handles, labels):
            if label not in seen_labels:
                unique_handles.append(handle)
                unique_labels.append(label)
                seen_labels.add(label)
        
        # 将图例放在图表右侧外部
        ax.legend(unique_handles, unique_labels, 
                 loc='center left', 
                 bbox_to_anchor=(1.02, 0.5),
                 frameon=True, 
                 fancybox=True, 
                 shadow=False,
                 framealpha=0.9,
                 edgecolor='#CCCCCC')

    # 网格和边框样式
    ax.grid(axis="x", linestyle="--", alpha=0.4, linewidth=0.6)
    ax.set_axisbelow(True)
    
    # 设置边框
    for spine in ax.spines.values():
        spine.set_color('#666666')
        spine.set_linewidth(0.8)

    # 设置x轴范围，确保所有条形和标签都能显示
    if xs:
        x_max = max(xs) * 1.15  # 留出15%的边距给标签
        ax.set_xlim(0, x_max)
    else:
        ax.set_xlim(0, 0.1)

    # 调整布局，为外部图例留出空间
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # 右侧留出15%的空间给图例

    # 保存图像
    out_path = out_dir / f"segment_length_physio_{backbone_name}_{metric}_hbar_ranked.png"
    plt.savefig(out_path, dpi=350, bbox_inches="tight", 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved:", out_path)


def main():
    summary_dict = load_physio_summaries()

    out_dir = Path(RESULT_DIR) / "comparison_physio" / "hbar_ranked"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric = "MAE"

    _plot_backbone_horizontal_ranked(summary_dict["resnet18"], "resnet18", metric, out_dir)
    _plot_backbone_horizontal_ranked(summary_dict["resnet50"], "resnet50", metric, out_dir)


if __name__ == "__main__":
    main()
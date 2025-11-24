import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import re
from config import RESULT_DIR, PIXEL_TO_CM
from utils.geometry import seg_lengths_from_coords, apply_linear_correction


# ============================================================
# Directory preparation
# ============================================================
def prepare_dirs(backbone):
    sub = Path(RESULT_DIR) / backbone
    folders = {
        "grouped": sub / "grouped_bars",
        "scatter": sub / "scatter",
        "residuals": sub / "residuals",
        "bland": sub / "bland_altman",
        "comparison": Path(RESULT_DIR) / "comparison"
    }
    for d in folders.values():
        d.mkdir(parents=True, exist_ok=True)
    return folders


# ============================================================
# Load summary file for a specific backbone
# ============================================================
def load_summary(backbone):
    fname = f"all_models_segment_summary_{backbone}.xlsx"
    summary_path = Path(RESULT_DIR) / fname
    return pd.read_excel(summary_path)


# ============================================================
# Load prediction results and match with manual ground truth
# ============================================================
def load_prediction_with_truth(backbone):
    pred_file_dict = {}
    book = pd.read_excel("C:/Users/Charlotte/Desktop/dissertation/US_new/Book2.xlsx")

    for model in ["knn", "linear", "rf", "xgb"]:
        pred_path = Path(RESULT_DIR).parent / "predictions" / f"{model}_{backbone}_unlabeled_pred.xlsx"
        if not pred_path.exists():
            print(f"Missing file: {pred_path}")
            continue

        pred_df = pd.read_excel(pred_path)

        pred_df["GroupID"] = pred_df["Filename"].str.replace(r"_\d+$", "", regex=True)
        pred_df["Patient"] = pred_df["GroupID"].str.extract(r"(Abbey_\d+)", expand=False)
        pred_df["Patient"] = pred_df["Patient"].str.replace("_", " ")

        renamed = pred_df.rename(columns={
            "x1": "seg1_x1", "y1": "seg1_y1", "x2": "seg1_x2", "y2": "seg1_y2",
            "x3": "seg2_x1", "y3": "seg2_y1", "x4": "seg2_x2", "y4": "seg2_y2",
            "x5": "seg3_x1", "y5": "seg3_y1", "x6": "seg3_x2", "y6": "seg3_y2"
        })
        px_len = seg_lengths_from_coords(renamed)
        cm_len = apply_linear_correction(px_len * PIXEL_TO_CM)
        cm_len.columns = ["lateral_pred", "femoral_pred", "medial_pred"]

        pred_all = pd.concat([pred_df[["GroupID", "Patient"]], cm_len], axis=1)

        true_lateral = []
        true_femoral = []
        true_medial = []

        for _, row in pred_all.iterrows():
            patient = row["Patient"]
            group = row["GroupID"]

            stage_match = re.findall(r"_(base|3|4|5)", group)
            stage = f"_{stage_match[0]}" if stage_match else "_base"

            def get_truth(seg):
                col1 = f"US_{seg}_treat{stage}"
                col2 = f"US_{seg}_contralateral{stage}"
                if col1 in book.columns:
                    v = book.loc[book["Patient"] == patient, col1]
                elif col2 in book.columns:
                    v = book.loc[book["Patient"] == patient, col2]
                else:
                    return np.nan
                return v.values[0] if len(v) > 0 else np.nan

            true_lateral.append(get_truth("lateral"))
            true_femoral.append(get_truth("femoral"))
            true_medial.append(get_truth("medial"))

        pred_all["true_lateral"] = true_lateral
        pred_all["true_femoral"] = true_femoral
        pred_all["true_medial"] = true_medial

        pred_file_dict[model] = pred_all

    return pred_file_dict


# ============================================================
# Visualization: grouped bar charts
# ============================================================
def plot_comparison_bars(summary_dict):
    out_dir = Path(RESULT_DIR) / "comparison" / "bars"
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = ["Lateral", "Femoral", "Medial"]
    metrics = ["MAE", "RMSE", "R2"]
    palette = {"resnet18": "#4C72B0", "resnet50": "#C44E52"}

    all_models = sorted(set(
        pd.concat([summary_dict["resnet18"], summary_dict["resnet50"]])["Model"].unique()
    ))

    for metric in metrics:
        for m in all_models:
            plt.figure(figsize=(9, 6))
            x = np.arange(len(segments))
            bar_width = 0.35

            for i, backbone in enumerate(["resnet18", "resnet50"]):
                df = summary_dict.get(backbone)
                if df is None:
                    continue
                subset = df[df["Model"] == m]
                plt.bar(
                    x + i * bar_width - bar_width / 2,
                    subset[metric].values,
                    width=bar_width,
                    color=palette[backbone],
                    label=backbone.upper()
                )

            plt.xticks(x, segments)
            plt.ylabel(metric)
            plt.title(f"{m.upper()} — ResNet18 vs ResNet50 — {metric}")
            plt.legend()

            out = out_dir / f"comparison_{m}_{metric}.png"
            plt.savefig(out, dpi=350, bbox_inches="tight")
            plt.close()
            print("Saved:", out)


# ============================================================
# Visualization: scatter plots
# ============================================================
def plot_scatter(pred_file_dict, folders, backbone):
    palette = {"knn": "#4C72B0", "linear": "#55A868", "rf": "#C44E52", "xgb": "#8172B2"}

    for model, df in pred_file_dict.items():
        for seg in ["lateral", "femoral", "medial"]:
            true_col = f"true_{seg}"
            pred_col = f"{seg}_pred"
            tmp = df[[true_col, pred_col]].dropna()

            plt.figure(figsize=(7, 6))
            sns.scatterplot(x=tmp[true_col], y=tmp[pred_col],
                            color=palette[model], s=55)
            sns.regplot(x=tmp[true_col], y=tmp[pred_col],
                        scatter=False, color="black")

            plt.xlabel("Manual (cm)")
            plt.ylabel("Predicted (cm)")
            plt.title(f"{backbone} — {model.upper()} — {seg.capitalize()}")

            out = folders["scatter"] / f"{model}_{seg}.png"
            plt.savefig(out, dpi=350, bbox_inches="tight")
            plt.close()
            print("Saved:", out)


# ============================================================
# Visualization: residual distributions
# ============================================================
def plot_residuals(pred_file_dict, folders):
    palette = {"knn": "#4C72B0", "linear": "#55A868", "rf": "#C44E52", "xgb": "#8172B2"}

    for model, df in pred_file_dict.items():
        residuals = []
        for seg in ["lateral", "femoral", "medial"]:
            tmp = df[[f"true_{seg}", f"{seg}_pred"]].dropna()
            residuals.extend(tmp[f"{seg}_pred"] - tmp[f"true_{seg}"])

        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True, bins=20,
                     color=palette[model], alpha=0.7)
        plt.xlabel("Residual (cm)")
        plt.title(f"{model.upper()} Residual Distribution")

        out = folders["residuals"] / f"{model}.png"
        plt.savefig(out, dpi=350, bbox_inches="tight")
        plt.close()
        print("Saved:", out)


# ============================================================
# Visualization: Bland–Altman plots
# ============================================================
def plot_bland_altman(pred_file_dict, folders):
    palette = {"knn": "#4C72B0", "linear": "#55A868", "rf": "#C44E52", "xgb": "#8172B2"}

    for model, df in pred_file_dict.items():
        for seg in ["lateral", "femoral", "medial"]:
            tmp = df[[f"true_{seg}", f"{seg}_pred"]].dropna()
            true_vals = tmp[f"true_{seg}"]
            pred_vals = tmp[f"{seg}_pred"]

            mean_vals = (true_vals + pred_vals) / 2
            diff = pred_vals - true_vals
            md = diff.mean()
            sd = diff.std()

            plt.figure(figsize=(7, 6))
            plt.scatter(mean_vals, diff, color=palette[model], alpha=0.6)
            plt.axhline(md, color="red", linestyle="--")
            plt.axhline(md + 1.96 * sd, color="gray", linestyle="--")
            plt.axhline(md - 1.96 * sd, color="gray", linestyle="--")

            plt.xlabel("Mean (cm)")
            plt.ylabel("Difference (cm)")
            plt.title(f"{model.upper()} — {seg.capitalize()}")

            out = folders["bland"] / f"{model}_{seg}.png"
            plt.savefig(out, dpi=350, bbox_inches="tight")
            plt.close()
            print("Saved:", out)


# ============================================================
# Cross-backbone comparison utilities
# ============================================================
def load_summary_backbones():
    """Load summary files for both ResNet18 and ResNet50."""
    paths = {
        "resnet18": Path(RESULT_DIR) / "all_models_segment_summary_resnet18.xlsx",
        "resnet50": Path(RESULT_DIR) / "all_models_segment_summary_resnet50.xlsx"
    }
    res = {}
    for k, p in paths.items():
        if p.exists():
            df = pd.read_excel(p)
            df["Backbone"] = k
            res[k] = df
        else:
            print(f"Missing file: {p}")
    return res


# ============================================================
# Cross-backbone bar chart comparison
# ============================================================
def plot_grouped_bars(summary_df, folders):
    palette = {"knn": "#4C72B0", "linear": "#55A868", "rf": "#C44E52", "xgb": "#8172B2"}
    segments = ["Lateral", "Femoral", "Medial"]
    metrics = ["MAE", "RMSE", "R2"]
    models = summary_df["Model"].unique()

    for metric in metrics:
        plt.figure(figsize=(9, 6))
        bar_width = 0.18
        x = np.arange(len(segments))

        for j, m in enumerate(models):
            d = summary_df[summary_df["Model"] == m]
            plt.bar(x + j * bar_width - 1.5 * bar_width,
                    d[metric].values, width=bar_width,
                    color=palette[m], label=m.upper())

        plt.xticks(x, segments)
        plt.ylabel(metric)
        plt.title(f"Comparison of {metric}")
        plt.legend()

        out = folders["grouped"] / f"{metric}.png"
        plt.savefig(out, dpi=350, bbox_inches="tight")
        plt.close()
        print("Saved:", out)

def plot_comparison_bars(summary_dict):
    out_dir = Path(RESULT_DIR) / "comparison" / "bars"
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = ["Lateral", "Femoral", "Medial"]
    metrics = ["MAE", "RMSE", "R2"]
    palette = {"resnet18": "#4C72B0", "resnet50": "#C44E52"}

    all_models = sorted(set(
        pd.concat([summary_dict["resnet18"], summary_dict["resnet50"]])["Model"].unique()
    ))

    for metric in metrics:
        for m in all_models:
            plt.figure(figsize=(9, 6))
            x = np.arange(len(segments))
            bar_width = 0.35

            for i, backbone in enumerate(["resnet18", "resnet50"]):
                df = summary_dict.get(backbone)
                if df is None:
                    continue
                subset = df[df["Model"] == m]
                plt.bar(x + i * bar_width - bar_width/2,
                        subset[metric].values, width=bar_width,
                        color=palette[backbone], label=backbone.upper())

            plt.xticks(x, segments)
            plt.ylabel(metric)
            plt.title(f"{m.upper()} — ResNet18 vs ResNet50 — {metric}")
            plt.legend()

            out = out_dir / f"comparison_{m}_{metric}.png"
            plt.savefig(out, dpi=350, bbox_inches="tight")
            plt.close()
            print("Saved:", out)



# ============================================================
# Cross-backbone scatter comparison
# ============================================================
def plot_comparison_scatter(pred18, pred50):
    out_dir = Path(RESULT_DIR) / "comparison" / "scatter"
    out_dir.mkdir(parents=True, exist_ok=True)

    palette = {"resnet18": "#4C72B0", "resnet50": "#C44E52"}

    for model in pred18.keys():
        df18 = pred18[model]
        df50 = pred50.get(model)
        if df50 is None:
            continue

        for seg in ["lateral", "femoral", "medial"]:
            plt.figure(figsize=(7, 6))

            tmp18 = df18.dropna(subset=[f"true_{seg}", f"{seg}_pred"])
            sns.scatterplot(
                x=tmp18[f"true_{seg}"],
                y=tmp18[f"{seg}_pred"],
                color=palette["resnet18"],
                label="ResNet18"
            )

            tmp50 = df50.dropna(subset=[f"true_{seg}", f"{seg}_pred"])
            sns.scatterplot(
                x=tmp50[f"true_{seg}"],
                y=tmp50[f"{seg}_pred"],
                color=palette["resnet50"],
                label="ResNet50"
            )

            plt.plot([0, 2], [0, 2], "k--", alpha=0.6)

            plt.xlabel("Manual (cm)")
            plt.ylabel("Predicted (cm)")
            plt.title(f"{model.upper()} — {seg.capitalize()} (18 vs 50)")
            plt.legend()

            out = out_dir / f"{model}_{seg}_comparison.png"
            plt.savefig(out, dpi=350, bbox_inches="tight")
            plt.close()
            print("Saved:", out)


# ============================================================
# Cross-backbone residual comparison
# ============================================================
def plot_comparison_residuals(pred18, pred50):
    out_dir = Path(RESULT_DIR) / "comparison" / "residuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    palette = {"resnet18": "#4C72B0", "resnet50": "#C44E52"}

    for model in pred18.keys():
        df18 = pred18[model]
        df50 = pred50.get(model)
        if df50 is None:
            continue

        residuals18 = []
        residuals50 = []

        for seg in ["lateral", "femoral", "medial"]:
            tmp18 = df18.dropna(subset=[f"true_{seg}", f"{seg}_pred"])
            tmp50 = df50.dropna(subset=[f"true_{seg}", f"{seg}_pred"])

            residuals18.extend(tmp18[f"{seg}_pred"] - tmp18[f"true_{seg}"])
            residuals50.extend(tmp50[f"{seg}_pred"] - tmp50[f"true_{seg}"])

        plt.figure(figsize=(8, 6))
        sns.histplot(residuals18, bins=25, kde=True, color=palette["resnet18"], label="ResNet18", alpha=0.7)
        sns.histplot(residuals50, bins=25, kde=True, color=palette["resnet50"], label="ResNet50", alpha=0.7)

        plt.xlabel("Residual (cm)")
        plt.title(f"{model.upper()} Residual Comparison (18 vs 50)")
        plt.legend()

        out = out_dir / f"{model}_residual_comparison.png"
        plt.savefig(out, dpi=350, bbox_inches="tight")
        plt.close()
        print("Saved:", out)


# ============================================================
# Cross-backbone Bland–Altman comparison
# ============================================================
def plot_comparison_bland(pred18, pred50):
    out_dir = Path(RESULT_DIR) / "comparison" / "bland_altman"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in pred18.keys():
        df18 = pred18[model]
        df50 = pred50.get(model)
        if df50 is None:
            continue

        for seg in ["lateral", "femoral", "medial"]:
            plt.figure(figsize=(7, 6))

            merged = df18[["GroupID", f"{seg}_pred"]].merge(
                df50[["GroupID", f"{seg}_pred"]],
                on="GroupID",
                suffixes=("_18", "_50")
            ).dropna()

            diff = merged[f"{seg}_pred_18"] - merged[f"{seg}_pred_50"]
            mean_vals = (merged[f"{seg}_pred_18"] + merged[f"{seg}_pred_50"]) / 2
            md = diff.mean()
            sd = diff.std()

            plt.scatter(mean_vals, diff, color="#8172B2", alpha=0.6)
            plt.axhline(md, linestyle="--", color="red")
            plt.axhline(md + 1.96 * sd, linestyle="--", color="gray")
            plt.axhline(md - 1.96 * sd, linestyle="--", color="gray")

            plt.xlabel("Mean Predicted (cm)")
            plt.ylabel("Difference (18 - 50)")
            plt.title(f"Bland-Altman Comparison — {model.upper()} — {seg.capitalize()}")

            out = out_dir / f"{model}_{seg}_bland_comparison.png"
            plt.savefig(out, dpi=350, bbox_inches="tight")
            plt.close()
            print("Saved:", out)

# ============================================================
# Compact cross-backbone bar charts: 2 figures (MAE, RMSE)
# ============================================================
def plot_comparison_bars_compact(summary_dict, metrics=("MAE", "RMSE", "R2")):
    out_dir = Path(RESULT_DIR) / "comparison" / "bars"
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
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.4,
        'grid.alpha': 0.3
    })

    models = ["knn", "linear", "rf", "xgb"]
    model_titles = {"knn": "KNN", "linear": "Linear", "rf": "Random Forest", "xgb": "XGBoost"}

    segments = ["lateral", "femoral", "medial"]
    segment_display = {"lateral": "Lateral", "femoral": "Femoral", "medial": "Medial"}
    backbones = ["resnet18", "resnet50"]

    # 使用你提供的柔和色系
    colors = {
        "resnet18": "#699FDA",  # 柔和的蓝色
        "resnet50": "#F9A857",  # 柔和的橙色
    }

    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Segment" in df.columns:
            df["Segment"] = df["Segment"].str.lower().str.strip()
        for cand in ["R2", "r2", "R²", "r²"]:
            if cand in df.columns:
                df.rename(columns={cand: "R2"}, inplace=True)
                break
        return df

    def _add_value_labels(ax, rects, fmt="{:.3f}"):
        for r in rects:
            height = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, height + 0.001,
                    fmt.format(height),
                    ha="center", va="bottom", fontsize=8, color="black", 
                    fontweight='bold')

    for metric in metrics:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
        axes = axes.ravel()
        x = np.arange(len(segments))
        width = 0.35  # 稍微调整宽度

        for i, m in enumerate(models):
            ax = axes[i]

            rects_all = []
            labels_once = []

            for j, bb in enumerate(backbones):
                df = _norm(summary_dict[bb])
                sub = df[df["Model"] == m].set_index("Segment").reindex(segments)
                vals = sub[metric].values.astype(float)

                rects = ax.bar(
                    x + (j - 0.5) * width,
                    vals,
                    width=width,
                    color=colors[bb],
                    edgecolor="white",
                    linewidth=1.0,
                    alpha=0.85,
                    label=bb.upper() if i == 0 else None,
                    zorder=3,
                )
                _add_value_labels(ax, rects)
                rects_all.append(rects)

            # 设置x轴标签
            ax.set_xticks(x)
            ax.set_xticklabels([segment_display[s] for s in segments], fontweight='bold')
            
            # 设置标题和y轴标签
            ax.set_title(model_titles[m], fontsize=11, fontweight='bold', pad=10)
            
            # 设置y轴标签
            if metric == "MAE":
                ylabel = "MAE (mm)"
            elif metric == "RMSE":
                ylabel = "RMSE (mm)"
            elif metric == "R2":
                ylabel = "R² Score"
            else:
                ylabel = metric
                
            ax.set_ylabel(ylabel if i in (0, 2) else "", fontweight='bold')
            
            # 美化边框和网格
            for spine in ax.spines.values():
                spine.set_color('#666666')
                spine.set_linewidth(0.8)
                
            ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)

        # 添加图例
        handles, labels = axes[0].get_legend_handles_labels()
        # 修改图例标签显示
        backbone_display = {
            "resnet18": "ResNet-18",
            "resnet50": "ResNet-50"
        }
        display_labels = [backbone_display.get(label.lower(), label) for label in labels]
        
        fig.legend(handles, display_labels, 
                  loc="upper center", 
                  ncol=2, 
                  frameon=True,
                  fancybox=True,
                  shadow=False,
                  framealpha=0.9,
                  edgecolor='#CCCCCC',
                  bbox_to_anchor=(0.5, 1.02))

        # 设置整体标题
        metric_display = {
            "MAE": "Mean Absolute Error",
            "RMSE": "Root Mean Square Error", 
            "R2": "R² Score"
        }
        fig.suptitle(f"Cartilage Thickness Estimation - {metric_display.get(metric, metric)}", 
                    fontsize=14, fontweight='bold', y=0.95)

        fig.tight_layout(rect=[0, 0, 1, 0.93])  # 为上方标题留出空间
        out = out_dir / f"comparison_compact_{metric}.png"
        fig.savefig(out, dpi=350, bbox_inches="tight", facecolor='white', edgecolor='none')
        plt.close(fig)
        print("Saved:", out)



# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    for backbone in ["resnet18", "resnet50"]:
        print("Running:", backbone)

        folders = prepare_dirs(backbone)
        summary_df = load_summary(backbone)
        pred_file_dict = load_prediction_with_truth(backbone)

        plot_grouped_bars(summary_df, folders)

        plot_scatter(pred_file_dict, folders, backbone)
        plot_residuals(pred_file_dict, folders)
        plot_bland_altman(pred_file_dict, folders)

        print("Finished:", backbone)

    print("Generating ResNet18 vs ResNet50 comparison")
    summary_dict = load_summary_backbones()
    pred18 = load_prediction_with_truth("resnet18")
    pred50 = load_prediction_with_truth("resnet50")

    plot_comparison_bars(summary_dict)
    plot_comparison_scatter(pred18, pred50)
    plot_comparison_residuals(pred18, pred50)
    plot_comparison_bland(pred18, pred50)
    plot_comparison_bars_compact(summary_dict, metrics=("MAE","RMSE","R2"))

    print("All visualizations completed.")



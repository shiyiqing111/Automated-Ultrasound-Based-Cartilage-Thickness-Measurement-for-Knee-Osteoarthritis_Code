from pathlib import Path

# ====== 路径与通用配置 ======
# 可在命令行传参覆盖
LABELED_EXCEL = Path(r"C:\Users\Charlotte\Desktop\dissertation\US_new\Model\data\merged_annotation.xlsx")
IMAGES_DIR    = Path(r"C:\Users\Charlotte\Desktop\dissertation\US_new\High_quality_images")
BOOK_EXCEL    = Path(r"C:\Users\Charlotte\Desktop\dissertation\US_new\Book2.xlsx")


# ====== 单位换算与线性校正（按你的实验） ======
PIXEL_TO_CM   = 0.003985
LINEAR_SLOPE  = 1.0
LINEAR_INTERCEPT = 0.0

# ====== 特征提取 Backbone 选项 ======
# 可选：'resnet18', 'resnet50'
BACKBONES = ['resnet18']

# ====== 模型清单 ======
# 可选：'knn', 'rf', 'xgb', 'linear'
MODELS = ['knn', 'rf']

# ====== 运行参数 ======
BATCH_SIZE = 8
EMBED_CACHE_DIR = Path(__file__).resolve().parent / "embeddings"
PRED_DIR   = Path(__file__).resolve().parent / "predictions"
RESULT_DIR = Path(__file__).resolve().parent / "results"

RANDOM_STATE = 42
N_JOBS = -1  # sklearn 并行

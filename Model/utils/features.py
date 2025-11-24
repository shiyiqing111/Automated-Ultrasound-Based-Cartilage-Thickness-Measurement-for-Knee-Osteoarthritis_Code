from pathlib import Path
from typing import List
import torch
from torchvision import models, transforms as T
from PIL import Image
import numpy as np
import pandas as pd

def _build_backbone(name: str):
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        outdim = 512
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        outdim = 2048
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    m.fc = torch.nn.Identity()
    m.eval()
    return m, outdim

def _build_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

@torch.inference_mode()
def extract_embeddings(img_paths: List[Path], backbone_name: str, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_idx = cache_dir / f"{backbone_name}_index.csv"
    cache_npy = cache_dir / f"{backbone_name}_embeddings.npy"

    if cache_idx.exists() and cache_npy.exists():
        idx_df = pd.read_csv(cache_idx)
        feats  = np.load(cache_npy)
    else:
        idx_df = pd.DataFrame(columns=["Filename","path"])
        feats  = np.zeros((0, 512 if backbone_name=="resnet18" else 2048), dtype=np.float32)

    cached_names = set(idx_df["Filename"].tolist())
    requested = [p.name for p in img_paths]
    need_compute = [p for p in img_paths if p.name not in cached_names]

    if need_compute:
        print(f"[INFO] New images to compute: {len(need_compute)}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, outdim = _build_backbone(backbone_name)
        model = model.to(device)
        tfm = _build_transform()

        new_feats = []
        new_rows  = []
        for p in need_compute:
            try:
                img = Image.open(p).convert("RGB")
                x = tfm(img).unsqueeze(0).to(device)
                f = model(x).view(1, -1).cpu().numpy()[0]
            except Exception:
                f = np.zeros(outdim, dtype=np.float32)
            new_feats.append(f)
            new_rows.append({"Filename": p.name, "path": str(p)})

        feats = np.concatenate([feats, np.asarray(new_feats, dtype=np.float32)], axis=0)
        idx_df = pd.concat([idx_df, pd.DataFrame(new_rows)], ignore_index=True)
        idx_df.to_csv(cache_idx, index=False)
        np.save(cache_npy, feats)
        print(f"[INFO] Cache updated: total {len(idx_df)} images.")

    name_to_idx = {n:i for i,n in enumerate(idx_df["Filename"].tolist())}
    out_feats = np.stack([feats[name_to_idx[p.name]] for p in img_paths])
    idx_out = pd.DataFrame({"Filename": requested, "path": [str(p) for p in img_paths]})
    return idx_out, out_feats

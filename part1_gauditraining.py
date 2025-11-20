#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-K GAUDI Training (Joint + MD-only + QSM-only)
===================================================
Stable VAE: KL ON with small weight, GPU-safe batch handling,
latent extraction works, no CPU/CUDA mismatches.
"""

import os
import sys
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.realpath(__file__))
GAUDI_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
COLUMNS_ROOT = SCRIPT_DIR
GRAPHS_BASE  = os.path.join(COLUMNS_ROOT, "graphs_knn")
sys.path.append(GAUDI_ROOT)

from components import GraphEncoder, GraphDecoder
from applications import VariationalGraphAutoEncoder
import deeplay as dl

# ---------------------------
# Training settings
# ---------------------------
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 8
EPOCHS      = 50
LATENT_DIM  = 8
DEBUG_SIMPLE = False

mode_to_idx = {
    "CORR":   (0, 0),
    "EUCLID": (1, 1),
    "WASS":   (2, 2),
}

K_LIST = [4, 6, 8, 10, 12, 20, 25, 30, 50] # [10, 20, 30, 50]



import torch.nn as nn
import torch

# UNIVERSAL PATCH: wrap Linear to always follow input device
_old_linear_forward = nn.Linear.forward

def _linear_forward_device_safe(self, input):
    if self.weight.device != input.device:
        self.to(input.device)
    return _old_linear_forward(self, input)

nn.Linear.forward = _linear_forward_device_safe
print("🔥 Patched nn.Linear.forward for device safety")

# ================================================================
# DEVICE FIX
# ================================================================
def move_batch_to_device(batch, device):
    """
    PyG Batch fix for Deeplay:
    - If batch has .to(), call batch.to(device)
    - If dict, move all tensors
    """
    if hasattr(batch, "to") and not isinstance(batch, dict):
        return batch.to(device)

    if isinstance(batch, dict):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        return batch

    return batch


def force_model_to_device(model, device):
    """
    Recursively push *ALL* GAUDI/Deeplay submodules to GPU.
    Deeplay sometimes leaves Linear/MPN blocks on CPU even after model.to().
    """
    for name, module in model.named_modules():
        for p in module.parameters(recurse=False):
            p.data = p.data.to(device)
            if p._grad is not None:
                p._grad.data = p._grad.data.to(device)
        # buffers: running_mean, running_var, cluster masks, etc.
        for bname, buf in module.named_buffers(recurse=False):
            module._buffers[bname] = buf.to(device)
    return model

def warmup_and_fix_device(model, nX, nE, device):
    """
    Deeplay creates many modules lazily on the FIRST forward pass.
    We push a dummy graph through the model once,
    THEN move all parameters/buffers to the GPU again.
    """
    import torch
    from torch_geometric.data import Data, Batch

    # ----- build a dummy 3-node graph -----
    x = torch.randn(3, nX)
    edge_index = torch.tensor([[0,1,2],[1,2,0]])
    edge_attr = torch.randn(edge_index.shape[1], nE)

    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=[x,edge_attr])
    batch = Batch.from_data_list([g]).to(device)

    # ----- first forward pass: initializes all lazy Deeplay modules -----
    with torch.no_grad():
        _ = model(batch)

    # ----- force all submodules to GPU AGAIN -----
    model = force_model_to_device(model, device)
    return model

# =====================================================================
# LOSS CALLBACK — SAVE TRAINING CURVE
# =====================================================================
class LossPlotCallback(Callback):
    def __init__(self, save_dir, tag):
        super().__init__()
        self.save_dir = save_dir
        self.tag = tag
        self.losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        metrics = trainer.logged_metrics
        for k in ["train_total_loss_step", "train_loss"]:
            if k in metrics:
                self.losses.append(metrics[k].item())
                break

    def on_train_end(self, trainer, pl_module):
        if not self.losses:
            print(f"[WARN] No loss values logged for {self.tag}")
            return

        plt.figure(figsize=(10, 4))
        plt.plot(self.losses, linewidth=2)
        plt.title(f"Training Loss — {self.tag}")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.grid(True)

        out_path = os.path.join(self.save_dir, f"training_curve_{self.tag}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"📈 Saved loss curve → {out_path}")


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================
def load_graph_list(path):
    graphs = torch.load(path, map_location="cpu")
    print(f"Loaded {len(graphs)} graphs from {path}")
    return graphs


def zscore_features(graphs):
    for g in graphs:
        mu = g.x.mean(0, keepdim=True)
        sd = g.x.std(0, keepdim=True)
        sd[sd == 0] = 1.0
        g.x = (g.x - mu) / sd
    return graphs


def zscore_edges(graphs):
    for g in graphs:
        if getattr(g, "edge_attr", None) is None:
            continue
        mu = g.edge_attr.mean(0, keepdim=True)
        sd = g.edge_attr.std(0, keepdim=True)
        sd[sd == 0] = 1.0
        g.edge_attr = (g.edge_attr - mu) / sd
    return graphs


def extract_single_channel_graphs(md_all, qsm_all, md_idx, qsm_idx):
    """
    Extract one metric channel per graph.
    """
    md_list, qsm_list = [], []

    for g_md, g_qsm in zip(md_all, qsm_all):
        md_scalar = g_md.edge_attr[:, md_idx]
        g_md_sc = Data(
            x = g_md.x.clone(),
            edge_index = g_md.edge_index.clone(),
            edge_attr = md_scalar.unsqueeze(1).clone(),
        )
        md_list.append(g_md_sc)

        qsm_scalar = g_qsm.edge_attr[:, qsm_idx]
        g_qsm_sc = Data(
            x = g_qsm.x.clone(),
            edge_index = g_qsm.edge_index.clone(),
            edge_attr = qsm_scalar.unsqueeze(1).clone(),
        )
        qsm_list.append(g_qsm_sc)

    return md_list, qsm_list


def fuse_graphs(md_graphs, qsm_graphs):
    fused = []
    for ga, gb in zip(md_graphs, qsm_graphs):

        ew = (ga.edge_attr + gb.edge_attr) / 2.0

        g = Data(
            x = torch.cat([ga.x, gb.x], dim=1),
            edge_index = ga.edge_index.clone(),
            edge_attr = ew.clone(),
        )
        fused.append(g)

    return fused


# =====================================================================
# BUILD STABLE GAUDI MODEL
# =====================================================================
def build_model(n_features, e_features):
    # ---------------------------------------------------------
    # 1. Create encoder / decoder
    # ---------------------------------------------------------
    encoder = GraphEncoder(
        hidden_features = 96,
        num_blocks      = 3,
        num_clusters    = [20, 5, 1],
        thresholds      = [1/19, 1/5, None],
    )

    decoder = GraphDecoder(
        hidden_features   = 96,
        num_blocks        = 3,
        output_node_dim   = n_features,
        output_edge_dim   = e_features,
    )

    # ---------------------------------------------------------
    # 2. Select VAE configuration
    # ---------------------------------------------------------
    if DEBUG_SIMPLE:
        vgae = VariationalGraphAutoEncoder(
            encoder    = encoder,
            decoder    = decoder,
            latent_dim = LATENT_DIM,
            optimizer  = dl.Adam(lr=1e-4),
            alpha      = 0.0,
            beta       = 0.0,
            gamma      = 0.0,
            delta      = 1.0,
        )
    else:
        vgae = VariationalGraphAutoEncoder(
            encoder    = encoder,
            decoder    = decoder,
            latent_dim = LATENT_DIM,
            optimizer  = dl.Adam(lr=1e-4),
            alpha      = 2.0,     # MinCut weight
            beta       = 1e-6,    # KL stabilizer
            gamma      = 3.0,     # orthogonality
            delta      = 1.0,     # reconstruction
        )

    # ---------------------------------------------------------
    # 3. Build Deeplay model (creates high-level modules)
    # ---------------------------------------------------------
    model = vgae.build()

    # Move initial modules to GPU now
    model = model.to(DEVICE)

    # ---------------------------------------------------------
    # 4. WARM-UP forward pass (creates lazy CPU layers)
    # ---------------------------------------------------------
    model = warmup_and_fix_device(model, n_features, e_features, DEVICE)

    # ---------------------------------------------------------
    # 5. FINAL full recursive to(device)
    # (forces all newly-created layers to GPU)
    # ---------------------------------------------------------
    model = force_model_to_device(model, DEVICE)

    return model



# =====================================================================
# TRAIN FUNCTION
# =====================================================================
def train_gaudi(graphs, out_dir, tag):

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[TRAIN {tag}] → {len(graphs)} graphs")

    # reconstruction targets
    for g in graphs:
        g.y = [g.x, g.edge_attr]

    nX = graphs[0].x.shape[1]
    nE = graphs[0].edge_attr.shape[1]

    model = build_model(nX, nE)
    print("Model first param device:", next(model.parameters()).device)

    loader = DataLoader(
        graphs,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = True,
    )

    plot_cb = LossPlotCallback(out_dir, tag)

    trainer = dl.Trainer(
        max_epochs        = EPOCHS,
        accelerator       = "gpu" if torch.cuda.is_available() else "cpu",
        devices           = [0] if torch.cuda.is_available() else None,
        log_every_n_steps = 1,
        callbacks         = [plot_cb],
    )

    trainer.fit(model, loader)

    # ----------------------------------------
    # EXTRACT LATENT Z
    # ----------------------------------------
    model.eval()

    zs = []
    with torch.no_grad():
        for g in graphs:
            batch = Batch.from_data_list([g])
            batch = move_batch_to_device(batch, DEVICE)   # <<< GPU FIX
            out = model(batch)
            zs.append(out["mu"].cpu().numpy())

    lat = np.vstack(zs)
    out_path = os.path.join(out_dir, f"latent_final_{tag}.npy")
    np.save(out_path, lat)
    print(f"Saved latent → {out_path}")
    return lat


# =====================================================================
# MAIN LOOP
# =====================================================================
if __name__ == "__main__":
    print("DEVICE =", DEVICE)
    print("Graphs at:", GRAPHS_BASE)

    for K in K_LIST:
        print("\n" + "="*80)
        print(f"🔥 TRAINING for K = {K}")
        print("="*80)

        graph_dir = os.path.join(GRAPHS_BASE, f"k{K}")
        md_path   = os.path.join(graph_dir, f"md_shared_knn_k{K}_corr_euclid_wass.pt")
        qsm_path  = os.path.join(graph_dir, f"qsm_shared_knn_k{K}_corr_euclid_wass.pt")

        md_all  = load_graph_list(md_path)
        qsm_all = load_graph_list(qsm_path)

        latent_root = os.path.join(COLUMNS_ROOT, f"latent_k{K}")
        os.makedirs(latent_root, exist_ok=True)

        for metric, (md_idx, qsm_idx) in mode_to_idx.items():
            print("\n" + "-"*60)
            print(f"Metric: {metric}")
            print("-"*60)

            # 1) Channel selection
            md_sc, qsm_sc = extract_single_channel_graphs(md_all, qsm_all, md_idx, qsm_idx)

            # 2) Normalization
            md_sc  = zscore_features(md_sc)
            md_sc  = zscore_edges(md_sc)

            qsm_sc = zscore_features(qsm_sc)
            qsm_sc = zscore_edges(qsm_sc)

            # 3) Fusion
            fused = fuse_graphs(md_sc, qsm_sc)

            # 4) Train 3 models
            train_gaudi(
                fused,
                os.path.join(latent_root, f"latent_epochs_Joint_{metric}"),
                tag=f"Joint_{metric}",
            )

            train_gaudi(
                md_sc,
                os.path.join(latent_root, f"latent_epochs_MD_{metric}"),
                tag=f"MD_{metric}",
            )

            train_gaudi(
                qsm_sc,
                os.path.join(latent_root, f"latent_epochs_QSM_{metric}"),
                tag=f"QSM_{metric}",
            )

    print("\n🎉 DONE — All GAUDI runs completed!\n")

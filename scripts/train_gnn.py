"""
train_gnn.py
============
Train Graph Neural Networks (GraphSAGE, GAT) for:
  1. Node classification — Mule/SAR account detection

Uses pre-computed graph data from build_graph.py.
Pure-PyTorch implementation (no PyG dependency).

A100 GPU Optimizations:
  - TensorFloat-32 (TF32) for faster matmuls
  - Automatic Mixed Precision (AMP) via torch.cuda.amp
  - Fused Adam optimizer
  - torch.compile (Linux only, GraphSAGE only — GAT attention 
    patterns cause OOM under Inductor due to edge-level tensor
    materialization)
  - Gradient checkpointing for GAT to reduce memory

Outputs:
  - models/graphsage_node.pt       (model checkpoint + config + normalization params)
  - models/gat_node.pt             (model checkpoint + config + normalization params)
  - results/gnn_results.json       (all metrics)
  - results/training_curves.png    (loss & PR-AUC over epochs)
  - results/model_comparison.png   (bar chart: GNN vs baselines)

Usage:
    python scripts/train_gnn.py                          # train both
    python scripts/train_gnn.py --model graphsage        # GraphSAGE only
    python scripts/train_gnn.py --model gat              # GAT only
    python scripts/train_gnn.py --epochs 100 --hidden 256
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.checkpoint import checkpoint as grad_checkpoint

try:
    from sklearn.metrics import (
        classification_report, precision_recall_curve, auc,
        f1_score, roc_auc_score, average_precision_score, confusion_matrix
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "amlsim_v1")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── DATA LOADING ────────────────────────────────────────────────────────────

def load_split(split):
    """Load graph arrays for one split."""
    gd = os.path.join(BASE_DIR, split, "graph_data")
    data = {}
    for name in ["account_features", "account_labels", "edge_features", "edge_labels",
                  "transfer_edge_index", "same_wallet_edge_index",
                  "wallet_edge_index", "vpa_edge_index", "atm_edge_index",
                  "bank_edge_index", "login_edge_index", "shared_device_edge_index"]:
        path = os.path.join(gd, f"{name}.npy")
        if os.path.exists(path):
            data[name] = np.load(path)
    return data


def prepare_node_data(train_d, val_d, test_d):
    """Prepare node classification tensors with combined adjacency."""
    X_train = torch.tensor(train_d["account_features"], dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(train_d["account_labels"], dtype=torch.long).to(DEVICE)
    X_val = torch.tensor(val_d["account_features"], dtype=torch.float32).to(DEVICE)
    y_val = torch.tensor(val_d["account_labels"], dtype=torch.long).to(DEVICE)
    X_test = torch.tensor(test_d["account_features"], dtype=torch.float32).to(DEVICE)
    y_test = torch.tensor(test_d["account_labels"], dtype=torch.long).to(DEVICE)

    # Handle NaN/Inf
    for X in [X_train, X_val, X_test]:
        X[torch.isnan(X)] = 0
        X[torch.isinf(X)] = 0

    # Normalize features
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0).clamp(min=1e-6)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Build combined adjacency from transfer + same_wallet edges
    def build_edges(d):
        edges_list = []
        for key in ["transfer_edge_index", "same_wallet_edge_index"]:
            if key in d and d[key].shape[1] > 0:
                edges_list.append(torch.tensor(d[key], dtype=torch.long))
        if edges_list:
            return torch.cat(edges_list, dim=1).to(DEVICE)
        return torch.tensor(d["transfer_edge_index"], dtype=torch.long).to(DEVICE)

    return {
        "X_train": X_train, "y_train": y_train, "edge_index_train": build_edges(train_d),
        "X_val": X_val, "y_val": y_val, "edge_index_val": build_edges(val_d),
        "X_test": X_test, "y_test": y_test, "edge_index_test": build_edges(test_d),
        "mean": mean, "std": std
    }


# ─── MODEL DEFINITIONS ──────────────────────────────────────────────────────

class SAGEConv(nn.Module):
    """GraphSAGE convolution layer (mean aggregation)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_self = nn.Linear(in_dim, out_dim)
        self.linear_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.size(0)

        # Mean aggregation of neighbor features
        neigh_sum = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
        neigh_count = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)
        neigh_sum.index_add_(0, dst, x[src])
        neigh_count.index_add_(0, dst, torch.ones(src.size(0), 1, device=x.device, dtype=x.dtype))
        neigh_mean = neigh_sum / neigh_count.clamp(min=1)

        out = self.linear_self(x) + self.linear_neigh(neigh_mean)
        return out


class GATConvLayer(nn.Module):
    """Memory-efficient Graph Attention Network convolution layer.
    
    Key difference from standard GAT:  we process edges in chunks to
    avoid allocating a single (E, H, D) tensor which OOMs on large graphs.
    """
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        E = src.size(0)

        h = self.W(x).view(N, self.num_heads, self.head_dim)  # (N, H, D)

        # Compute attention scores
        score_src = (h * self.a_src).sum(dim=-1)  # (N, H)
        score_dst = (h * self.a_dst).sum(dim=-1)  # (N, H)

        e = self.leaky_relu(score_src[src] + score_dst[dst])  # (E, H)

        # Softmax over neighbors (numerically stable)
        e_max = torch.zeros(N, self.num_heads, device=x.device, dtype=e.dtype)
        e_max.index_reduce_(0, dst, e, 'amax', include_self=True)
        e_exp = torch.exp(e - e_max[dst])
        e_sum = torch.zeros(N, self.num_heads, device=x.device, dtype=e.dtype)
        e_sum.index_add_(0, dst, e_exp)
        alpha = self.dropout(e_exp / e_sum[dst].clamp(min=1e-9))  # (E, H)

        # MEMORY-EFFICIENT aggregation: process in chunks to avoid
        # allocating the full (E, H, D) tensor which causes OOM.
        # Use float32 for accumulation to avoid AMP dtype mismatches.
        CHUNK = 500_000
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device, dtype=torch.float32)
        for i in range(0, E, CHUNK):
            j = min(i + CHUNK, E)
            src_chunk = src[i:j]
            dst_chunk = dst[i:j]
            alpha_chunk = alpha[i:j]            # (chunk, H)
            msg_chunk = h[src_chunk].float() * alpha_chunk.unsqueeze(-1).float()  # (chunk, H, D) fp32
            out.index_add_(0, dst_chunk, msg_chunk)

        return out.reshape(N, -1)  # (N, H*D) — always float32


class GraphSAGEModel(nn.Module):
    """2-layer GraphSAGE for node classification."""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.dropout(h)

        out = self.classifier(h)
        return out

    def get_embeddings(self, x, edge_index):
        """Return node embeddings (for downstream use / frontend)."""
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        return h


class GATModel(nn.Module):
    """2-layer GAT for node classification (memory-efficient)."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConvLayer(in_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.conv2 = GATConvLayer(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.use_checkpointing = True  # Enable gradient checkpointing by default

    def _conv_block_1(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = self.dropout(h)
        return h

    def _conv_block_2(self, h, edge_index):
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)
        h = self.dropout(h)
        return h

    def forward(self, x, edge_index):
        if self.training and self.use_checkpointing:
            # Gradient checkpointing: recompute activations during backward
            # to save ~50% GPU memory at cost of ~30% more compute
            h = grad_checkpoint(self._conv_block_1, x, edge_index, use_reentrant=False)
            h = grad_checkpoint(self._conv_block_2, h, edge_index, use_reentrant=False)
        else:
            h = self._conv_block_1(x, edge_index)
            h = self._conv_block_2(h, edge_index)

        out = self.classifier(h)
        return out

    def get_embeddings(self, x, edge_index):
        """Return node embeddings (for downstream use / frontend)."""
        h = self._conv_block_1(x, edge_index)
        h = self._conv_block_2(h, edge_index)
        return h


# ─── TRAINING ────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob):
    """Compute comprehensive metrics."""
    if not HAS_SKLEARN:
        return {"note": "sklearn not available"}

    metrics = {}
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["accuracy"] = report["accuracy"]
    metrics["f1_macro"] = report["macro avg"]["f1-score"]
    metrics["f1_weighted"] = report["weighted avg"]["f1-score"]

    if "1" in report:
        metrics["precision_sar"] = report["1"]["precision"]
        metrics["recall_sar"] = report["1"]["recall"]
        metrics["f1_sar"] = report["1"]["f1-score"]

    if y_prob is not None:
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            metrics["pr_auc"] = float(auc(recall, precision))
            metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    n_pos = int(y_true.sum())
    for k_mult in [1, 2, 5]:
        k = n_pos * k_mult
        if k <= len(y_prob) and y_prob is not None:
            top_k_idx = np.argsort(y_prob)[-k:]
            recall_at_k = y_true[top_k_idx].sum() / max(n_pos, 1)
            metrics[f"recall@{k_mult}x"] = float(recall_at_k)

    return metrics


def train_epoch(model, X, y, edge_index, optimizer, criterion, scaler, batch_size=65536):
    """Train one epoch with AMP (full-graph forward, optional node-batched loss)."""
    model.train()

    with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
        logits = model(X, edge_index)
        if batch_size >= X.size(0):
            loss = criterion(logits, y)
        else:
            idx = torch.randint(0, X.size(0), (batch_size,), device=X.device)
            loss = criterion(logits[idx], y[idx])

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    return loss.item()


@torch.no_grad()
def evaluate(model, X, y, edge_index):
    """Evaluate model on a split."""
    model.eval()
    with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
        logits = model(X, edge_index)
        probs = F.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    y_np = y.cpu().numpy()
    metrics = compute_metrics(y_np, preds, probs)
    return metrics, probs


def train_model(model_name, model, data, epochs=100, lr=0.005, patience=15,
                use_compile=False):
    """Full training loop with early stopping, AMP, and training history.
    
    Returns:
        test_metrics: dict of final test metrics
        model: the trained model (best checkpoint)
        history: dict of training curves {epoch, loss, val_pr_auc, val_f1_sar}
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"  Device:      {DEVICE}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:  {n_params:,}")
    if DEVICE.type == "cuda":
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory:  {mem_gb:.1f} GB")

    # torch.compile — only for GraphSAGE on Linux (GAT OOMs under Inductor)
    if use_compile and os.name != "nt":
        try:
            print("  Compiling model with torch.compile()...")
            model = torch.compile(model)
            print("  ✓ Compiled successfully")
        except Exception as e:
            print(f"  [WARN] torch.compile skipped: {e}")
    else:
        reason = "Windows (no Triton)" if os.name == "nt" else "disabled for this model"
        print(f"  torch.compile: skipped ({reason})")

    # Class weights for imbalanced data
    n_pos = (data["y_train"] == 1).sum().float()
    n_neg = (data["y_train"] == 0).sum().float()
    ratio = (n_neg / n_pos).item()
    weight = torch.tensor([1.0, ratio], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)
    print(f"  Class ratio: 1:{ratio:.1f} (SAR weight = {ratio:.1f})")

    # Optimizer
    use_fused = DEVICE.type == "cuda"
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4, fused=use_fused)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # Training state
    best_val_prauc = 0.0
    best_epoch = 0
    no_improve = 0
    best_state = None
    history = {"epoch": [], "loss": [], "val_pr_auc": [], "val_f1_sar": []}

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        loss = train_epoch(
            model, data["X_train"], data["y_train"],
            data["edge_index_train"], optimizer, criterion, scaler
        )

        # Evaluate every 5 epochs or last epoch
        if epoch % 5 == 0 or epoch == epochs:
            val_metrics, _ = evaluate(model, data["X_val"], data["y_val"], data["edge_index_val"])
            val_prauc = val_metrics.get("pr_auc", 0)
            val_f1 = val_metrics.get("f1_sar", 0)
            scheduler.step(val_prauc)

            history["epoch"].append(epoch)
            history["loss"].append(loss)
            history["val_pr_auc"].append(val_prauc)
            history["val_f1_sar"].append(val_f1)

            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Val PR-AUC: {val_prauc:.4f} "
                  f"| Val F1(SAR): {val_f1:.4f} | Time: {elapsed:.0f}s")

            if val_prauc > best_val_prauc:
                best_val_prauc = val_prauc
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience // 5:
                print(f"  ✓ Early stopping at epoch {epoch} (best: {best_epoch})")
                break

    total_time = time.time() - t_start
    print(f"  Training time: {total_time:.1f}s")

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # Final test evaluation
    test_metrics, test_probs = evaluate(model, data["X_test"], data["y_test"], data["edge_index_test"])
    test_metrics["best_epoch"] = best_epoch
    test_metrics["training_time_s"] = round(total_time, 1)
    test_metrics["n_params"] = n_params

    print(f"\n  === {model_name} Test Results ===")
    print(f"  Best epoch:   {best_epoch}")
    print(f"  PR-AUC:       {test_metrics.get('pr_auc', 0):.4f}")
    print(f"  ROC-AUC:      {test_metrics.get('roc_auc', 0):.4f}")
    print(f"  F1 (SAR):     {test_metrics.get('f1_sar', 0):.4f}")
    print(f"  Precision:    {test_metrics.get('precision_sar', 0):.4f}")
    print(f"  Recall:       {test_metrics.get('recall_sar', 0):.4f}")
    if "recall@1x" in test_metrics:
        print(f"  Recall@1x:    {test_metrics['recall@1x']:.4f}")
        print(f"  Recall@2x:    {test_metrics.get('recall@2x', 0):.4f}")
    cm = test_metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    print(f"    FN={cm[1][0]:,}  TP={cm[1][1]:,}")

    return test_metrics, model, history


def save_model_bundle(model, model_name, config, metrics, norm_params, save_dir):
    """Save model checkpoint + config for frontend integration.
    
    The saved bundle contains everything needed to reload the model:
      - model_state_dict: the trained weights
      - config: architecture hyperparameters (in_dim, hidden_dim, etc.)
      - norm: feature normalization params (mean, std)
      - metrics: evaluation metrics
      - model_class: string name of the model class
    """
    bundle = {
        "model_state_dict": model.state_dict(),
        "model_class": model_name,
        "config": config,
        "metrics": metrics,
        "norm": norm_params,
    }
    path = os.path.join(save_dir, f"{model_name.lower()}_node.pt")
    torch.save(bundle, path)
    print(f"  ✓ Model saved: {path}")
    return path


# ─── VISUALIZATION ───────────────────────────────────────────────────────────

def plot_training_curves(histories, save_dir):
    """Plot training loss and validation PR-AUC curves."""
    if not HAS_MPL:
        print("  [WARN] matplotlib not installed, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"GraphSAGE": "#2196F3", "GAT": "#FF5722"}

    # 1. Training Loss
    ax = axes[0]
    for name, h in histories.items():
        ax.plot(h["epoch"], h["loss"], "-o", color=colors.get(name, "#333"),
                label=name, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Validation PR-AUC
    ax = axes[1]
    for name, h in histories.items():
        ax.plot(h["epoch"], h["val_pr_auc"], "-o", color=colors.get(name, "#333"),
                label=name, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Validation PR-AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 3. Validation F1 (SAR)
    ax = axes[2]
    for name, h in histories.items():
        ax.plot(h["epoch"], h["val_f1_sar"], "-o", color=colors.get(name, "#333"),
                label=name, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 (SAR)")
    ax.set_title("Validation F1 (SAR class)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Training curves saved: {path}")


def plot_model_comparison(gnn_results, baseline_path, save_dir):
    """Bar chart comparing GNN models vs baselines."""
    if not HAS_MPL:
        return

    all_models = {}

    # Load baselines
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            bl = json.load(f)
        if "node_classification" in bl:
            for name, m in bl["node_classification"].items():
                all_models[f"BL:{name}"] = m

    # Add GNN results
    for name, m in gnn_results.items():
        all_models[f"GNN:{name}"] = m

    if not all_models:
        return

    metrics_to_plot = ["pr_auc", "roc_auc", "f1_sar", "recall_sar", "precision_sar"]
    labels = ["PR-AUC", "ROC-AUC", "F1(SAR)", "Recall", "Precision"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(all_models)

    colors_list = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#C62828", "#00796B"]
    for i, (model_name, m) in enumerate(all_models.items()):
        vals = [m.get(k, 0) for k in metrics_to_plot]
        offset = (i - len(all_models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model_name,
                      color=colors_list[i % len(colors_list)], alpha=0.85)
        # Add value labels
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: GNN vs Baselines (Node Classification)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Model comparison chart saved: {path}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    # ── A100 Optimization: Enable TF32 ──
    torch.set_float32_matmul_precision("high")
    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="Train GNN models (A100 optimized)")
    parser.add_argument("--model", choices=["graphsage", "gat", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=256,
                        help="Hidden dim for GraphSAGE. GAT automatically uses hidden//2 for memory safety.")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load data ──
    print("Loading graph data...")
    train_d = load_split("train")
    val_d = load_split("val")
    test_d = load_split("test")

    required = ["account_features", "account_labels", "transfer_edge_index"]
    for r in required:
        if r not in train_d:
            print(f"[ERROR] Missing {r}. Run build_graph.py first.")
            return

    print("Preparing tensors...")
    data = prepare_node_data(train_d, val_d, test_d)
    in_dim = data["X_train"].size(1)
    print(f"  Input dim: {in_dim}, Hidden: {args.hidden}")
    print(f"  Train edges: {data['edge_index_train'].size(1):,}")
    print(f"  Val edges:   {data['edge_index_val'].size(1):,}")
    print(f"  Test edges:  {data['edge_index_test'].size(1):,}")

    if DEVICE.type == "cuda":
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU memory after loading: {alloc:.2f} / {total:.1f} GB")

    all_results = {}
    all_histories = {}
    norm_params = {
        "mean": data["mean"].cpu().numpy().tolist(),
        "std": data["std"].cpu().numpy().tolist()
    }

    # ── Train GraphSAGE ──
    if args.model in ["graphsage", "both"]:
        sage_model = GraphSAGEModel(in_dim, args.hidden, 2, dropout=0.3).to(DEVICE)
        sage_metrics, sage_model, sage_history = train_model(
            "GraphSAGE", sage_model, data,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            use_compile=(not args.no_compile)  # compile for SAGE
        )
        all_results["graphsage"] = sage_metrics
        all_histories["GraphSAGE"] = sage_history

        # Save model bundle
        save_model_bundle(sage_model, "GraphSAGE",
                          {"in_dim": in_dim, "hidden_dim": args.hidden, "out_dim": 2, "dropout": 0.3},
                          sage_metrics, norm_params, MODELS_DIR)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ── Train GAT ──
    if args.model in ["gat", "both"]:
        # GAT uses half the hidden dim of GraphSAGE for memory safety
        # The multi-head attention creates (E, H, D) tensors which are huge
        gat_hidden = args.hidden // 2
        gat_heads = 4
        print(f"\n  GAT config: hidden={gat_hidden}, heads={gat_heads} "
              f"(reduced from {args.hidden} for memory safety)")

        gat_model = GATModel(in_dim, gat_hidden, 2, num_heads=gat_heads, dropout=0.3).to(DEVICE)

        # NEVER compile GAT — Inductor materializes the full (E, H, D) tensor
        gat_metrics, gat_model, gat_history = train_model(
            "GAT", gat_model, data,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            use_compile=False  # GAT + compile = OOM
        )
        all_results["gat"] = gat_metrics
        all_histories["GAT"] = gat_history

        # Save model bundle
        save_model_bundle(gat_model, "GAT",
                          {"in_dim": in_dim, "hidden_dim": gat_hidden, "out_dim": 2,
                           "num_heads": gat_heads, "dropout": 0.3},
                          gat_metrics, norm_params, MODELS_DIR)

    # ── Save results ──
    results_path = os.path.join(RESULTS_DIR, "gnn_results.json")
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\n✓ Results saved: {results_path}")

    # ── Generate visualizations ──
    print("\nGenerating visualizations...")
    if all_histories:
        plot_training_curves(all_histories, RESULTS_DIR)

    baseline_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    plot_model_comparison(all_results, baseline_path, RESULTS_DIR)

    # ── Summary comparison ──
    print(f"\n{'='*70}")
    print("FINAL MODEL COMPARISON — Node Classification (Mule/SAR Detection)")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'PR-AUC':>8} {'ROC-AUC':>8} {'F1(SAR)':>8} {'Recall':>8} {'Precision':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    # GNN models
    for name, m in all_results.items():
        print(f"  {'GNN:'+name:<25} {m.get('pr_auc', 0):>8.4f} {m.get('roc_auc', 0):>8.4f} "
              f"{m.get('f1_sar', 0):>8.4f} {m.get('recall_sar', 0):>8.4f} "
              f"{m.get('precision_sar', 0):>10.4f}")

    # Baseline comparison
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        if "node_classification" in baseline:
            print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
            for name, m in baseline["node_classification"].items():
                print(f"  {'BL:'+name:<25} {m.get('pr_auc', 0):>8.4f} {m.get('roc_auc', 0):>8.4f} "
                      f"{m.get('f1_sar', 0):>8.4f} {m.get('recall_sar', 0):>8.4f} "
                      f"{m.get('precision_sar', 0):>10.4f}")

    print(f"\n{'='*70}")
    print("Saved artifacts:")
    print(f"  Models:       {MODELS_DIR}/")
    print(f"  Results:      {results_path}")
    print(f"  Curves:       {RESULTS_DIR}/training_curves.png")
    print(f"  Comparison:   {RESULTS_DIR}/model_comparison.png")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

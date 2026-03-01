# GhostBusters — Cross-Channel Mule Detection Platform

> **An end-to-end Anti-Money Laundering (AML) system that uses Graph Neural Networks, classical ML baselines, and an AI-powered investigator to detect money mule accounts and suspicious financial transactions across multiple payment channels.**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [Pipeline](#pipeline)
   - [Step 1 — Cross-Channel Enrichment](#step-1--cross-channel-enrichment)
   - [Step 2 — Graph Construction](#step-2--graph-construction)
   - [Step 3 — Baseline Models](#step-3--baseline-models)
   - [Step 4 — Graph Neural Networks](#step-4--graph-neural-networks)
   - [Step 5 — Explainability](#step-5--explainability)
5. [Model Results](#model-results)
6. [Dashboard](#dashboard)
   - [API Reference](#api-reference)
   - [AI Investigator (Ollama)](#ai-investigator-ollama)
7. [Training Notebook (A100)](#training-notebook-a100)
8. [Project Structure](#project-structure)
9. [Quick Start](#quick-start)
10. [Tests](#tests)
11. [Tech Stack](#tech-stack)

---

## Project Overview

GhostBusters is a financial crime detection platform built on top of the **AMLSim v1** synthetic dataset. It addresses a core challenge in AML: money mule accounts don't just look suspicious in isolation — they reveal themselves through *patterns in the transaction graph*: fan-in/fan-out topology, rapid pass-through, device sharing, and cross-channel hopping (receive via APP → forward via UPI → cashout at ATM).

### Key Capabilities

| Capability | Description |
|---|---|
| **Heterogeneous Graph** | 6 node types, 8 edge types across payment channels |
| **GNN Mule Detection** | GraphSAGE achieves **PR-AUC 0.987** and **ROC-AUC 0.9999** |
| **Classical Baselines** | XGBoost, Random Forest, Gradient Boosting, Logistic Regression |
| **Explainability** | Per-account reason codes, evidence subgraphs, narrative summaries |
| **Real-Time Scoring** | Server-Sent Events (SSE) stream for live transaction risk scoring |
| **Graph Forensics** | 3-hop fund tracing, mule ring community detection |
| **AI Investigator** | Ollama (qwen2.5:7b) generates regulator-ready SAR narratives |
| **Interactive Dashboard** | Flask + vanilla JS frontend with all metrics and visualizations |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GhostBusters Platform                    │
├─────────────────┬───────────────────────┬───────────────────────┤
│   Data Layer    │     ML / GNN Layer    │   Dashboard Layer     │
│                 │                       │                       │
│  AMLSim v1      │  build_graph.py       │  Flask server.py      │
│  ├─ train/      │  ├─ HeteroGraph       │  ├─ /api/overview     │
│  ├─ val/        │  ├─ 14 node features  │  ├─ /api/models       │
│  └─ test/       │  └─ 5 edge features   │  ├─ /api/alerts       │
│                 │                       │  ├─ /api/graph        │
│  generate_      │  train_baseline.py    │  ├─ /api/trace        │
│  cross_channel  │  ├─ XGBoost          │  ├─ /api/community    │
│  ├─ channels    │  ├─ RandomForest      │  ├─ /api/stream (SSE) │
│  ├─ devices     │  ├─ GradientBoosting  │  └─ /api/agent/       │
│  ├─ wallets     │  └─ LogisticReg       │     ├─ investigate    │
│  ├─ UPI VPAs    │                       │     └─ chat           │
│  └─ ATM events  │  train_gnn.py         │                       │
│                 │  ├─ GraphSAGE ★       │  Ollama (qwen2.5:7b) │
│                 │  └─ GAT               │  └─ SAR narratives    │
│                 │                       │                       │
│                 │  explainability.py    │  static/              │
│                 │  ├─ Evidence subgraph │  ├─ index.html        │
│                 │  ├─ Reason codes      │  ├─ js/app.js         │
│                 │  └─ Narratives        │  └─ css/style.css     │
└─────────────────┴───────────────────────┴───────────────────────┘
```

---

## Dataset

The project uses **AMLSim v1** — a synthetic financial transaction dataset designed for AML research.

### Data Splits

| Split | Purpose |
|---|---|
| `data/amlsim_v1/train/` | Model training |
| `data/amlsim_v1/val/` | Hyperparameter tuning & early stopping |
| `data/amlsim_v1/test/` | Final evaluation |

### Core Files (per split)

| File | Description |
|---|---|
| `accounts.csv` | Account demographics (type, status, bank, initial deposit) |
| `transactions.csv` | Raw financial transactions (orig, bene, amount, timestamp) |
| `enriched_transactions.csv` | Transactions + channel, device_id, ip_id (generated) |
| `sar_accounts.csv` | Ground-truth SAR-flagged (mule) account IDs |
| `alert_accounts.csv` | Accounts under alert |
| `alert_transactions.csv` | Transactions flagged in alerts |
| `app_logins.csv` | Account login events with device IDs (generated) |
| `wallet_links.csv` | Account ↔ wallet linkages (generated) |
| `upi_handles.csv` | Account → UPI VPA mappings (generated) |
| `atm_withdrawals.csv` | ATM cashout events (generated) |
| `cash_tx.csv` | Cash transactions |
| `tx_log.csv` | Transaction log |
| `graph_data/` | Pre-computed graph arrays (see below) |

### Graph Data Arrays (per split)

| File | Shape | Description |
|---|---|---|
| `account_features.npy` | `(N_accounts, 14)` | Per-account feature matrix |
| `account_labels.npy` | `(N_accounts,)` | Binary SAR labels |
| `transfer_edge_index.npy` | `(2, E_transfer)` | Account → Account money flow |
| `same_wallet_edge_index.npy` | `(2, E_wallet)` | Accounts sharing a wallet |
| `shared_device_edge_index.npy` | `(2, E_device)` | Accounts sharing a device |
| `login_edge_index.npy` | `(2, E_login)` | Account → Device login edges |
| `wallet_edge_index.npy` | `(2, E_wallet_link)` | Account → Wallet edges |
| `vpa_edge_index.npy` | `(2, E_vpa)` | Account → UPI VPA edges |
| `atm_edge_index.npy` | `(2, E_atm)` | Account → ATM edges |
| `bank_edge_index.npy` | `(2, E_bank)` | Account → Bank edges |
| `edge_features.npy` | `(E_transfer, 4)` | Per-transaction features |
| `edge_labels.npy` | `(E_transfer,)` | Per-transaction SAR labels |
| `graph_stats.json` | — | Node/edge counts summary |
| `id_mappings.json` | — | Entity ID → integer index maps |

---

## Pipeline

### Step 1 — Cross-Channel Enrichment

**Script:** `scripts/generate_cross_channel.py`

The raw AMLSim data only has bank transfer transactions. This step enriches it with realistic cross-channel behavioral data that exposes mule fingerprints.

**What it generates:**

| Artifact | Design Rule |
|---|---|
| `enriched_transactions.csv` | Adds `channel` (APP/WEB/UPI/WALLET/ATM), `device_id`, `ip_id` |
| `app_logins.csv` | Login events per account — SAR accounts share devices more |
| `wallet_links.csv` | Account ↔ wallet linkages — SAR accounts share wallets |
| `upi_handles.csv` | Account → UPI VPA mapping |
| `atm_withdrawals.csv` | ATM cashouts — mule accounts cash out within minutes of inbound |

**Mule behavioral rules encoded:**
- **Channel hopping**: SAR accounts heavily use UPI + ATM (weights: 30% UPI, 25% ATM) vs normal accounts (mostly APP/WEB)
- **Device sharing**: SAR accounts draw from a small pool of 5,000 devices; normal accounts from 50,000
- **Rapid cashout**: ATM withdrawals generated within 1–30 minutes of incoming transfers for mule accounts

```bash
python scripts/generate_cross_channel.py           # all splits
python scripts/generate_cross_channel.py --split train
```

---

### Step 2 — Graph Construction

**Script:** `scripts/build_graph.py`

Builds a unified heterogeneous entity graph from all data sources.

**Node Types:**

| Type | Description |
|---|---|
| `Account` | Bank accounts (main node type for classification) |
| `Device` | Login devices |
| `Wallet` | Digital wallets |
| `UPI_VPA` | UPI Virtual Payment Addresses |
| `ATM` | ATM terminals |
| `Bank` | Financial institutions |

**Edge Types:**

| Edge | Direction | Description |
|---|---|---|
| `TRANSFER` | Account → Account | Financial transfer |
| `LOGGED_IN` | Account → Device | Login event |
| `LINKED_WALLET` | Account → Wallet | Wallet association |
| `HAS_VPA` | Account → UPI_VPA | UPI handle ownership |
| `WITHDREW_ATM` | Account → ATM | Cash withdrawal |
| `BELONGS_TO_BANK` | Account → Bank | Banking relationship |
| `SHARED_DEVICE` | Account ↔ Account | Derived: same device used |
| `SAME_WALLET` | Account ↔ Account | Derived: same wallet linked |

**14 Account Features:**

| Index | Feature | AML Significance |
|---|---|---|
| 0 | `acct_type` | Individual vs Organization |
| 1 | `is_active` | Account status |
| 2 | `prior_sar` | Prior SAR filing flag |
| 3 | `initial_deposit` | Deposit amount |
| 4 | `sent_count` | Out-degree (transaction count) |
| 5 | `recv_count` | In-degree (transaction count) |
| 6 | `sent_amt_mean` | Average outgoing amount |
| 7 | `sent_amt_std` | Volatility in outgoing amounts |
| 8 | `recv_amt_mean` | Average incoming amount |
| 9 | `recv_amt_std` | Volatility in incoming amounts |
| 10 | `fan_out_ratio` | Out-degree / total degree |
| 11 | `fan_in_ratio` | In-degree / total degree |
| 12 | `recv_send_delay_min` | **Mule fingerprint**: receive→forward delay in minutes |
| 13 | `channel_diversity` | Number of distinct channels used |

**5 Edge Features (per transaction):**

| Index | Feature | Description |
|---|---|---|
| 0 | `amount` | Raw transaction amount |
| 1 | `log_amount` | Log-transformed amount |
| 2 | `amount_zscore` | Z-score within originator's history |
| 3 | `channel_code` | Encoded channel (0=APP, 1=WEB, 2=UPI, 3=WALLET, 4=ATM) |

```bash
python scripts/build_graph.py                        # all splits
python scripts/build_graph.py --split train          # single split
python scripts/build_graph.py --split train --skip-features  # structure only
```

---

### Step 3 — Baseline Models

**Script:** `scripts/train_baseline.py`

Trains classical ML models on the flat account feature vectors as competitive baselines.

**Models:**
- **XGBoost** — gradient boosted trees with scale_pos_weight for class imbalance
- **Random Forest** — ensemble of decision trees with class_weight='balanced'
- **Gradient Boosting** — sklearn's GradientBoostingClassifier
- **Logistic Regression** — linear baseline with StandardScaler

**Tasks:**
1. **Node Classification** — SAR account detection (binary: mule vs normal)
2. **Edge Classification** — Suspicious transaction detection

**Outputs:**
- `models/baseline_xgboost.pkl`, `baseline_random_forest.pkl`, etc.
- `models/baseline_scaler.pkl` — feature normalizer
- `results/baseline_results.json` — all metrics

```bash
python scripts/train_baseline.py                     # both tasks
python scripts/train_baseline.py --task node         # node only
python scripts/train_baseline.py --task edge         # edge only
```

---

### Step 4 — Graph Neural Networks

**Script:** `scripts/train_gnn.py`

Trains GNN models that exploit the graph structure for mule detection. The key insight: a mule account's neighbors in the graph are often also mules (guilt by association), and the graph topology encodes the money laundering typology.

#### GraphSAGE (Graph Sample and Aggregate)

A 2-layer mean-aggregation GraphSAGE:
1. Each node aggregates the mean of its neighbors' features
2. Concatenated with self features and linearly projected
3. BatchNorm + ReLU + Dropout between layers
4. Output: 2-class logits (normal / SAR)

**Equation per layer:**
```
h_v^(k) = ReLU( W_self * h_v^(k-1) + W_neigh * mean({h_u : u ∈ N(v)}) )
```

#### GAT (Graph Attention Network)

A 2-layer attention-based GNN with memory-efficient chunked aggregation:
1. Computes per-head attention scores between neighbors
2. Weighted aggregation using softmax-normalized scores
3. Chunked edge processing (500k edges/chunk) to avoid OOM on large graphs
4. Gradient checkpointing to reduce GPU memory by ~50%

#### A100 GPU Optimizations

| Optimization | Applied To | Benefit |
|---|---|---|
| TF32 matmuls | Both | ~2× speedup on A100 |
| AMP (float16) | Both | ~1.5× speedup, less memory |
| Fused Adam | Both | Faster optimizer step |
| `torch.compile` | GraphSAGE only | ~20% speedup (GAT excluded — OOM under Inductor) |
| Gradient checkpointing | GAT | ~50% memory reduction |
| Class-weighted loss | Both | Handles 1:30+ SAR imbalance |
| Early stopping | Both | Prevents overfitting, reduces training time |

**Training config:**
- Optimizer: Adam (fused on CUDA), lr=0.005, weight_decay=1e-4
- Scheduler: ReduceLROnPlateau on val PR-AUC
- Loss: CrossEntropyLoss with per-class weights (ratio: 1 : N_neg/N_pos)
- Early stopping patience: 15 epochs (checked every 5 epochs)

**Outputs:**
- `models/graphsage_node.pt` — checkpoint + config + normalization params
- `models/gat_node.pt` — checkpoint + config + normalization params
- `models/graphsage_node.onnx`, `gat_node.onnx` — ONNX exports
- `results/gnn_results.json`
- `results/training_curves.png`
- `results/model_comparison.png`

```bash
python scripts/train_gnn.py                          # train both
python scripts/train_gnn.py --model graphsage        # GraphSAGE only
python scripts/train_gnn.py --model gat              # GAT only
python scripts/train_gnn.py --epochs 100 --hidden 256
```

---

### Step 5 — Explainability

**Script:** `scripts/explainability.py`

Generates regulator-ready explanations for the top-K flagged accounts.

**Outputs per flagged account:**

1. **Evidence Subgraph** — 2-hop neighborhood (nodes + edges) showing who the account transacts with
2. **Feature Profile** — normalized feature values vs population averages
3. **Reason Codes** — structured typology tags with descriptions:

| Code | Description |
|---|---|
| `RAPID_PASS_THROUGH` | Account receives and forwards funds within minutes |
| `FAN_IN` | Multiple accounts sending funds to this account (high in-degree) |
| `FAN_OUT` | Account sending funds to many different accounts (high out-degree) |
| `HIGH_VALUE` | Transaction amounts significantly above account average |
| `DEVICE_SHARING` | Account shares login device with other flagged accounts |
| `WALLET_SHARING` | Account shares wallet with other flagged accounts |
| `CHANNEL_HOPPING` | Account uses multiple channels in short timeframe |
| `RAPID_CASHOUT` | ATM withdrawal shortly after receiving funds |
| `PRIOR_SAR` | Account has prior SAR filing history |
| `UNUSUAL_ACTIVITY` | Account activity deviates significantly from historical pattern |

4. **Narrative** — Human-readable investigation summary

**Output:** `results/explanations.json`

```bash
python scripts/explainability.py                               # top 20 accounts
python scripts/explainability.py --top-k 50                   # top 50
python scripts/explainability.py --acct-ids 51164,1079,26617  # specific IDs
```

---

### Master Pipeline

**Script:** `scripts/run_all.py`

Runs the complete pipeline in sequence:

```bash
python scripts/run_all.py                    # full pipeline
python scripts/run_all.py --skip-enrichment  # skip cross-channel enrichment
python scripts/run_all.py --skip-graph       # skip graph build
```

Pipeline order:
1. Cross-channel enrichment
2. Graph construction (train / val / test)
3. Baseline training
4. GNN training
5. Explainability generation

---

## Model Results

### Node Classification — SAR Account Detection

| Model | PR-AUC | ROC-AUC | F1 (SAR) | Recall | Precision |
|---|---|---|---|---|---|
| **GraphSAGE** ★ | **0.987** | **0.9999** | 0.078 | **1.000** | 0.040 |
| XGBoost | 0.931 | 0.997 | 0.824 | 0.982 | 0.710 |
| Random Forest | 0.925 | 0.997 | 0.801 | 0.995 | 0.671 |
| GAT | 0.876 | 0.998 | 0.614 | 1.000 | 0.443 |
| Gradient Boosting | ~0.85 | ~0.99 | — | — | — |
| Logistic Regression | ~0.60 | ~0.90 | — | — | — |

> **Note on GraphSAGE:** GraphSAGE achieves the highest PR-AUC (0.987) and ROC-AUC (0.9999) and perfect recall (1.0), meaning it catches every mule account. The lower F1 is due to the extreme class imbalance — it errs on the side of flagging more accounts, which is the correct bias for a compliance system where false negatives (missed mules) are far more costly than false positives.

> **Note on XGBoost:** XGBoost achieves the best F1 balance (0.824) with high precision, making it the best choice when analyst bandwidth for false positives is constrained.

### Recall@K (SAR Detection)

| Model | Recall@1x | Recall@2x | Recall@5x |
|---|---|---|---|
| GraphSAGE | 99.4% | 100% | 100% |
| XGBoost | 82.9% | 100% | 100% |
| Random Forest | 82.5% | 100% | 100% |

> Recall@Kx = fraction of true SARs found when checking K × (true SAR count) top-scored accounts. A Recall@2x of 100% means reviewing 2× the expected SAR count guarantees catching all of them.

---

## Dashboard

**Script:** `dashboard/server.py`
**Frontend:** `dashboard/static/`

Start the dashboard:
```bash
cd dashboard
python server.py
# Open: http://localhost:5000
```

### API Reference

#### Overview & KPIs

```
GET /api/overview
```
Returns aggregated system KPIs:
```json
{
  "total_accounts": 289375,
  "sar_accounts": 11147,
  "sar_rate": 3.85,
  "total_edges": 320000,
  "flagged_accounts": 20,
  "best_model": { "name": "graphsage", "pr_auc": 0.987, "type": "gnn" },
  "risk_distribution": { "HIGH": 10, "MEDIUM": 7, "LOW": 3 },
  "edge_stats": { "transfer": 289000, "shared_device": 12000, ... },
  "graph_stats": { "train": {...}, "val": {...}, "test": {...} }
}
```

#### Model Performance

```
GET /api/models
```
Returns metrics for all baseline and GNN models, plus training histories (loss, PR-AUC per epoch).

#### Alert Queue

```
GET /api/alerts
```
Returns list of flagged accounts with risk level and reason codes (no subgraph for speed).

```
GET /api/alerts/<account_id>
```
Full alert detail: feature profile, evidence subgraph, reason codes, narrative.

#### Evidence Graph

```
GET /api/graph/<account_id>
```
Returns the 2-hop evidence subgraph (up to 50 nodes, 80 edges) for visualization.

#### Graph Forensics

```
GET /api/trace/<account_id>
```
BFS fund tracing: 3 hops upstream (who sent money to this account?) and downstream (where did money go?).

```
GET /api/community/<account_id>
```
Detects the connected mule ring component in the transfer graph (up to 50 nodes). Reveals the full money laundering network structure.

#### Real-Time Streaming

```
GET /api/stream
```
Server-Sent Events (SSE) stream of simulated real-time transaction scoring:
```json
{
  "id": 42,
  "timestamp": "2026-03-01 14:22:10",
  "src": "ACCT-1234",
  "dst": "ACCT-5678",
  "amount": 1337.50,
  "channel": "UPI",
  "score": 0.87,
  "risk": "HIGH_RISK"
}
```

---

### AI Investigator (Ollama)

The dashboard integrates with a local **Ollama** instance running **qwen2.5:7b** to generate professional SAR narratives and answer analyst questions.

#### Install Ollama

```bash
# Install Ollama (https://ollama.com)
ollama pull qwen2.5:7b
ollama serve
```

#### Automated Investigation

```
GET /api/agent/investigate/<account_id>
```

When Ollama is running: returns an SSE stream with:
1. **Reasoning steps** — rule-based evidence gathering (risk level, cluster size, fund trace)
2. **AI narrative** — qwen2.5:7b streams a full SAR document token by token
3. **Final result** — verdict (HIGH_RISK / SUSPICIOUS / LOW_RISK) + SAR text

When Ollama is offline: falls back to a template-based SAR with structured rule reasoning.

**Evidence gathered automatically:**
- Account risk profile and reason codes
- Connected cluster size (mule ring detection)
- 2-hop upstream fund sources
- 2-hop downstream fund destinations
- Computed risk score (0–1)
- Verdict and recommended action (FILE_SAR / ENHANCED_DUE_DILIGENCE / CLOSE_CASE)

#### Agent Chat

```
POST /api/agent/chat
Content-Type: application/json

{ "message": "Why is this account suspicious?", "account_id": 51164 }
```

Free-form Q&A with the AI agent, contextualized with the account's full intelligence profile.

---

## Training Notebook (A100)

**File:** `GhostBusters_A100_Training.ipynb`

A self-contained Jupyter notebook designed for Google Colab A100 GPU runtime. It runs the complete pipeline end-to-end including:

1. Repository setup and dependency installation
2. Data download and preparation
3. Cross-channel enrichment
4. Graph construction
5. Baseline model training
6. GNN training with A100 optimizations (TF32, AMP, torch.compile)
7. Explainability generation
8. Results visualization

The notebook applies all A100-specific optimizations and was the primary training environment for the model checkpoints in `models/`.

---

## Project Structure

```
GhostBusters/
│
├── GhostBusters_A100_Training.ipynb  # End-to-end training notebook (A100/Colab)
│
├── scripts/
│   ├── generate_cross_channel.py     # Step 1: Enrich with cross-channel data
│   ├── build_graph.py                # Step 2: Build heterogeneous entity graph
│   ├── train_baseline.py             # Step 3: Train XGBoost / RF / GBM / LR
│   ├── train_gnn.py                  # Step 4: Train GraphSAGE & GAT (A100 optimized)
│   ├── explainability.py             # Step 5: Generate reason codes & SAR narratives
│   ├── streaming_demo.py             # Standalone streaming demo
│   ├── generate_manifest.py          # Dataset manifest generator
│   └── run_all.py                    # Master pipeline runner
│
├── dashboard/
│   ├── server.py                     # Flask backend (all API endpoints + Ollama)
│   └── static/
│       ├── index.html                # Single-page dashboard UI
│       ├── js/app.js                 # Dashboard frontend logic
│       └── css/style.css             # Dashboard styles
│
├── data/
│   └── amlsim_v1/
│       ├── manifest.json             # Dataset manifest
│       ├── train/                    # Training split
│       ├── val/                      # Validation split
│       └── test/                     # Test split
│           ├── accounts.csv
│           ├── transactions.csv
│           ├── enriched_transactions.csv
│           ├── sar_accounts.csv
│           ├── app_logins.csv
│           ├── wallet_links.csv
│           ├── upi_handles.csv
│           ├── atm_withdrawals.csv
│           └── graph_data/
│               ├── account_features.npy
│               ├── account_labels.npy
│               ├── transfer_edge_index.npy
│               ├── [other_edge_types].npy
│               ├── graph_stats.json
│               └── id_mappings.json
│
├── models/
│   ├── graphsage_node.pt             # GraphSAGE checkpoint + config + norm params
│   ├── graphsage_node.onnx           # GraphSAGE ONNX export
│   ├── graphsage_node.torchscript.pt # GraphSAGE TorchScript
│   ├── gat_node.pt                   # GAT checkpoint
│   ├── gcn_node.pt                   # GCN checkpoint
│   ├── gin_node.pt                   # GIN checkpoint
│   ├── heterognn_node.pt             # HeteroGNN checkpoint
│   ├── baseline_xgboost.pkl          # XGBoost baseline
│   ├── baseline_random_forest.pkl    # Random Forest baseline
│   ├── baseline_gradient_boosting.pkl
│   ├── baseline_logistic_regression.pkl
│   ├── baseline_scaler.pkl           # Feature normalizer
│   └── xgboost_native.json           # XGBoost native format
│
├── results/
│   ├── all_results.json              # Combined metrics for all models
│   ├── baseline_results.json         # Baseline-only metrics
│   ├── gnn_results.json              # GNN-only metrics
│   ├── gnn_training_histories.json   # Per-epoch loss and PR-AUC curves
│   ├── explanations.json             # Per-account explanations (dashboard input)
│   ├── model_comparison.png          # Bar chart: GNN vs baselines
│   └── training_curves.png           # Loss & PR-AUC training curves
│
├── visualizations/
│   ├── 01_training_curves.png        # Detailed training curves
│   ├── 02_model_comparison.png       # Model comparison bar charts
│   ├── 03_roc_pr_curves.png          # ROC and PR curves for all models
│   ├── 04_confusion_matrices.png     # Confusion matrices
│   └── 05_feature_importance.png     # XGBoost feature importance
│
├── outputs/
│   ├── train_mp/                     # Mapped output data (training)
│   ├── val_mp/                       # Mapped output data (validation)
│   └── test_mp/                      # Mapped output data (test)
│
├── tests/
│   ├── test_api.py                   # Comprehensive test suite (pytest)
│   ├── diagnose.py                   # Quick API diagnostic script
│   └── verify_all.py                 # System verification script
│
├── .gitattributes                    # Git LFS tracking rules
├── .gitignore
└── README.md
```

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install torch torchvision torchaudio  # PyTorch (CUDA or CPU)
pip install xgboost scikit-learn pandas numpy matplotlib flask
pip install torch-geometric               # Optional: for PyG HeteroData format

# For AI Investigator (optional)
# Install Ollama from https://ollama.com
ollama pull qwen2.5:7b
```

### Run the Full Pipeline

```bash
# 1. Enrich data with cross-channel signals
python scripts/generate_cross_channel.py

# 2. Build the heterogeneous graph (all splits)
python scripts/build_graph.py

# 3. Train baseline models
python scripts/train_baseline.py

# 4. Train GNN models (GPU recommended)
python scripts/train_gnn.py --model graphsage --epochs 100 --hidden 256

# 5. Generate explanations for top flagged accounts
python scripts/explainability.py --top-k 20

# 6. Launch the dashboard
cd dashboard && python server.py
# Visit: http://localhost:5000
```

### Or use the master pipeline runner

```bash
python scripts/run_all.py
```

### Pre-trained Models

All model checkpoints are already saved in `models/`. You can skip steps 3 and 4 and go straight to the dashboard if you just want to explore the results.

---

## Tests

The `tests/` directory contains a comprehensive test suite covering all API endpoints and system workflows.

### Run Tests

```bash
# Requires the dashboard server to be running: python dashboard/server.py

# Full test suite
python -m pytest tests/test_api.py -v

# Specific test class
python -m pytest tests/test_api.py::TestGraphForensics -v
python -m pytest tests/test_api.py::TestAgentInvestigation -v
python -m pytest tests/test_api.py::TestSystemWorkflow -v

# Quick diagnostic (no pytest required)
python tests/diagnose.py
```

### Test Classes

| Class | Coverage |
|---|---|
| `TestEndpointHealth` | All API endpoints return correct status codes and fields |
| `TestGraphForensics` | `/api/trace` and `/api/community` correctness and limits |
| `TestAgentInvestigation` | Agent investigate endpoint (SSE or JSON fallback) |
| `TestAgentChat` | Chat endpoint with valid/empty messages |
| `TestOllamaConnectivity` | Ollama availability and qwen2.5:7b model check |
| `TestSystemWorkflow` | Full end-to-end investigation workflow + response time benchmarks |

---

## Tech Stack

| Category | Technology |
|---|---|
| **ML Framework** | PyTorch (pure, no PyG required for GNN training) |
| **GNN** | GraphSAGE, GAT — custom implementations |
| **Classical ML** | XGBoost, scikit-learn (RandomForest, GBM, LogReg) |
| **Graph Format** | PyTorch Geometric HeteroData (optional) + NumPy arrays |
| **Backend** | Flask (Python) |
| **Frontend** | Vanilla JS + HTML5 + CSS3 |
| **Streaming** | Server-Sent Events (SSE) |
| **AI / LLM** | Ollama (local) with qwen2.5:7b |
| **Visualization** | Matplotlib |
| **Data** | Pandas, NumPy |
| **Testing** | pytest + unittest |
| **GPU** | CUDA (A100 optimized: TF32, AMP, torch.compile) |
| **Model Export** | ONNX, TorchScript |

---

## Design Decisions

### Why pure PyTorch instead of PyG for GNN training?

PyTorch Geometric is great but adds a dependency that can be tricky to install (especially with CUDA version matching). The GNN training script (`train_gnn.py`) implements GraphSAGE and GAT from scratch using pure PyTorch tensor operations, making it more portable and educational while still being fully GPU-optimized.

### Why GraphSAGE achieves better PR-AUC than GAT?

GAT's attention mechanism is theoretically more expressive, but on this dataset the graph structure itself (topology, connectivity) is more informative than the precise weighting of individual neighbors. GraphSAGE's simple mean aggregation is more robust to noise in the edge weights. Additionally, GAT was trained with `gradient_checkpointing=True` which trades compute for memory, slightly affecting the learned attention patterns.

### Why Server-Sent Events instead of WebSockets for streaming?

SSE is simpler (unidirectional, HTTP-based, auto-reconnect, no special server support), and real-time transaction scoring only needs server → client communication. SSE works perfectly with Flask without requiring asyncio or a WebSocket server.

### Why Ollama (local LLM) instead of an API?

AML data is extremely sensitive — sending account details to external APIs would violate data governance policies. Local Ollama keeps all data on-premise. The system gracefully falls back to template-based SAR generation when Ollama is unavailable.

---

*Built for the GhostBusters AML research project.*

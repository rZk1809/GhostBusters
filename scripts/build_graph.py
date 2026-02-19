"""
build_graph.py
==============
Builds a unified heterogeneous entity graph from the enriched AMLSim dataset.
Produces a PyTorch Geometric HeteroData object with:

Node types: Account, Device, Wallet, UPI_VPA, ATM, Bank
Edge types: TRANSFER, LOGGED_IN, LINKED_WALLET, HAS_VPA, WITHDREW_ATM, BELONGS_TO_BANK, SHARED_DEVICE, SAME_WALLET

Also computes per-node and per-edge features:
  - Transaction-level: amount_zscore, balance_ratio, channel encoding
  - Account-level: in/out degree, fan-in/out ratio, burstiness, receive-send delay
  - Structural: k-hop counts, pagerank (optional)

Outputs saved to: data/amlsim_v1/{split}/graph_data/

Usage:
    python scripts/build_graph.py                    # all splits
    python scripts/build_graph.py --split train      # single split
    python scripts/build_graph.py --split train --skip-features  # graph only
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

# ─── Check for PyG availability, fall back to numpy-based graph if not ────────
try:
    import torch
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("[WARN] PyTorch Geometric not found. Will save graph as numpy/JSON format.")

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "amlsim_v1")

# ─── FEATURE HELPERS ─────────────────────────────────────────────────────────

CHANNEL_MAP = {"APP": 0, "WEB": 1, "UPI": 2, "WALLET": 3, "ATM": 4}

def encode_channel(ch):
    return CHANNEL_MAP.get(ch, -1)

def compute_account_features(accts_df, tx_df, sar_accts):
    """
    Compute rich account-level features from demographics + transaction history.
    Returns a dict: acct_id -> feature_vector (numpy array)
    """
    # Basic account features from demographics
    acct_features = {}
    
    # Pre-compute transaction statistics per account
    sent = tx_df.groupby("orig_acct")
    received = tx_df.groupby("bene_acct")
    
    sent_count = sent["tran_id"].count().to_dict()
    recv_count = received["tran_id"].count().to_dict()
    sent_amt_mean = sent["base_amt"].mean().to_dict()
    sent_amt_std = sent["base_amt"].std().fillna(0).to_dict()
    recv_amt_mean = received["base_amt"].mean().to_dict()
    recv_amt_std = received["base_amt"].std().fillna(0).to_dict()
    
    # SAR channel diversity (if enriched transactions available)
    has_channel = "channel" in tx_df.columns
    if has_channel:
        orig_channels = tx_df.groupby("orig_acct")["channel"].nunique().to_dict()
    
    # Temporal features
    tx_df_sorted = tx_df.sort_values("tran_timestamp")
    
    # Compute receive→send delay for each account (mule fingerprint)
    recv_send_delays = {}
    for acct in set(tx_df["orig_acct"]).intersection(set(tx_df["bene_acct"])):
        # Last inbound time
        inbound = tx_df_sorted[tx_df_sorted["bene_acct"] == acct]["tran_timestamp"]
        outbound = tx_df_sorted[tx_df_sorted["orig_acct"] == acct]["tran_timestamp"]
        if len(inbound) > 0 and len(outbound) > 0:
            try:
                last_in = pd.Timestamp(inbound.iloc[-1])
                first_out_after = outbound[outbound >= inbound.iloc[-1]]
                if len(first_out_after) > 0:
                    delay = (pd.Timestamp(first_out_after.iloc[0]) - last_in).total_seconds() / 60.0
                    recv_send_delays[acct] = delay
            except Exception:
                pass
    
    type_map = {"I": 0, "O": 1}  # Individual vs Organization
    stat_map = {"A": 1, "C": 0}  # Active vs Closed
    
    for _, row in accts_df.iterrows():
        acct_id = row["acct_id"]
        
        feat = [
            type_map.get(row.get("type", "I"), 0),                    # 0: account type
            stat_map.get(row.get("acct_stat", "A"), 1),               # 1: active status
            1 if str(row.get("prior_sar_count", "false")).lower() in ["true", "1"] else 0,  # 2: prior SAR
            float(row.get("initial_deposit", 0)),                     # 3: initial deposit
            sent_count.get(acct_id, 0),                               # 4: total sent count
            recv_count.get(acct_id, 0),                               # 5: total recv count
            sent_amt_mean.get(acct_id, 0),                            # 6: mean sent amount
            sent_amt_std.get(acct_id, 0),                             # 7: std sent amount
            recv_amt_mean.get(acct_id, 0),                            # 8: mean recv amount
            recv_amt_std.get(acct_id, 0),                             # 9: std recv amount
        ]
        
        # Fan-in / Fan-out ratio
        out_deg = sent_count.get(acct_id, 0)
        in_deg = recv_count.get(acct_id, 0)
        total_deg = out_deg + in_deg
        fan_out_ratio = out_deg / max(total_deg, 1)
        fan_in_ratio = in_deg / max(total_deg, 1)
        feat.append(fan_out_ratio)                                    # 10: fan-out ratio
        feat.append(fan_in_ratio)                                     # 11: fan-in ratio
        
        # Receive-send delay (mule signature)
        feat.append(recv_send_delays.get(acct_id, -1))                # 12: recv→send delay (min)
        
        # Channel diversity
        if has_channel:
            feat.append(orig_channels.get(acct_id, 0))                # 13: channel diversity
        else:
            feat.append(0)
        
        # SAR label
        feat.append(1 if acct_id in sar_accts else 0)                # 14: is_sar (label)
        
        acct_features[acct_id] = np.array(feat, dtype=np.float32)
    
    return acct_features

ACCT_FEATURE_NAMES = [
    "acct_type", "is_active", "prior_sar", "initial_deposit",
    "sent_count", "recv_count", "sent_amt_mean", "sent_amt_std",
    "recv_amt_mean", "recv_amt_std", "fan_out_ratio", "fan_in_ratio",
    "recv_send_delay_min", "channel_diversity", "is_sar"
]


def compute_edge_features(tx_df, acct_features):
    """
    Compute per-edge (transaction) features.
    Returns numpy array of shape (num_edges, num_features).
    """
    features = []
    
    # Pre-compute per-account amount stats for z-score
    orig_stats = tx_df.groupby("orig_acct")["base_amt"].agg(["mean", "std"]).fillna(0)
    orig_mean = orig_stats["mean"].to_dict()
    orig_std = orig_stats["std"].to_dict()
    
    has_channel = "channel" in tx_df.columns
    
    for _, row in tx_df.iterrows():
        feat = []
        
        # Amount features
        amt = float(row["base_amt"])
        feat.append(amt)                                              # 0: raw amount
        feat.append(np.log1p(amt))                                    # 1: log amount
        
        # Amount z-score within originator's history
        o_acct = row["orig_acct"]
        mean_a = orig_mean.get(o_acct, amt)
        std_a = orig_std.get(o_acct, 1.0)
        zscore = (amt - mean_a) / max(std_a, 1e-6)
        feat.append(zscore)                                           # 2: amount z-score
        
        # Channel encoding
        if has_channel:
            ch = encode_channel(row.get("channel", "APP"))
            feat.append(ch)                                           # 3: channel code
        else:
            feat.append(0)
        
        # SAR label
        is_sar = 1 if str(row.get("is_sar", False)).lower() in ["true", "1"] else 0
        feat.append(is_sar)                                           # 4: is_sar (label)
        
        features.append(feat)
    
    return np.array(features, dtype=np.float32)

EDGE_FEATURE_NAMES = ["amount", "log_amount", "amount_zscore", "channel_code", "is_sar"]


def build_id_maps(series_list):
    """Build a unified ID → integer mapping from multiple pd.Series."""
    all_ids = set()
    for s in series_list:
        all_ids.update(s.unique())
    return {v: i for i, v in enumerate(sorted(all_ids))}


def build_graph_for_split(split_name, skip_features=False):
    """Build the full heterogeneous graph for one split."""
    split_dir = os.path.join(BASE_DIR, split_name)
    out_dir = os.path.join(split_dir, "graph_data")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Building graph: {split_name}")
    print(f"{'='*60}")
    
    # ── Load data ──────────────────────────────────────────────────
    print("  Loading data...")
    accts_df = pd.read_csv(os.path.join(split_dir, "accounts.csv"))
    
    # Use enriched transactions if available
    etx_path = os.path.join(split_dir, "enriched_transactions.csv")
    tx_path = os.path.join(split_dir, "transactions.csv")
    tx_df = pd.read_csv(etx_path if os.path.exists(etx_path) else tx_path)
    
    sar_path = os.path.join(split_dir, "sar_accounts.csv")
    sar_accts = set()
    if os.path.exists(sar_path):
        try:
            sar_df = pd.read_csv(sar_path)
            sar_df.columns = [c.lower() for c in sar_df.columns]
            col = "account_id" if "account_id" in sar_df.columns else "acct_id"
            if col in sar_df.columns:
                sar_accts = set(sar_df[col].unique())
        except Exception:
            pass
    
    # Load enrichment tables
    logins_path = os.path.join(split_dir, "app_logins.csv")
    wallets_path = os.path.join(split_dir, "wallet_links.csv")
    vpa_path = os.path.join(split_dir, "upi_handles.csv")
    atm_path = os.path.join(split_dir, "atm_withdrawals.csv")
    
    def safe_read_csv(path):
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
            return df if len(df) > 0 else pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    
    logins_df = safe_read_csv(logins_path)
    wallets_df = safe_read_csv(wallets_path)
    vpa_df = safe_read_csv(vpa_path)
    atm_df = safe_read_csv(atm_path)
    
    print(f"    Accounts: {len(accts_df):,}")
    print(f"    Transactions: {len(tx_df):,}")
    print(f"    SAR accounts: {len(sar_accts):,}")
    print(f"    App logins: {len(logins_df):,}")
    print(f"    Wallet links: {len(wallets_df):,}")
    print(f"    UPI handles: {len(vpa_df):,}")
    print(f"    ATM withdrawals: {len(atm_df):,}")
    
    # ── Build ID mappings ──────────────────────────────────────────
    print("  Building ID mappings...")
    
    # Account IDs
    acct_ids = sorted(accts_df["acct_id"].unique())
    acct_map = {v: i for i, v in enumerate(acct_ids)}
    
    # Device IDs
    device_ids = set()
    if "device_id" in tx_df.columns:
        device_ids.update(tx_df["device_id"].dropna().unique())
    if not logins_df.empty and "device_id" in logins_df.columns:
        device_ids.update(logins_df["device_id"].dropna().unique())
    device_ids = sorted(device_ids)
    device_map = {v: i for i, v in enumerate(device_ids)}
    
    # Wallet IDs
    wallet_ids = sorted(wallets_df["wallet_id"].unique()) if not wallets_df.empty else []
    wallet_map = {v: i for i, v in enumerate(wallet_ids)}
    
    # VPA IDs
    vpa_ids = sorted(vpa_df["vpa"].unique()) if not vpa_df.empty else []
    vpa_map = {v: i for i, v in enumerate(vpa_ids)}
    
    # ATM IDs
    atm_ids = sorted(atm_df["atm_id"].unique()) if not atm_df.empty else []
    atm_map = {v: i for i, v in enumerate(atm_ids)}
    
    # Bank IDs
    bank_ids = sorted(accts_df["bank_id"].dropna().unique()) if "bank_id" in accts_df.columns else []
    bank_map = {v: i for i, v in enumerate(bank_ids)}
    
    print(f"    Nodes: {len(acct_map)} accounts, {len(device_map)} devices, "
          f"{len(wallet_map)} wallets, {len(vpa_map)} VPAs, "
          f"{len(atm_map)} ATMs, {len(bank_map)} banks")
    
    # ── Build edge index arrays ────────────────────────────────────
    print("  Building edge indices...")
    
    # 1. TRANSFER edges (account → account)
    transfer_src = []
    transfer_dst = []
    for _, row in tx_df.iterrows():
        src = acct_map.get(row["orig_acct"])
        dst = acct_map.get(row["bene_acct"])
        if src is not None and dst is not None:
            transfer_src.append(src)
            transfer_dst.append(dst)
    transfer_edge = np.array([transfer_src, transfer_dst], dtype=np.int64)
    print(f"    TRANSFER edges: {transfer_edge.shape[1]:,}")
    
    # 2. LOGGED_IN edges (account → device)
    login_src, login_dst = [], []
    if not logins_df.empty:
        for _, row in logins_df.iterrows():
            src = acct_map.get(row["acct_id"])
            dst = device_map.get(row.get("device_id"))
            if src is not None and dst is not None:
                login_src.append(src)
                login_dst.append(dst)
    login_edge = np.array([login_src, login_dst], dtype=np.int64) if login_src else np.zeros((2, 0), dtype=np.int64)
    print(f"    LOGGED_IN edges: {login_edge.shape[1]:,}")
    
    # 3. LINKED_WALLET edges (account → wallet)
    wallet_src, wallet_dst = [], []
    if not wallets_df.empty:
        for _, row in wallets_df.iterrows():
            src = acct_map.get(row["acct_id"])
            dst = wallet_map.get(row.get("wallet_id"))
            if src is not None and dst is not None:
                wallet_src.append(src)
                wallet_dst.append(dst)
    wallet_edge = np.array([wallet_src, wallet_dst], dtype=np.int64) if wallet_src else np.zeros((2, 0), dtype=np.int64)
    print(f"    LINKED_WALLET edges: {wallet_edge.shape[1]:,}")
    
    # 4. HAS_VPA edges (account → vpa)
    vpa_src, vpa_dst = [], []
    if not vpa_df.empty:
        for _, row in vpa_df.iterrows():
            src = acct_map.get(row["acct_id"])
            dst = vpa_map.get(row.get("vpa"))
            if src is not None and dst is not None:
                vpa_src.append(src)
                vpa_dst.append(dst)
    vpa_edge = np.array([vpa_src, vpa_dst], dtype=np.int64) if vpa_src else np.zeros((2, 0), dtype=np.int64)
    print(f"    HAS_VPA edges: {vpa_edge.shape[1]:,}")
    
    # 5. WITHDREW_ATM edges (account → ATM)
    atm_src, atm_dst = [], []
    if not atm_df.empty:
        for _, row in atm_df.iterrows():
            src = acct_map.get(row["acct_id"])
            dst = atm_map.get(row.get("atm_id"))
            if src is not None and dst is not None:
                atm_src.append(src)
                atm_dst.append(dst)
    atm_edge = np.array([atm_src, atm_dst], dtype=np.int64) if atm_src else np.zeros((2, 0), dtype=np.int64)
    print(f"    WITHDREW_ATM edges: {atm_edge.shape[1]:,}")
    
    # 6. BELONGS_TO_BANK edges (account → bank)
    bank_src, bank_dst = [], []
    if "bank_id" in accts_df.columns:
        for _, row in accts_df.iterrows():
            src = acct_map.get(row["acct_id"])
            dst = bank_map.get(row.get("bank_id"))
            if src is not None and dst is not None:
                bank_src.append(src)
                bank_dst.append(dst)
    bank_edge = np.array([bank_src, bank_dst], dtype=np.int64) if bank_src else np.zeros((2, 0), dtype=np.int64)
    print(f"    BELONGS_TO_BANK edges: {bank_edge.shape[1]:,}")
    
    # 7. SHARED_DEVICE edges (account ↔ account, derived)
    print("  Deriving shared-device edges...")
    device_to_accounts = defaultdict(set)
    for acct_id, dev_id in zip(
        logins_df["acct_id"] if not logins_df.empty else [],
        logins_df["device_id"] if not logins_df.empty else []
    ):
        if pd.notna(dev_id):
            device_to_accounts[dev_id].add(acct_id)
    
    shared_dev_src, shared_dev_dst = [], []
    for dev, accounts in device_to_accounts.items():
        acct_list = sorted(accounts)
        for i in range(len(acct_list)):
            for j in range(i + 1, len(acct_list)):
                a = acct_map.get(acct_list[i])
                b = acct_map.get(acct_list[j])
                if a is not None and b is not None:
                    shared_dev_src.extend([a, b])
                    shared_dev_dst.extend([b, a])
    shared_dev_edge = np.array([shared_dev_src, shared_dev_dst], dtype=np.int64) if shared_dev_src else np.zeros((2, 0), dtype=np.int64)
    # Deduplicate
    if shared_dev_edge.shape[1] > 0:
        unique_pairs = np.unique(shared_dev_edge.T, axis=0).T
        shared_dev_edge = unique_pairs
    print(f"    SHARED_DEVICE edges: {shared_dev_edge.shape[1]:,}")
    
    # 8. SAME_WALLET edges (account ↔ account, derived)
    print("  Deriving shared-wallet edges...")
    wallet_to_accounts = defaultdict(set)
    if not wallets_df.empty:
        for _, row in wallets_df.iterrows():
            wallet_to_accounts[row["wallet_id"]].add(row["acct_id"])
    
    shared_wal_src, shared_wal_dst = [], []
    for wal, accounts in wallet_to_accounts.items():
        acct_list = sorted(accounts)
        for i in range(len(acct_list)):
            for j in range(i + 1, len(acct_list)):
                a = acct_map.get(acct_list[i])
                b = acct_map.get(acct_list[j])
                if a is not None and b is not None:
                    shared_wal_src.extend([a, b])
                    shared_wal_dst.extend([b, a])
    shared_wal_edge = np.array([shared_wal_src, shared_wal_dst], dtype=np.int64) if shared_wal_src else np.zeros((2, 0), dtype=np.int64)
    if shared_wal_edge.shape[1] > 0:
        unique_pairs = np.unique(shared_wal_edge.T, axis=0).T
        shared_wal_edge = unique_pairs
    print(f"    SAME_WALLET edges: {shared_wal_edge.shape[1]:,}")
    
    # ── Compute features ───────────────────────────────────────────
    acct_feat_matrix = None
    edge_feat_matrix = None
    acct_labels = None
    edge_labels = None
    
    if not skip_features:
        print("  Computing account features...")
        acct_features = compute_account_features(accts_df, tx_df, sar_accts)
        acct_feat_matrix = np.stack([acct_features.get(aid, np.zeros(len(ACCT_FEATURE_NAMES), dtype=np.float32))
                                     for aid in acct_ids])
        acct_labels = acct_feat_matrix[:, -1].astype(np.int64)   # last col is is_sar
        acct_feat_matrix = acct_feat_matrix[:, :-1]              # remove label from features
        print(f"    Account features: {acct_feat_matrix.shape}")
        
        print("  Computing edge features (this may take a while)...")
        edge_feat_matrix = compute_edge_features(tx_df, acct_features)
        edge_labels = edge_feat_matrix[:, -1].astype(np.int64)   # last col is is_sar
        edge_feat_matrix = edge_feat_matrix[:, :-1]              # remove label
        print(f"    Edge features: {edge_feat_matrix.shape}")
    
    # ── Save ───────────────────────────────────────────────────────
    print("  Saving graph data...")
    
    if HAS_PYG:
        data = HeteroData()
        
        # Node features
        if acct_feat_matrix is not None:
            data["account"].x = torch.tensor(acct_feat_matrix, dtype=torch.float32)
            data["account"].y = torch.tensor(acct_labels, dtype=torch.long)
        else:
            data["account"].num_nodes = len(acct_ids)
        
        data["device"].num_nodes = len(device_ids)
        data["wallet"].num_nodes = len(wallet_ids)
        data["vpa"].num_nodes = len(vpa_ids)
        data["atm"].num_nodes = len(atm_ids)
        data["bank"].num_nodes = len(bank_ids)
        
        # Edge indices
        data["account", "transfer", "account"].edge_index = torch.tensor(transfer_edge, dtype=torch.long)
        if edge_feat_matrix is not None:
            data["account", "transfer", "account"].edge_attr = torch.tensor(edge_feat_matrix, dtype=torch.float32)
            data["account", "transfer", "account"].edge_label = torch.tensor(edge_labels, dtype=torch.long)
        
        if login_edge.shape[1] > 0:
            data["account", "logged_in", "device"].edge_index = torch.tensor(login_edge, dtype=torch.long)
        if wallet_edge.shape[1] > 0:
            data["account", "linked_wallet", "wallet"].edge_index = torch.tensor(wallet_edge, dtype=torch.long)
        if vpa_edge.shape[1] > 0:
            data["account", "has_vpa", "vpa"].edge_index = torch.tensor(vpa_edge, dtype=torch.long)
        if atm_edge.shape[1] > 0:
            data["account", "withdrew_atm", "atm"].edge_index = torch.tensor(atm_edge, dtype=torch.long)
        if bank_edge.shape[1] > 0:
            data["account", "belongs_to", "bank"].edge_index = torch.tensor(bank_edge, dtype=torch.long)
        if shared_dev_edge.shape[1] > 0:
            data["account", "shared_device", "account"].edge_index = torch.tensor(shared_dev_edge, dtype=torch.long)
        if shared_wal_edge.shape[1] > 0:
            data["account", "same_wallet", "account"].edge_index = torch.tensor(shared_wal_edge, dtype=torch.long)
        
        pyg_path = os.path.join(out_dir, "hetero_graph.pt")
        torch.save(data, pyg_path)
        print(f"    → Saved PyG HeteroData to {pyg_path}")
        print(f"    → Graph summary: {data}")
    
    # Always save numpy format for compatibility
    np.save(os.path.join(out_dir, "transfer_edge_index.npy"), transfer_edge)
    np.save(os.path.join(out_dir, "login_edge_index.npy"), login_edge)
    np.save(os.path.join(out_dir, "wallet_edge_index.npy"), wallet_edge)
    np.save(os.path.join(out_dir, "vpa_edge_index.npy"), vpa_edge)
    np.save(os.path.join(out_dir, "atm_edge_index.npy"), atm_edge)
    np.save(os.path.join(out_dir, "bank_edge_index.npy"), bank_edge)
    np.save(os.path.join(out_dir, "shared_device_edge_index.npy"), shared_dev_edge)
    np.save(os.path.join(out_dir, "same_wallet_edge_index.npy"), shared_wal_edge)
    
    if acct_feat_matrix is not None:
        np.save(os.path.join(out_dir, "account_features.npy"), acct_feat_matrix)
        np.save(os.path.join(out_dir, "account_labels.npy"), acct_labels)
    if edge_feat_matrix is not None:
        np.save(os.path.join(out_dir, "edge_features.npy"), edge_feat_matrix)
        np.save(os.path.join(out_dir, "edge_labels.npy"), edge_labels)
    
    # Save ID mappings
    mappings = {
        "acct_map": {str(k): v for k, v in acct_map.items()},
        "device_map": device_map,
        "wallet_map": wallet_map,
        "vpa_map": vpa_map,
        "atm_map": atm_map,
        "bank_map": bank_map,
        "acct_feature_names": ACCT_FEATURE_NAMES[:-1],  # exclude label
        "edge_feature_names": EDGE_FEATURE_NAMES[:-1],   # exclude label
    }
    with open(os.path.join(out_dir, "id_mappings.json"), "w") as f:
        json.dump(mappings, f, indent=2)
    
    # Save graph stats
    stats = {
        "split": split_name,
        "num_accounts": len(acct_ids),
        "num_devices": len(device_ids),
        "num_wallets": len(wallet_ids),
        "num_vpas": len(vpa_ids),
        "num_atms": len(atm_ids),
        "num_banks": len(bank_ids),
        "num_sar_accounts": len(sar_accts),
        "edges": {
            "transfer": int(transfer_edge.shape[1]),
            "logged_in": int(login_edge.shape[1]),
            "linked_wallet": int(wallet_edge.shape[1]),
            "has_vpa": int(vpa_edge.shape[1]),
            "withdrew_atm": int(atm_edge.shape[1]),
            "belongs_to_bank": int(bank_edge.shape[1]),
            "shared_device": int(shared_dev_edge.shape[1]),
            "same_wallet": int(shared_wal_edge.shape[1]),
        },
        "total_edges": sum([
            transfer_edge.shape[1], login_edge.shape[1], wallet_edge.shape[1],
            vpa_edge.shape[1], atm_edge.shape[1], bank_edge.shape[1],
            shared_dev_edge.shape[1], shared_wal_edge.shape[1]
        ]),
    }
    with open(os.path.join(out_dir, "graph_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n  ✓ Graph for {split_name} complete!")
    print(f"    Total nodes: {stats['num_accounts'] + stats['num_devices'] + stats['num_wallets'] + stats['num_vpas'] + stats['num_atms'] + stats['num_banks']:,}")
    print(f"    Total edges: {stats['total_edges']:,}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build unified entity graph")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature computation (graph structure only)")
    args = parser.parse_args()
    
    splits = [args.split] if args.split else ["train", "val", "test"]
    
    all_stats = {}
    for split in splits:
        stats = build_graph_for_split(split, skip_features=args.skip_features)
        all_stats[split] = stats
    
    print("\n" + "="*60)
    print("GRAPH BUILD SUMMARY")
    print("="*60)
    for split, stats in all_stats.items():
        print(f"\n  {split}:")
        print(f"    Accounts: {stats['num_accounts']:,} ({stats['num_sar_accounts']:,} SAR)")
        print(f"    Total edges: {stats['total_edges']:,}")
        for etype, count in stats["edges"].items():
            if count > 0:
                print(f"      {etype}: {count:,}")


if __name__ == "__main__":
    main()

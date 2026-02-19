"""
generate_cross_channel.py
=========================
Enriches the frozen AMLSim dataset with synthetic cross-channel attributes:
  1. enriched_transactions.csv  — adds channel, device_id, ip_id per transaction
  2. app_logins.csv             — login events per account
  3. wallet_links.csv           — account ↔ wallet linkages
  4. upi_handles.csv            — account → UPI VPA mapping
  5. atm_withdrawals.csv        — ATM cashout events (especially post-inbound for mules)

Design principles:
  - SAR/mule accounts exhibit cross-channel hopping (receive APP → forward UPI → cashout ATM)
  - Normal accounts show single-channel consistency
  - Device/IP sharing is higher among SAR-linked accounts
  - ATM cashouts happen within minutes of inbound transfers for mule accounts

Usage:
    python scripts/generate_cross_channel.py          # processes all 3 splits
    python scripts/generate_cross_channel.py --split train  # single split
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import timedelta

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SEED = 42
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "amlsim_v1")

# Channel distribution for normal vs SAR accounts
CHANNELS = ["APP", "WEB", "UPI", "WALLET", "ATM"]
NORMAL_CHANNEL_WEIGHTS = [0.35, 0.25, 0.25, 0.10, 0.05]  # normal: mostly APP/WEB
SAR_CHANNEL_WEIGHTS    = [0.15, 0.10, 0.30, 0.20, 0.25]  # mules: heavy UPI/ATM

# Device/IP pools
NUM_DEVICES_NORMAL = 50000   # large pool → low sharing
NUM_DEVICES_SAR    = 500     # small pool → high sharing (mule infrastructure)
NUM_IPS_NORMAL     = 80000
NUM_IPS_SAR        = 300

# Wallet / VPA settings
WALLET_POOL_NORMAL = 40000
WALLET_POOL_SAR    = 200     # shared wallets for mule hubs
VPA_POOL_NORMAL    = 60000
VPA_POOL_SAR       = 300

# ATM settings
NUM_ATMS = 2000
ATM_CASHOUT_DELAY_MINUTES_MULE = (2, 30)    # mules: 2-30 min after inbound
ATM_CASHOUT_DELAY_MINUTES_NORMAL = (60, 1440)  # normals: 1h-24h if at all
ATM_CASHOUT_PROB_MULE = 0.70    # 70% of mule inbound txns lead to ATM cashout
ATM_CASHOUT_PROB_NORMAL = 0.02  # 2% for normal accounts


def load_sar_accounts(split_dir):
    """Load SAR account IDs from sar_accounts.csv."""
    sar_path = os.path.join(split_dir, "sar_accounts.csv")
    if os.path.exists(sar_path):
        try:
            df = pd.read_csv(sar_path)
            df.columns = [c.lower() for c in df.columns]
            col = "account_id" if "account_id" in df.columns else "acct_id"
            if col in df.columns:
                return set(df[col].unique())
        except Exception:
            pass
    return set()


def load_alert_groups(split_dir):
    """Load alert group memberships — accounts in the same alert share infrastructure."""
    alert_path = os.path.join(split_dir, "alert_accounts.csv")
    if os.path.exists(alert_path):
        df = pd.read_csv(alert_path)
        if "alert_id" in df.columns and "acct_id" in df.columns:
            groups = defaultdict(list)
            for _, row in df.iterrows():
                groups[row["alert_id"]].append(row["acct_id"])
            return dict(groups)
    return {}


def assign_devices_and_ips(accounts_df, sar_accts, alert_groups, rng):
    """
    Assign device_id and ip_id to each account.
    SAR accounts in the same alert group share devices/IPs at high rates.
    """
    acct_device = {}
    acct_ip = {}
    
    # Pre-assign shared devices for alert groups (mule rings share infrastructure)
    group_devices = {}
    group_ips = {}
    for alert_id, members in alert_groups.items():
        # Each alert group gets 1-3 shared devices
        n_shared = min(rng.integers(1, 4), len(members))
        shared_devs = [f"DEV_SAR_{rng.integers(0, NUM_DEVICES_SAR)}" for _ in range(n_shared)]
        shared_ips = [f"IP_SAR_{rng.integers(0, NUM_IPS_SAR)}" for _ in range(n_shared)]
        group_devices[alert_id] = shared_devs
        group_ips[alert_id] = shared_ips
    
    # Assign per account
    for acct_id in accounts_df["acct_id"]:
        is_sar = acct_id in sar_accts
        
        # Check if this account belongs to an alert group
        assigned_from_group = False
        if is_sar:
            for alert_id, members in alert_groups.items():
                if acct_id in members:
                    devs = group_devices.get(alert_id, [])
                    ips = group_ips.get(alert_id, [])
                    if devs and rng.random() < 0.7:  # 70% chance to use shared device
                        acct_device[acct_id] = rng.choice(devs)
                        acct_ip[acct_id] = rng.choice(ips)
                        assigned_from_group = True
                        break
        
        if not assigned_from_group:
            if is_sar:
                acct_device[acct_id] = f"DEV_SAR_{rng.integers(0, NUM_DEVICES_SAR)}"
                acct_ip[acct_id] = f"IP_SAR_{rng.integers(0, NUM_IPS_SAR)}"
            else:
                acct_device[acct_id] = f"DEV_{rng.integers(0, NUM_DEVICES_NORMAL)}"
                acct_ip[acct_id] = f"IP_{rng.integers(0, NUM_IPS_NORMAL)}"
    
    return acct_device, acct_ip


def assign_wallets_and_vpas(accounts_df, sar_accts, alert_groups, rng):
    """Assign wallet IDs and UPI VPAs. SAR groups share wallets."""
    acct_wallet = {}
    acct_vpa = {}
    
    # Shared wallets for alert groups
    group_wallets = {}
    for alert_id, members in alert_groups.items():
        shared_wallet = f"WALLET_SAR_{rng.integers(0, WALLET_POOL_SAR)}"
        group_wallets[alert_id] = shared_wallet
    
    for acct_id in accounts_df["acct_id"]:
        is_sar = acct_id in sar_accts
        
        assigned = False
        if is_sar:
            for alert_id, members in alert_groups.items():
                if acct_id in members and rng.random() < 0.6:
                    acct_wallet[acct_id] = group_wallets.get(alert_id, f"WALLET_SAR_{rng.integers(0, WALLET_POOL_SAR)}")
                    acct_vpa[acct_id] = f"mule{rng.integers(0, VPA_POOL_SAR)}@upi"
                    assigned = True
                    break
        
        if not assigned:
            if is_sar:
                acct_wallet[acct_id] = f"WALLET_SAR_{rng.integers(0, WALLET_POOL_SAR)}"
                acct_vpa[acct_id] = f"mule{rng.integers(0, VPA_POOL_SAR)}@upi"
            else:
                acct_wallet[acct_id] = f"WALLET_{rng.integers(0, WALLET_POOL_NORMAL)}"
                acct_vpa[acct_id] = f"user{acct_id}@upi"
    
    return acct_wallet, acct_vpa


def enrich_transactions(tx_df, sar_accts, acct_device, acct_ip, rng):
    """
    Add channel, device_id, ip_id columns to transactions.
    SAR transactions use mule-skewed channel distribution.
    """
    n = len(tx_df)
    channels = np.empty(n, dtype=object)
    devices = np.empty(n, dtype=object)
    ips = np.empty(n, dtype=object)
    
    is_sar_mask = tx_df["is_sar"].astype(str).str.lower().isin(["true", "1", "yes"])
    orig_accts = tx_df["orig_acct"].values
    
    # Vectorized channel assignment
    sar_idx = np.where(is_sar_mask)[0]
    normal_idx = np.where(~is_sar_mask)[0]
    
    channels[sar_idx] = rng.choice(CHANNELS, size=len(sar_idx), p=SAR_CHANNEL_WEIGHTS)
    channels[normal_idx] = rng.choice(CHANNELS, size=len(normal_idx), p=NORMAL_CHANNEL_WEIGHTS)
    
    # Assign device/IP from originator account
    for i in range(n):
        acct = orig_accts[i]
        devices[i] = acct_device.get(acct, f"DEV_{acct}")
        ips[i] = acct_ip.get(acct, f"IP_{acct}")
    
    tx_df = tx_df.copy()
    tx_df["channel"] = channels
    tx_df["device_id"] = devices
    tx_df["ip_id"] = ips
    
    return tx_df


def generate_app_logins(accounts_df, tx_df, acct_device, acct_ip, sar_accts, rng):
    """
    Generate app login events. SAR accounts login from shared devices more.
    Each account gets 1-5 logins around their transaction times.
    """
    records = []
    
    # Collect timestamps from BOTH sent and received transactions (vectorized)
    tx_times = defaultdict(list)
    orig_groups = tx_df.groupby("orig_acct")["tran_timestamp"].apply(list).to_dict()
    bene_groups = tx_df.groupby("bene_acct")["tran_timestamp"].apply(list).to_dict()
    for acct, times in orig_groups.items():
        tx_times[acct].extend(times)
    for acct, times in bene_groups.items():
        tx_times[acct].extend(times)
    
    # Base timestamp for accounts without transactions
    base_ts = "2017-01-15T12:00:00Z"
    
    for acct_id in accounts_df["acct_id"].values:
        is_sar = acct_id in sar_accts
        n_logins = rng.integers(2, 8) if is_sar else rng.integers(1, 4)
        
        times = tx_times.get(acct_id, [base_ts])
        
        device = acct_device.get(acct_id, f"DEV_{acct_id}")
        ip = acct_ip.get(acct_id, f"IP_{acct_id}")
        
        for _ in range(n_logins):
            ts = str(rng.choice(times))  # convert numpy.str_ to Python str
            try:
                login_time = pd.Timestamp(ts) - timedelta(minutes=int(rng.integers(1, 60)))
                records.append({
                    "acct_id": acct_id,
                    "device_id": device,
                    "ip_id": ip,
                    "login_timestamp": login_time.isoformat()
                })
            except Exception:
                continue
    
    return pd.DataFrame(records)


def generate_wallet_links(accounts_df, acct_wallet, sar_accts, rng):
    """Generate wallet linkage events."""
    records = []
    for acct_id in accounts_df["acct_id"].values:
        wallet = acct_wallet.get(acct_id)
        if wallet:
            # Link timestamp: sometime around account open date
            records.append({
                "acct_id": acct_id,
                "wallet_id": wallet,
                "link_timestamp": "2017-01-01T00:00:00Z"  # simplified: use account creation
            })
    return pd.DataFrame(records)


def generate_upi_handles(accounts_df, acct_vpa, rng):
    """Generate UPI VPA mappings."""
    records = []
    for acct_id in accounts_df["acct_id"].values:
        vpa = acct_vpa.get(acct_id)
        if vpa:
            records.append({"acct_id": acct_id, "vpa": vpa})
    return pd.DataFrame(records)


def generate_atm_withdrawals(tx_df, sar_accts, rng):
    """
    Generate ATM withdrawal events.
    Mule accounts: high probability of ATM cashout within minutes of inbound transfer.
    Normal accounts: rare and delayed ATM usage.
    """
    records = []
    
    # Focus on inbound transactions (bene_acct receives money)
    for _, row in tx_df.iterrows():
        bene = row["bene_acct"]
        is_sar = bene in sar_accts
        
        prob = ATM_CASHOUT_PROB_MULE if is_sar else ATM_CASHOUT_PROB_NORMAL
        
        if rng.random() < prob:
            try:
                inbound_time = pd.Timestamp(row["tran_timestamp"])
                if is_sar:
                    delay_min = rng.integers(*ATM_CASHOUT_DELAY_MINUTES_MULE)
                else:
                    delay_min = rng.integers(*ATM_CASHOUT_DELAY_MINUTES_NORMAL)
                
                cashout_time = inbound_time + timedelta(minutes=int(delay_min))
                cashout_amt = float(row["base_amt"]) * rng.uniform(0.80, 0.99)
                
                records.append({
                    "acct_id": bene,
                    "atm_id": f"ATM_{rng.integers(0, NUM_ATMS)}",
                    "amount": round(cashout_amt, 2),
                    "timestamp": cashout_time.isoformat(),
                    "source_tran_id": row["tran_id"]
                })
            except Exception:
                continue
    
    return pd.DataFrame(records)


def process_split(split_name, rng):
    """Process a single data split (train/val/test)."""
    split_dir = os.path.join(BASE_DIR, split_name)
    print(f"\n{'='*60}")
    print(f"Processing split: {split_name}")
    print(f"{'='*60}")
    
    # Load core data
    print("  Loading transactions...")
    tx_df = pd.read_csv(os.path.join(split_dir, "transactions.csv"))
    print(f"    {len(tx_df):,} transactions")
    
    print("  Loading accounts...")
    accts_df = pd.read_csv(os.path.join(split_dir, "accounts.csv"))
    print(f"    {len(accts_df):,} accounts")
    
    # Load SAR accounts and alert groups
    sar_accts = load_sar_accounts(split_dir)
    alert_groups = load_alert_groups(split_dir)
    print(f"    {len(sar_accts):,} SAR accounts, {len(alert_groups)} alert groups")
    
    # Step 1: Assign devices, IPs, wallets, VPAs
    print("  Assigning devices & IPs...")
    acct_device, acct_ip = assign_devices_and_ips(accts_df, sar_accts, alert_groups, rng)
    
    print("  Assigning wallets & VPAs...")
    acct_wallet, acct_vpa = assign_wallets_and_vpas(accts_df, sar_accts, alert_groups, rng)
    
    # Step 2: Enrich transactions
    print("  Enriching transactions with channels...")
    enriched_tx = enrich_transactions(tx_df, sar_accts, acct_device, acct_ip, rng)
    out_path = os.path.join(split_dir, "enriched_transactions.csv")
    enriched_tx.to_csv(out_path, index=False)
    print(f"    → {out_path} ({len(enriched_tx):,} rows)")
    
    # Step 3: App logins
    print("  Generating app logins...")
    logins_df = generate_app_logins(accts_df, tx_df, acct_device, acct_ip, sar_accts, rng)
    out_path = os.path.join(split_dir, "app_logins.csv")
    logins_df.to_csv(out_path, index=False)
    print(f"    → {out_path} ({len(logins_df):,} rows)")
    
    # Step 4: Wallet links
    print("  Generating wallet links...")
    wallets_df = generate_wallet_links(accts_df, acct_wallet, sar_accts, rng)
    out_path = os.path.join(split_dir, "wallet_links.csv")
    wallets_df.to_csv(out_path, index=False)
    print(f"    → {out_path} ({len(wallets_df):,} rows)")
    
    # Step 5: UPI handles
    print("  Generating UPI handles...")
    vpa_df = generate_upi_handles(accts_df, acct_vpa, rng)
    out_path = os.path.join(split_dir, "upi_handles.csv")
    vpa_df.to_csv(out_path, index=False)
    print(f"    → {out_path} ({len(vpa_df):,} rows)")
    
    # Step 6: ATM withdrawals (most expensive — iterates transactions)
    print("  Generating ATM withdrawals (this may take a moment)...")
    atm_df = generate_atm_withdrawals(tx_df, sar_accts, rng)
    out_path = os.path.join(split_dir, "atm_withdrawals.csv")
    atm_df.to_csv(out_path, index=False)
    print(f"    → {out_path} ({len(atm_df):,} rows)")
    
    # Summary stats
    print(f"\n  Summary for {split_name}:")
    print(f"    Channels: {enriched_tx['channel'].value_counts().to_dict()}")
    sar_channels = enriched_tx[enriched_tx['is_sar'].astype(str).str.lower().isin(['true', '1'])]['channel'].value_counts()
    print(f"    SAR channels: {sar_channels.to_dict()}")
    
    # Count shared devices among SAR accounts
    sar_devices = {acct_device[a] for a in sar_accts if a in acct_device}
    print(f"    Unique devices used by SAR accounts: {len(sar_devices)} (from pool of {NUM_DEVICES_SAR})")


def main():
    parser = argparse.ArgumentParser(description="Generate cross-channel enrichment tables")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None,
                        help="Process a single split (default: all)")
    args = parser.parse_args()
    
    rng = np.random.default_rng(SEED)
    
    splits = [args.split] if args.split else ["train", "val", "test"]
    
    for split in splits:
        process_split(split, rng)
    
    print("\n✓ Cross-channel enrichment complete!")
    print("  Generated files per split:")
    print("    - enriched_transactions.csv (transactions + channel/device/IP)")
    print("    - app_logins.csv (login events)")
    print("    - wallet_links.csv (wallet linkages)")
    print("    - upi_handles.csv (UPI VPA mappings)")
    print("    - atm_withdrawals.csv (ATM cashout events)")


if __name__ == "__main__":
    main()

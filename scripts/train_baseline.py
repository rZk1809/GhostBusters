"""
train_baseline.py
=================
Train XGBoost/LightGBM baseline models for:
  1. Node classification — Mule/SAR account detection
  2. Edge classification — Suspicious transaction detection

Uses pre-computed features from build_graph.py (numpy arrays in graph_data/).

Usage:
    python scripts/train_baseline.py                    # both tasks, all defaults
    python scripts/train_baseline.py --task node        # node classification only
    python scripts/train_baseline.py --task edge        # edge classification only
"""

import os
import json
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        classification_report, precision_recall_curve, auc,
        f1_score, roc_auc_score, average_precision_score,
        confusion_matrix
    )
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[ERROR] scikit-learn is required. Install with: pip install scikit-learn")

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "amlsim_v1")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_graph_data(split):
    """Load pre-computed features and labels from graph_data/."""
    gd = os.path.join(BASE_DIR, split, "graph_data")
    data = {}
    for name in ["account_features", "account_labels", "edge_features", "edge_labels"]:
        path = os.path.join(gd, f"{name}.npy")
        if os.path.exists(path):
            data[name] = np.load(path)
            print(f"  Loaded {split}/{name}: {data[name].shape}")
    return data


def evaluate_model(y_true, y_pred, y_prob, task_name):
    """Compute comprehensive metrics."""
    metrics = {}
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["accuracy"] = report["accuracy"]
    metrics["f1_macro"] = report["macro avg"]["f1-score"]
    metrics["f1_weighted"] = report["weighted avg"]["f1-score"]
    
    # Per-class metrics
    if "1" in report:
        metrics["precision_sar"] = report["1"]["precision"]
        metrics["recall_sar"] = report["1"]["recall"]
        metrics["f1_sar"] = report["1"]["f1-score"]
    
    # PR-AUC (most important for imbalanced SAR detection)
    if y_prob is not None:
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            metrics["pr_auc"] = auc(recall, precision)
            metrics["avg_precision"] = average_precision_score(y_true, y_prob)
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            pass
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    # Recall@K (top K predictions by probability)
    if y_prob is not None:
        n_pos = int(y_true.sum())
        for k_mult in [1, 2, 5]:
            k = n_pos * k_mult
            if k <= len(y_prob):
                top_k_idx = np.argsort(y_prob)[-k:]
                recall_at_k = y_true[top_k_idx].sum() / max(n_pos, 1)
                metrics[f"recall@{k_mult}x"] = float(recall_at_k)
    
    # Print summary
    print(f"\n  === {task_name} Results ===")
    print(f"  Accuracy:     {metrics.get('accuracy', 0):.4f}")
    print(f"  F1 (macro):   {metrics.get('f1_macro', 0):.4f}")
    print(f"  F1 (SAR):     {metrics.get('f1_sar', 0):.4f}")
    print(f"  PR-AUC:       {metrics.get('pr_auc', 0):.4f}")
    print(f"  ROC-AUC:      {metrics.get('roc_auc', 0):.4f}")
    print(f"  Precision(SAR): {metrics.get('precision_sar', 0):.4f}")
    print(f"  Recall(SAR):    {metrics.get('recall_sar', 0):.4f}")
    if "recall@1x" in metrics:
        print(f"  Recall@1x:    {metrics['recall@1x']:.4f}")
        print(f"  Recall@2x:    {metrics.get('recall@2x', 0):.4f}")
        print(f"  Recall@5x:    {metrics.get('recall@5x', 0):.4f}")
    
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    print(f"    FN={cm[1][0]:,}  TP={cm[1][1]:,}")
    
    return metrics


def train_node_classifier(train_data, val_data, test_data):
    """Train XGBoost for mule/SAR account detection."""
    print("\n" + "="*60)
    print("NODE CLASSIFICATION — Mule/SAR Account Detection")
    print("="*60)
    
    X_train = train_data["account_features"]
    y_train = train_data["account_labels"]
    X_val = val_data["account_features"]
    y_val = val_data["account_labels"]
    X_test = test_data["account_features"]
    y_test = test_data["account_labels"]
    
    # Handle NaN/Inf
    for X in [X_train, X_val, X_test]:
        X[np.isnan(X)] = 0
        X[np.isinf(X)] = 0
    
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # Class imbalance ratio
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / max(n_pos, 1)
    print(f"  Train: {len(y_train):,} samples ({n_pos:,} SAR, {n_neg:,} normal)")
    print(f"  Class imbalance: 1:{scale_pos:.1f}")
    
    results = {}
    
    # Model 1: XGBoost (if available)
    if HAS_XGB:
        print("\n  --- XGBoost ---")
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,
            eval_metric="aucpr",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
        xgb.fit(X_train_s, y_train,
                eval_set=[(X_val_s, y_val)],
                verbose=False)
        
        y_pred = xgb.predict(X_test_s)
        y_prob = xgb.predict_proba(X_test_s)[:, 1]
        
        metrics = evaluate_model(y_test, y_pred, y_prob, "XGBoost Node Classification")
        results["xgboost"] = metrics
        
        # Feature importance
        imp = xgb.feature_importances_
        feat_names = [
            "acct_type", "is_active", "prior_sar", "initial_deposit",
            "sent_count", "recv_count", "sent_amt_mean", "sent_amt_std",
            "recv_amt_mean", "recv_amt_std", "fan_out_ratio", "fan_in_ratio",
            "recv_send_delay", "channel_diversity"
        ]
        sorted_idx = np.argsort(imp)[::-1]
        print("\n  Top 10 Features:")
        for i in range(min(10, len(sorted_idx))):
            idx = sorted_idx[i]
            name = feat_names[idx] if idx < len(feat_names) else f"feat_{idx}"
            print(f"    {i+1}. {name}: {imp[idx]:.4f}")
    
    # Model 2: Random Forest
    print("\n  --- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    y_pred_rf = rf.predict(X_test_s)
    y_prob_rf = rf.predict_proba(X_test_s)[:, 1]
    metrics_rf = evaluate_model(y_test, y_pred_rf, y_prob_rf, "Random Forest Node Classification")
    results["random_forest"] = metrics_rf
    
    # Model 3: Logistic Regression (linear baseline)
    print("\n  --- Logistic Regression ---")
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    y_prob_lr = lr.predict_proba(X_test_s)[:, 1]
    metrics_lr = evaluate_model(y_test, y_pred_lr, y_prob_lr, "Logistic Regression Node Classification")
    results["logistic_regression"] = metrics_lr
    
    return results


def train_edge_classifier(train_data, val_data, test_data):
    """Train XGBoost for suspicious transaction detection."""
    print("\n" + "="*60)
    print("EDGE CLASSIFICATION — Suspicious Transaction Detection")
    print("="*60)
    
    X_train = train_data["edge_features"]
    y_train = train_data["edge_labels"]
    X_val = val_data["edge_features"]
    y_val = val_data["edge_labels"]
    X_test = test_data["edge_features"]
    y_test = test_data["edge_labels"]
    
    # Handle NaN/Inf
    for X in [X_train, X_val, X_test]:
        X[np.isnan(X)] = 0
        X[np.isinf(X)] = 0
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / max(n_pos, 1)
    print(f"  Train: {len(y_train):,} samples ({n_pos:,} SAR, {n_neg:,} normal)")
    print(f"  Class imbalance: 1:{scale_pos:.1f}")
    
    results = {}
    
    if HAS_XGB:
        print("\n  --- XGBoost ---")
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,
            eval_metric="aucpr",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
        xgb.fit(X_train_s, y_train,
                eval_set=[(X_val_s, y_val)],
                verbose=False)
        
        y_pred = xgb.predict(X_test_s)
        y_prob = xgb.predict_proba(X_test_s)[:, 1]
        
        metrics = evaluate_model(y_test, y_pred, y_prob, "XGBoost Edge Classification")
        results["xgboost"] = metrics
        
        # Feature importance
        imp = xgb.feature_importances_
        feat_names = ["amount", "log_amount", "amount_zscore", "channel_code"]
        print("\n  Feature Importances:")
        for i, name in enumerate(feat_names):
            if i < len(imp):
                print(f"    {name}: {imp[i]:.4f}")
    
    # Random Forest
    print("\n  --- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    y_pred_rf = rf.predict(X_test_s)
    y_prob_rf = rf.predict_proba(X_test_s)[:, 1]
    metrics_rf = evaluate_model(y_test, y_pred_rf, y_prob_rf, "Random Forest Edge Classification")
    results["random_forest"] = metrics_rf
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--task", choices=["node", "edge", "both"], default="both")
    args = parser.parse_args()
    
    if not HAS_SKLEARN:
        print("[ERROR] scikit-learn required. pip install scikit-learn")
        return
    
    # Load data
    print("Loading graph data...")
    train_data = load_graph_data("train")
    val_data = load_graph_data("val")
    test_data = load_graph_data("test")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}
    
    if args.task in ["node", "both"]:
        if all(k in d for k in ["account_features", "account_labels"] for d in [train_data, val_data, test_data]):
            node_results = train_node_classifier(train_data, val_data, test_data)
            all_results["node_classification"] = node_results
        else:
            print("[WARN] Account features/labels not found. Run build_graph.py first.")
    
    if args.task in ["edge", "both"]:
        if all(k in d for k in ["edge_features", "edge_labels"] for d in [train_data, val_data, test_data]):
            edge_results = train_edge_classifier(train_data, val_data, test_data)
            all_results["edge_classification"] = edge_results
        else:
            print("[WARN] Edge features/labels not found. Run build_graph.py first.")
    
    # Save results
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to {results_path}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    for task, models in all_results.items():
        print(f"\n  {task}:")
        print(f"  {'Model':<25} {'PR-AUC':>8} {'ROC-AUC':>8} {'F1(SAR)':>8} {'Recall@1x':>10}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        for model, m in models.items():
            print(f"  {model:<25} {m.get('pr_auc', 0):>8.4f} {m.get('roc_auc', 0):>8.4f} "
                  f"{m.get('f1_sar', 0):>8.4f} {m.get('recall@1x', 0):>10.4f}")


if __name__ == "__main__":
    main()

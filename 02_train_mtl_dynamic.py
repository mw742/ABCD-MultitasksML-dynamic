#!/usr/bin/env python3
import os, json, argparse, random
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLPMultiTask(nn.Module):
    def __init__(self, input_dim, n_tasks, hidden=(256, 128), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.shared = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_tasks)

    def forward(self, x):
        return self.head(self.shared(x))


def masked_bce_with_logits(logits, y, m):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    loss = loss * m
    denom = torch.clamp(m.sum(), min=1.0)
    return loss.sum() / denom


def safe_auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    if len(y) == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_ap(y, p):
    y = np.asarray(y); p = np.asarray(p)
    if len(y) == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def subject_level_cumrisk(df_task, prob_col, y_col):
    out = []
    for iid, g in df_task.groupby("IID", sort=False):
        p = g[prob_col].values.astype(float)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        cumrisk = 1.0 - np.prod(1.0 - p)
        ytrue = int((g[y_col].values == 1).any())
        out.append((iid, ytrue, cumrisk))
    return pd.DataFrame(out, columns=["IID", "y_subject", "p_subject"])


def bootstrap_subject_ci(subj_df, metric_fn, n_boot=500, seed=123):
    rng = np.random.RandomState(seed)
    iids = subj_df["IID"].unique()
    vals = []
    subj_ix = subj_df.set_index("IID")
    for _ in range(n_boot):
        bs = rng.choice(iids, size=len(iids), replace=True)
        d = subj_ix.loc[bs].reset_index()
        y = d["y_subject"].values
        p = d["p_subject"].values
        if np.unique(y).size < 2:
            vals.append(np.nan)
        else:
            vals.append(metric_fn(y, p))
    vals = np.array(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 20:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="MTL_dynamic_dataset.tsv")
    ap.add_argument("--outdir", default="results/mtl_dynamic_run")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--perm_repeats", type=int, default=2)
    ap.add_argument("--perm_max_features", type=int, default=250)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data, sep="\t")

    for col in ["split", "IID"]:
        if col not in df.columns:
            raise ValueError(f"Dataset must have {col} column.")

    # Normalize split labels
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    splits = df["split"].values

    # Labels + masks
    label_cols = sorted([c for c in df.columns if c.startswith("y_")])
    if not label_cols:
        raise ValueError("No y_* label columns found.")
    mask_cols = [c.replace("y_", "m_", 1) for c in label_cols]
    for mc in mask_cols:
        if mc not in df.columns:
            raise ValueError(f"Missing mask column: {mc}")

    # Key/id columns to EXCLUDE from features
    # (site + interval_key were all-NaN after numeric coercion in your diagnostic)
    key_cols = [
        "FID","IID","split","src_subject_id","eventname",
        "start","stop","duration","log_start","log_stop",
        "site","interval_key"
    ]
    key_cols = [c for c in key_cols if c in df.columns]

    feature_cols = [c for c in df.columns if c not in key_cols + label_cols + mask_cols]

    # Add time features explicitly as numeric predictors (safe)
    time_cols = [c for c in ["start","stop","duration","log_start","log_stop"] if c in df.columns]

    # Build X
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")

    # attach time cols as numeric predictors
    for tc in time_cols:
        X[tc] = pd.to_numeric(df[tc], errors="coerce")

    # Drop all-NaN columns after coercion
    allnan_cols = X.columns[X.isna().all()]
    if len(allnan_cols) > 0:
        print(f"[warn] Dropping {len(allnan_cols)} all-NaN feature cols: {list(allnan_cols[:20])}", flush=True)
        X = X.drop(columns=list(allnan_cols))

    # Impute: median then 0
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    # Labels + masks numeric
    Y = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    M = df[mask_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)

    # Split indices
    ix_tr = np.where(splits == "train")[0]
    ix_va = np.where(splits == "val")[0]
    ix_te = np.where(splits == "test")[0]
    if len(ix_tr) == 0 or len(ix_va) == 0 or len(ix_te) == 0:
        vc = pd.Series(splits).value_counts(dropna=False)
        raise ValueError(f"Empty split detected. split value_counts:\n{vc}\nExpected split in {{train,val,test}}.")

    # Arrays
    X_tr, X_va, X_te = X.values[ix_tr], X.values[ix_va], X.values[ix_te]
    Y_tr, Y_va, Y_te = Y[ix_tr], Y[ix_va], Y[ix_te]
    M_tr, M_va, M_te = M[ix_tr], M[ix_va], M[ix_te]

    # Standardize using TRAIN only
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    # Hard finite checks (prevents silent NaNs -> best_state=None)
    if not np.isfinite(X_tr_s).all():
        raise ValueError("Non-finite values in X_tr after StandardScaler. Check feature coercion/imputation.")
    if not np.isfinite(X_va_s).all():
        raise ValueError("Non-finite values in X_val after StandardScaler. Check feature coercion/imputation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(16, os.cpu_count() or 1))

    Xtr_t = torch.tensor(X_tr_s, dtype=torch.float32)
    Ytr_t = torch.tensor(Y_tr, dtype=torch.float32)
    Mtr_t = torch.tensor(M_tr, dtype=torch.float32)

    Xva_t = torch.tensor(X_va_s, dtype=torch.float32)
    Yva_t = torch.tensor(Y_va, dtype=torch.float32)
    Mva_t = torch.tensor(M_va, dtype=torch.float32)

    Xte_t = torch.tensor(X_te_s, dtype=torch.float32)
    Yte_t = torch.tensor(Y_te, dtype=torch.float32)
    Mte_t = torch.tensor(M_te, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xtr_t, Ytr_t, Mtr_t), batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xva_t, Yva_t, Mva_t), batch_size=args.batch, shuffle=False)
    test_loader  = DataLoader(TensorDataset(Xte_t, Yte_t, Mte_t), batch_size=args.batch, shuffle=False)

    n_features = X_tr_s.shape[1]
    n_tasks = Y.shape[1]

    print(f"[data] rows train/val/test = {len(ix_tr)}/{len(ix_va)}/{len(ix_te)}", flush=True)
    print(f"[data] n_features={n_features} n_tasks={n_tasks}", flush=True)

    model = MLPMultiTask(input_dim=n_features, n_tasks=n_tasks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = float("inf")
    best_state = None
    bad = 0

    def eval_loader(loader):
        model.eval()
        all_logits, all_y, all_m = [], [], []
        tot = 0.0
        n_batches = 0
        with torch.no_grad():
            for xb, yb, mb in loader:
                xb = xb.to(device); yb = yb.to(device); mb = mb.to(device)
                logits = model(xb)
                loss = masked_bce_with_logits(logits, yb, mb)
                tot += loss.item()
                n_batches += 1
                all_logits.append(logits.cpu().numpy())
                all_y.append(yb.cpu().numpy())
                all_m.append(mb.cpu().numpy())
        logits = np.vstack(all_logits)
        y = np.vstack(all_y)
        m = np.vstack(all_m)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return (tot / max(1, n_batches), probs, y, m)

    # Train (early stopping on masked val loss)
    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb, mb in train_loader:
            xb = xb.to(device); yb = yb.to(device); mb = mb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = masked_bce_with_logits(logits, yb, mb)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        val_loss, _, _, _ = eval_loader(val_loader)
        if not np.isfinite(val_loss):
            raise RuntimeError("val_loss is NaN/Inf. Usually caused by non-finite features or exploding logits. "
                               "Re-check feature columns and consider lowering lr/batch.")
        print(f"[epoch {ep:03d}] val_loss={val_loss:.6f}", flush=True)

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {
                "model_state": model.state_dict(),
                "scaler": scaler,
                "feature_cols": list(X.columns),
                "label_cols": label_cols,
                "mask_cols": mask_cols,
                "seed": args.seed
            }
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[early-stop] best_val={best_val:.6f}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Training failed (no best_state). This should not happen now; check stdout for NaN/Inf guard.")

    model.load_state_dict(best_state["model_state"])

    # Test predictions
    test_loss, P_te, Y_te2, M_te2 = eval_loader(test_loader)

    # Save interval-level prediction table
    test_meta = df.iloc[ix_te][["FID","IID","split"]].copy() if "FID" in df.columns else df.iloc[ix_te][["IID","split"]].copy()
    for c in ["start","stop","duration"]:
        if c in df.columns:
            test_meta[c] = df.iloc[ix_te][c].values

    for t, lab in enumerate(label_cols):
        test_meta[lab] = Y_te2[:, t]
        test_meta[lab.replace("y_", "m_")] = M_te2[:, t]
        test_meta[lab.replace("y_", "p_")] = P_te[:, t]

    test_meta.to_csv(os.path.join(args.outdir, "predictions_test_intervals.tsv"), sep="\t", index=False)

    # Metrics
    metrics = {
        "run": {
            "seed": args.seed,
            "device": str(device),
            "n_train_rows": int(len(ix_tr)),
            "n_val_rows": int(len(ix_va)),
            "n_test_rows": int(len(ix_te)),
            "n_features": int(n_features),
            "n_tasks": int(n_tasks),
            "test_loss_masked": float(test_loss)
        },
        "interval_level": {},
        "subject_level": {},
        "baseline_lr_interval": {}
    }

    # Interval-level per task
    for t, lab in enumerate(label_cols):
        obs = (M_te2[:, t] == 1.0)
        y = Y_te2[obs, t].astype(int)
        p = P_te[obs, t]
        pred = (p >= 0.5).astype(int)

        auc = safe_auc(y, p)
        apv = safe_ap(y, p)
        brier = float(brier_score_loss(y, p)) if len(y) > 0 else float("nan")
        acc = float(accuracy_score(y, pred)) if len(y) > 0 else float("nan")

        metrics["interval_level"][lab] = {
            "n_obs_test": int(obs.sum()),
            "prev_test": float(np.mean(y)) if len(y) > 0 else float("nan"),
            "auc": auc,
            "pr_auc": apv,
            "brier": brier,
            "acc@0.5": acc
        }

    # Subject-level cumulative risk per task + bootstrap CI
    subj_rows = []
    for t, lab in enumerate(label_cols):
        ycol = lab
        mcol = lab.replace("y_", "m_")
        pcol = lab.replace("y_", "p_")

        d = test_meta[test_meta[mcol] == 1].copy()
        if d.empty:
            continue
        subj = subject_level_cumrisk(d, prob_col=pcol, y_col=ycol)

        if subj["y_subject"].nunique() < 2:
            auc = float("nan"); apv = float("nan")
            auc_ci = (float("nan"), float("nan"))
            ap_ci  = (float("nan"), float("nan"))
        else:
            auc = float(roc_auc_score(subj["y_subject"], subj["p_subject"]))
            apv = float(average_precision_score(subj["y_subject"], subj["p_subject"]))
            auc_ci = bootstrap_subject_ci(subj, roc_auc_score, n_boot=500, seed=args.seed+7)
            ap_ci  = bootstrap_subject_ci(subj, average_precision_score, n_boot=500, seed=args.seed+11)

        metrics["subject_level"][lab] = {
            "n_subjects": int(subj.shape[0]),
            "prev_subject": float(subj["y_subject"].mean()),
            "auc": auc,
            "auc_ci95": [auc_ci[0], auc_ci[1]],
            "pr_auc": apv,
            "pr_auc_ci95": [ap_ci[0], ap_ci[1]]
        }

        subj["task"] = lab
        subj_rows.append(subj)

    if subj_rows:
        subj_all = pd.concat(subj_rows, axis=0, ignore_index=True)
        subj_all.to_csv(os.path.join(args.outdir, "predictions_test_subject.tsv"), sep="\t", index=False)

    # Baseline per-task Logistic Regression (interval-level; train only where mask==1)
    for t, lab in enumerate(label_cols):
        obs_tr = (M_tr[:, t] == 1.0)
        obs_te = (M_te2[:, t] == 1.0)
        if obs_tr.sum() < 200 or np.unique(Y_tr[obs_tr, t]).size < 2:
            metrics["baseline_lr_interval"][lab] = {"note": "insufficient labels"}
            continue

        clf = LogisticRegression(solver="liblinear", penalty="l2", C=1.0, max_iter=300)
        clf.fit(X_tr_s[obs_tr], Y_tr[obs_tr, t].astype(int))
        p = clf.predict_proba(X_te_s[obs_te])[:, 1]
        y = Y_te2[obs_te, t].astype(int)

        metrics["baseline_lr_interval"][lab] = {
            "n_obs_test": int(obs_te.sum()),
            "auc": safe_auc(y, p),
            "pr_auc": safe_ap(y, p)
        }

    # Permutation importance (interval-level AUC drop), limited features to control runtime
    maxf = min(args.perm_max_features, n_features)
    feat_idx = np.arange(maxf)
    rng = np.random.RandomState(args.seed + 101)

    base_auc = {}
    for t, lab in enumerate(label_cols):
        obs = (M_te2[:, t] == 1.0)
        y = Y_te2[obs, t].astype(int)
        base_auc[lab] = safe_auc(y, P_te[obs, t])

    model.eval()
    def predict_probs(X_np):
        with torch.no_grad():
            xb = torch.tensor(X_np, dtype=torch.float32, device=device)
            logits = model(xb).detach().cpu().numpy()
        return 1.0 / (1.0 + np.exp(-logits))

    imp_rows = []
    for r in range(args.perm_repeats):
        for j in feat_idx:
            Xp = X_te_s.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            Pp = predict_probs(Xp)
            for t, lab in enumerate(label_cols):
                obs = (M_te2[:, t] == 1.0)
                y = Y_te2[obs, t].astype(int)
                auc_perm = safe_auc(y, Pp[obs, t])
                delta = (base_auc[lab] - auc_perm) if (np.isfinite(base_auc[lab]) and np.isfinite(auc_perm)) else float("nan")
                imp_rows.append({
                    "task": lab,
                    "feature": best_state["feature_cols"][j],
                    "repeat": r,
                    "auc_drop": delta
                })

    imp = pd.DataFrame(imp_rows)
    imp.to_csv(os.path.join(args.outdir, "feature_importance_permutation.tsv"), sep="\t", index=False)
    agg = imp.groupby(["task","feature"], as_index=False)["auc_drop"].mean().sort_values(["task","auc_drop"], ascending=[True, False])
    agg.to_csv(os.path.join(args.outdir, "feature_importance_mean.tsv"), sep="\t", index=False)

    # Save model + metrics + manifest
    torch.save(best_state, os.path.join(args.outdir, "mtl_dynamic_model.pt"))
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    manifest = {
        "data": args.data,
        "outdir": args.outdir,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "wd": args.wd,
        "patience": args.patience,
        "perm_repeats": args.perm_repeats,
        "perm_max_features": args.perm_max_features,
        "grad_clip": args.grad_clip,
        "label_cols": label_cols,
        "n_features": n_features
    }
    with open(os.path.join(args.outdir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("[done] wrote:", args.outdir, flush=True)


if __name__ == "__main__":
    main()

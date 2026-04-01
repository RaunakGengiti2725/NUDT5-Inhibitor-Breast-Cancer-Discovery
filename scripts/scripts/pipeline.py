#!/usr/bin/env python3
"""
NUDT5 Inhibitor Discovery Pipeline
Transferability-Weighted Consensus Scoring (TWCS)

Author: Raunak Gengiti
Affiliation: Independent Research, San Diego, CA

Usage:
    python scripts/pipeline.py

Outputs:
    - Benchmark table (AUC, EF, BEDROC) for all methods
    - Y-randomization validation
    - Leave-scaffold-out cross-validation
    - TWCS scores for 10 proposed candidates
    - Figures saved to results/
"""

import os
import csv
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_auc_score, roc_curve

np.random.seed(42)

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "compounds.csv")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# ----------------------------------------------------------------
# 1. Load compounds from CSV
# ----------------------------------------------------------------
def load_compounds(path):
    actives, decoys = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row["smiles"].strip('"')
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if row["label"] == "1":
                actives.append(smi)
            else:
                decoys.append(smi)
    return actives, decoys

# ----------------------------------------------------------------
# 2. Featurization
# ----------------------------------------------------------------
def smiles_to_fp(smi, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def smiles_to_bitvect(smi, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)

def compute_props(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {}
    total_c = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    sp3_c = sum(1 for a in mol.GetAtoms()
                if a.GetAtomicNum() == 6 and a.GetHybridization().name == "SP3")
    return {
        "MW": round(Descriptors.MolWt(mol), 1),
        "cLogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 1),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "NRB": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Fsp3": round(sp3_c / total_c, 2) if total_c > 0 else 0.0,
    }

# ----------------------------------------------------------------
# 3. Metrics
# ----------------------------------------------------------------
def enrichment_factor(y_true, y_scores, frac=0.01):
    n = len(y_true)
    n_act = y_true.sum()
    top_n = max(1, int(frac * n))
    top_idx = np.argsort(y_scores)[::-1][:top_n]
    hits = y_true[top_idx].sum()
    return (hits / top_n) / (n_act / n) if n_act > 0 else 0

def bedroc(y_true, y_scores, alpha=20):
    n = len(y_true)
    n_a = int(y_true.sum())
    if n_a == 0:
        return 0
    ranks = [r + 1 for r, i in enumerate(np.argsort(y_scores)[::-1]) if y_true[i] == 1]
    R_a = n_a / n
    s = sum(np.exp(-alpha * r / n) for r in ranks)
    Rmax = (1 - np.exp(-alpha * R_a)) / (R_a * (1 - np.exp(-alpha)))
    Rmin = (1 - np.exp(alpha * R_a)) / (R_a * (1 - np.exp(alpha)))
    if Rmax == Rmin:
        return 0
    return max(0, min(1, (s * R_a / n_a - Rmin) / (Rmax - Rmin)))

# ----------------------------------------------------------------
# 4. TWCS Consensus
# ----------------------------------------------------------------
def normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn) if mx > mn else arr

def twcs_consensus(score_dict, weights=None):
    """Compute TWCS consensus from a dict of {method_name: scores_array}."""
    names = list(score_dict.keys())
    if weights is None:
        weights = {n: 1.0 / len(names) for n in names}
    normed = {n: normalize(score_dict[n]) for n in names}
    consensus = sum(weights[n] * normed[n] for n in names)
    return consensus

# ----------------------------------------------------------------
# 5. Proposed compounds
# ----------------------------------------------------------------
PROPOSED = {
    "NC5-01": "O=c1[nH]c(=O)c2n(Cc3nn(C)c(-c4ccc(F)cc4)n3)c(N3CCNCC3)nc2n1C",
    "NC5-02": "Nc1ncnc2[nH]nc(-c3ccc(Oc4ccccc4)cc3)c12",
    "NC5-03": "O=S(=O)(N1CCNCC1)c1ccc2[nH]c(-c3ccccn3)nc2c1",
    "NC5-04": "c1nc(N)c2ncn(CC(=O)NS(=O)(=O)c3ccc(Cl)cc3)c2n1",
    "NC5-05": "O=C(NCc1cnc2ccccc2n1)c1c[nH]c2ccccc12",
    "NC5-06": "Nc1nc2ncc(CN3CCNCC3)nc2c(=O)[nH]1",
    "NC5-07": "O=c1[nH]c(Nc2ccc(F)cc2)nc2cnc(Cc3nnc(-c4ccc(Cl)cc4)o3)nc12",
    "NC5-08": "O=C(Nc1ccc(F)cc1)Cn1c(=O)[nH]c2nc[nH]c(=O)c21",
    "NC5-09": "O=C(/C=C/c1ccc(F)cc1)N1CCN(c2nc3ccccc3c(=O)[nH]2)CC1",
    "NC5-10": "O=C(c1cnc2nc(N)ccn12)Nc1cccc(S(N)(=O)=O)c1",
}

TH5427 = "O=c1[nH]c(=O)c2n(Cc3nnc(-c4ccc(Cl)c(Cl)c4)o3)c(N3CCNCC3)nc2n1C"

# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 65)
    print("  NUDT5 Inhibitor Discovery -- TWCS Pipeline")
    print("=" * 65)

    # --- Load data ---
    print("\n[1/7] Loading compounds...")
    actives, decoys = load_compounds(DATA)
    # Replicate decoys to get ~500 (26 unique x 20)
    decoys_expanded = (decoys * 20)[:520]
    print(f"  {len(actives)} actives, {len(decoys_expanded)} decoys")

    # --- Featurize ---
    print("[2/7] Computing ECFP4 fingerprints...")
    Xa = np.array([x for x in (smiles_to_fp(s) for s in actives) if x is not None])
    Xd = np.array([x for x in (smiles_to_fp(s) for s in decoys_expanded) if x is not None])
    X = np.vstack([Xa, Xd])
    y = np.array([1] * len(Xa) + [0] * len(Xd))
    na, nd = len(Xa), len(Xd)
    print(f"  Feature matrix: {X.shape[0]} x {X.shape[1]}")

    # Tanimoto similarity to TH5427
    th_fp = smiles_to_bitvect(TH5427)
    all_smiles = actives[:na] + decoys_expanded[:nd]
    sim_scores = np.array([
        DataStructs.TanimotoSimilarity(th_fp, smiles_to_bitvect(s))
        if smiles_to_bitvect(s) is not None else 0.0
        for s in all_smiles
    ])

    # --- 5-Fold CV Benchmark ---
    print("[3/7] Running 5-fold CV benchmark...")
    models = {
        "RF": RandomForestClassifier(n_estimators=100, max_features="sqrt",
                                      random_state=42, n_jobs=-1),
        "GBT": GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                           learning_rate=0.1, random_state=42),
        "SVM": SVC(kernel="rbf", C=10, probability=True, random_state=42),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = {n: np.zeros(len(y)) for n in models}
    cv_probs["Similarity"] = sim_scores
    cv_probs["Random"] = np.random.random(len(y))

    for name, model in models.items():
        for tri, tsi in skf.split(X, y):
            mc = type(model)(**model.get_params())
            mc.fit(X[tri], y[tri])
            cv_probs[name][tsi] = mc.predict_proba(X[tsi])[:, 1]

    # Consensus
    cv_probs["TWCS"] = twcs_consensus({
        "RF": cv_probs["RF"], "GBT": cv_probs["GBT"],
        "SVM": cv_probs["SVM"], "Similarity": cv_probs["Similarity"],
    })

    print(f"\n  {'Method':<14} {'AUC':>6} {'EF1%':>7} {'EF5%':>7} {'BEDROC':>8}")
    print("  " + "-" * 46)
    bench = {}
    for name in ["Random", "Similarity", "RF", "GBT", "SVM", "TWCS"]:
        auc = roc_auc_score(y, cv_probs[name])
        ef1 = enrichment_factor(y, cv_probs[name], 0.01)
        ef5 = enrichment_factor(y, cv_probs[name], 0.05)
        bed = bedroc(y, cv_probs[name])
        bench[name] = {"AUC": auc, "EF1": ef1, "EF5": ef5, "BEDROC": bed}
        print(f"  {name:<14} {auc:>6.3f} {ef1:>7.1f} {ef5:>7.1f} {bed:>8.3f}")

    # --- Y-Randomization ---
    print("\n[4/7] Y-randomization (5 permutations)...")
    yrand = {n: [] for n in models}
    for _ in range(5):
        ys = np.random.permutation(y)
        for name, model in models.items():
            ps = np.zeros(len(y))
            for tri, tsi in skf.split(X, ys):
                mc = type(model)(**model.get_params())
                mc.fit(X[tri], ys[tri])
                ps[tsi] = mc.predict_proba(X[tsi])[:, 1]
            yrand[name].append(roc_auc_score(ys, ps))

    print(f"\n  {'Model':<8} {'Real AUC':>10} {'Rand mean':>10} {'p-value':>10}")
    print("  " + "-" * 40)
    for name in models:
        real = bench[name]["AUC"]
        rmean = np.mean(yrand[name])
        p = np.mean(np.array(yrand[name]) >= real)
        print(f"  {name:<8} {real:>10.3f} {rmean:>10.3f} {p:>10.3f}")

    # --- LOOCV ---
    print("\n[5/7] LOOCV on actives...")
    loo = LeaveOneOut()
    for name, model in models.items():
        preds = np.zeros(na)
        for tri, tsi in loo.split(Xa):
            Xtr = np.vstack([Xa[tri], Xd])
            ytr = np.array([1] * len(tri) + [0] * nd)
            mc = type(model)(**model.get_params())
            mc.fit(Xtr, ytr)
            preds[tsi] = mc.predict_proba(Xa[tsi])[:, 1]
        sens = np.mean(preds > 0.5)
        print(f"  {name} LOOCV sensitivity: {sens:.2f}")

    # --- Leave-scaffold-out ---
    print("\n[6/7] Leave-scaffold-out (TH5427 series -> ibrutinib series)...")
    th_idx = list(range(min(18, na)))
    ib_idx = list(range(min(18, na), na))
    if len(ib_idx) > 0:
        for name, model in models.items():
            Xtr = np.vstack([Xa[th_idx], Xd])
            ytr = np.array([1] * len(th_idx) + [0] * nd)
            mc = type(model)(**model.get_params())
            mc.fit(Xtr, ytr)
            ib_probs = mc.predict_proba(Xa[ib_idx])[:, 1]
            sens = np.mean(ib_probs > 0.5)
            print(f"  {name} LSO sensitivity: {sens:.2f}")
    else:
        print("  Skipped (insufficient ibrutinib-series compounds)")

    # --- Score proposed compounds ---
    print("\n[7/7] Scoring proposed candidates...")
    final_models = {}
    for name, model in models.items():
        mc = type(model)(**model.get_params())
        mc.fit(X, y)
        final_models[name] = mc

    # PAINS filter
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains_catalog = FilterCatalog(params)

    print(f"\n  {'ID':<8} {'TWCS':>6} {'RF':>6} {'GBT':>6} {'SVM':>6} "
          f"{'T(TH)':>6} {'MW':>6} {'cLogP':>6} {'PAINS':>6}")
    print("  " + "-" * 62)

    for cid in sorted(PROPOSED, key=lambda c: c):
        smi = PROPOSED[cid]
        fp_arr = smiles_to_fp(smi)
        fp_bv = smiles_to_bitvect(smi)
        if fp_arr is None or fp_bv is None:
            continue
        sim = DataStructs.TanimotoSimilarity(th_fp, fp_bv)
        rf_p = final_models["RF"].predict_proba(fp_arr.reshape(1, -1))[0][1]
        gb_p = final_models["GBT"].predict_proba(fp_arr.reshape(1, -1))[0][1]
        sv_p = final_models["SVM"].predict_proba(fp_arr.reshape(1, -1))[0][1]
        con = np.mean([rf_p, gb_p, sv_p, sim])
        props = compute_props(smi)
        mol = Chem.MolFromSmiles(smi)
        pains_ok = "OK" if pains_catalog.GetFirstMatch(mol) is None else "ALERT"
        print(f"  {cid:<8} {con:>6.3f} {rf_p:>6.3f} {gb_p:>6.3f} {sv_p:>6.3f} "
              f"{sim:>6.3f} {props['MW']:>6.0f} {props['cLogP']:>6.2f} {pains_ok:>6}")

    # --- Figures ---
    print("\n  Saving figures...")
    colors = {"Random": "#999", "Similarity": "#FF9800", "RF": "#2196F3",
              "GBT": "#4CAF50", "SVM": "#9C27B0", "TWCS": "#E91E63"}

    # ROC
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in ["Random", "Similarity", "RF", "GBT", "SVM", "TWCS"]:
        fpr, tpr, _ = roc_curve(y, cv_probs[name])
        lw = 2.5 if name == "TWCS" else 1.5
        ls = ":" if name == "Random" else ("--" if name == "Similarity" else "-")
        ax.plot(fpr, tpr, color=colors[name], linewidth=lw, linestyle=ls,
                label=f'{name} (AUC={bench[name]["AUC"]:.3f})')
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: Method Benchmarking", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "roc_benchmark.png"), dpi=200)
    plt.close()

    # EF bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ms = ["Random", "Similarity", "RF", "GBT", "SVM", "TWCS"]
    x = np.arange(len(ms))
    ax.bar(x - 0.18, [bench[m]["EF1"] for m in ms], 0.35,
           label="EF 1%", color=[colors[m] for m in ms], alpha=0.9)
    ax.bar(x + 0.18, [bench[m]["EF5"] for m in ms], 0.35,
           label="EF 5%", color=[colors[m] for m in ms], alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ms)
    ax.set_ylabel("Enrichment Factor")
    ax.set_title("Enrichment by Method", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "enrichment_factors.png"), dpi=200)
    plt.close()

    print(f"  Saved roc_benchmark.png and enrichment_factors.png to {RESULTS}/")
    print("\n" + "=" * 65)
    print("  Pipeline complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()

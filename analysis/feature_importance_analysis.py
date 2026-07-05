# -*- coding: utf-8 -*-
"""
feature_importance_analysis.py  (Task C — R4-7)
Group-level feature importance for the classification RF model.
Groups: Elemental/Compositional (cols 0-391), Crystal System (cols 392-398),
        Structural/Lattice (cols 399-407), Sine-matrix eigenvalues (cols 408-599),
        CGCNN embeddings (cols 600-631)

Input: core_data+code/active_learning_csv/input_labeled_classification_1323.csv
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(
    'core_data+code/active_learning_csv/input_labeled_classification_1323.csv'
)
df = df.dropna()
X = df.iloc[:, 1:-1].values   # 634 features
y_raw = df.iloc[:, -1].values
y = (y_raw >= 0).astype(int)
feature_names = list(df.columns[1:-1])

print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")

# ── Scale ──────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 5-fold CV feature importance ───────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
importances_list = []

print("Computing feature importance (5-fold CV)...")
for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
    print(f"  Fold {fold}/5...", flush=True)
    clf = RandomForestClassifier(
        n_estimators=500, max_depth=10, random_state=42, n_jobs=-1
    )
    clf.fit(X_scaled[train_idx], y[train_idx])
    importances_list.append(clf.feature_importances_)

mean_imp = np.mean(importances_list, axis=0)
std_imp  = np.std(importances_list, axis=0)

# ── Feature group definitions (matching regression+classification.py) ──────────
# Col indices (0-based, after removing ID col from original CSV)
# Original file: [ID | 0-391 compositional | 392-398 crystal_sys (7) |
#                 399-407 lattice/structural (9) | 408-599 sine-matrix (192) |
#                 600-631 CGCNN (32) | reference]
# After iloc[:, 1:-1]: feature index = original_col - 1

group_defs = {
    'Elemental/\nCompositional\n(n=392)': list(range(0, 392)),
    'Crystal\nSystem\n(n=7)':             list(range(392, 399)),
    'Structural\nLattice\n(n=9)':         list(range(399, 408)),
    'Sine-Matrix\nEigenvalues\n(n=192)':  list(range(408, 600)),
    'CGCNN\nEmbeddings\n(n=32)':          list(range(600, 632)),
}

group_importance = {}
group_importance_std = {}
for group_name, indices in group_defs.items():
    group_importance[group_name]     = mean_imp[indices].sum()
    group_importance_std[group_name] = np.sqrt((std_imp[indices]**2).sum())

total = sum(group_importance.values())

print()
print("=== Feature Group Importance (Classification) ===")
print(f"{'Group':<35} {'Importance':>12}  {'% of total':>12}")
print("-"*62)
for g, imp in sorted(group_importance.items(), key=lambda x: -x[1]):
    pct = 100 * imp / total
    print(f"{g.replace(chr(10),' '):<35} {imp:>12.6f}  {pct:>11.1f}%")

print()
print("=== Key answer for R4-7 ===")
cgcnn_pct  = 100 * group_importance['CGCNN\nEmbeddings\n(n=32)'] / total
comp_pct   = 100 * group_importance['Elemental/\nCompositional\n(n=392)'] / total
sm_pct     = 100 * group_importance['Sine-Matrix\nEigenvalues\n(n=192)'] / total
cryst_pct  = 100 * group_importance['Crystal\nSystem\n(n=7)'] / total
struct_pct = 100 * group_importance['Structural\nLattice\n(n=9)'] / total

print(f"CGCNN embeddings (32 features):     {cgcnn_pct:.1f}%")
print(f"Elemental/compositional (392 feat): {comp_pct:.1f}%")
print(f"Sine-matrix eigenvalues (192 feat): {sm_pct:.1f}%")
print(f"Crystal system (7 feat):            {cryst_pct:.1f}%")
print(f"Structural/lattice (9 feat):        {struct_pct:.1f}%")
k_cgcnn  = 'CGCNN\nEmbeddings\n(n=32)'
k_comp   = 'Elemental/\nCompositional\n(n=392)'
k_sm     = 'Sine-Matrix\nEigenvalues\n(n=192)'
print()
print("Per-feature mean importance:")
print(f"  CGCNN:          {group_importance[k_cgcnn] / 32 * 1000:.4f} x10^-3")
print(f"  Elemental/comp: {group_importance[k_comp] / 392 * 1000:.4f} x10^-3")
print(f"  Sine-matrix:    {group_importance[k_sm] / 192 * 1000:.4f} x10^-3")

# ── Top 20 individual features ────────────────────────────────────────────────
top_idx = np.argsort(mean_imp)[::-1][:20]
print()
print("=== Top 20 individual features ===")
for rank, idx in enumerate(top_idx, 1):
    print(f"  {rank:>2}. [{idx:>3}] {feature_names[idx]:<40}  {mean_imp[idx]:.5f}")

# ── Bar chart ─────────────────────────────────────────────────────────────────
groups_ordered = sorted(group_importance.items(), key=lambda x: -x[1])
labels = [g[0] for g in groups_ordered]
values = [g[1] / total * 100 for g in groups_ordered]
errors = [group_importance_std[g[0]] / total * 100 for g in groups_ordered]

colors = ['#4472C4', '#ED7D31', '#A9D18E', '#FF0000', '#FFC000']
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(range(len(labels)), values, yerr=errors, capsize=5,
              color=colors[:len(labels)], edgecolor='black', linewidth=0.7)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Cumulative Feature Importance (%)', fontsize=11)
ax.set_title('Feature Group Importance — Classification Model (RF, 5-fold CV)', fontsize=11)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('feature_importance_classification.png', dpi=200, bbox_inches='tight')
print()
print("Saved: feature_importance_classification.png")

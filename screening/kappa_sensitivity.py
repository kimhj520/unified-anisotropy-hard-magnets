# -*- coding: utf-8 -*-
"""
kappa_sensitivity.py  (Task A — R1-iv)
κ threshold sensitivity analysis for DFT-confirmed and ML-predicted candidates.
Uses:
  - core_data+code/mattergen/raw+mag_labeled_mattergen_TM_24_realmag.csv (24 DFT MatterGen)
  - core_data+code/DFT_validation_predicted_structs/update_calculated_mca_merged_68+mag.csv (68 DFT MP)
  - core_data+code/active_learning_csv/raw+mag_unlabeled_predicted_7575.csv (7575 ML predictions)
  - core_data+code/mattergen/raw+mag_unlabeled_predicted_mattergen_256.csv (256 ML MatterGen)
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

# ── Load DFT-confirmed MatterGen (realmag) ─────────────────────────────────────
df_mg_dft = pd.read_csv(
    'core_data+code/mattergen/raw+mag_labeled_mattergen_TM_24_realmag.csv',
    header=None
)
df_mg_dft.columns = ['id','formula','col2','col3','col4','col5','Keff','kappa']

# ── Load DFT-confirmed MP ──────────────────────────────────────────────────────
df_mp_dft = pd.read_csv(
    'core_data+code/DFT_validation_predicted_structs/update_calculated_mca_merged_68+mag.csv',
    header=None
)
df_mp_dft.columns = ['id','formula','col2','col3','col4','col5','Keff','kappa']

# ── Load ML-predicted unlabeled MP (7575) ─────────────────────────────────────
df_mp_ml = pd.read_csv(
    'core_data+code/active_learning_csv/raw+mag_unlabeled_predicted_7575.csv',
    header=None
)
df_mp_ml.columns = ['id','formula','col2','col3','col4','col5','Keff','kappa']

# ── Load ML-predicted MatterGen (251 non-null) ────────────────────────────────
df_mg_ml = pd.read_csv(
    'core_data+code/mattergen/raw+mag_unlabeled_predicted_mattergen_256.csv',
    header=None
)
df_mg_ml.columns = ['id','formula','col2','col3','col4','col5','Keff','kappa']
df_mg_ml = df_mg_ml.dropna(subset=['kappa'])

print("=== Data loaded ===")
print(f"  MP DFT validated:       {len(df_mp_dft)} structures")
print(f"  MatterGen DFT validated:{len(df_mg_dft)} structures")
print(f"  MP ML-predicted:        {len(df_mp_ml)} structures")
print(f"  MatterGen ML-predicted: {len(df_mg_ml)} structures (non-null)")
print()

thresholds = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5]

print("=== DFT-confirmed κ sensitivity ===")
print(f"{'κ threshold':<14} {'MP DFT (n=68)':<18} {'MatterGen DFT (n=24)':<22} {'Total DFT'}")
print("-"*70)
for t in thresholds:
    mp_n   = (df_mp_dft['kappa'] > t).sum()
    mg_n   = (df_mg_dft['kappa'] > t).sum()
    total  = mp_n + mg_n
    marker = " ← current" if abs(t - 1.0) < 0.01 else ""
    print(f"κ > {t:<10.1f} {mp_n:<18} {mg_n:<22} {total}{marker}")

print()
print("=== ML-predicted κ sensitivity (screening stage, before DFT) ===")
print(f"{'κ threshold':<14} {'MP ML (n=7575)':<20} {'MatterGen ML (n=256)':<24}")
print("-"*60)
for t in thresholds:
    mp_n   = (df_mp_ml['kappa'] > t).sum()
    mg_n   = (df_mg_ml['kappa'] > t).sum()
    marker = " ← current" if abs(t - 1.0) < 0.01 else ""
    print(f"κ > {t:<10.1f} {mp_n:<20} {mg_n:<24}{marker}")

print()

# ── List the 8 DFT-confirmed MatterGen hard magnets with κ values ───────────
hard_mg = df_mg_dft[df_mg_dft['kappa'] > 1.0].sort_values('kappa', ascending=False)
print("=== 8 DFT-confirmed MatterGen hard magnets (κ > 1.0) ===")
print(f"{'ID':<12} {'Formula':<20} {'K_eff (MJ/m³)':<18} {'κ'}")
print("-"*60)
for _, row in hard_mg.iterrows():
    print(f"{row['id']:<12} {row['formula']:<20} {row['Keff']:>12.3f}      {row['kappa']:.4f}")
print()
print(f"At κ > 1.2: {(hard_mg['kappa']>1.2).sum()} of 8 remain")
print(f"At κ > 0.9: {(df_mg_dft['kappa']>0.9).sum()} total pass (including near-borderline)")
print()

# ── Summary table for SI ────────────────────────────────────────────────────
print("=== SI Table: κ threshold sensitivity summary ===")
print()
print("| κ threshold | DFT-confirmed (MP) | DFT-confirmed (Generated) |")
print("|-------------|-------------------|--------------------------|")
for t in [0.8, 1.0, 1.2]:
    mp_n = (df_mp_dft['kappa'] > t).sum()
    mg_n = (df_mg_dft['kappa'] > t).sum()
    marker = " (current criterion)" if abs(t - 1.0) < 0.01 else ""
    print(f"| κ > {t}      | {mp_n}                 | {mg_n}{marker}                     |")

print()
print("Note: DFT-confirmed MP structures were selected by ML screening (κ_ML > 0.5).")
print("DFT-confirmed MatterGen uses experimental Ms (realmag) for κ calculation.")

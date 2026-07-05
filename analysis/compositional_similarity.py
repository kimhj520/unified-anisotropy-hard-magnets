"""
Compositional similarity analysis: Generated 7 vs MP 46 confirmed hard magnets.

R3 비판 ("생성 구조가 MP 고성능과 유사") 반박용 정량 분석.

조성 유사도 메트릭:
  1. Element-presence Jaccard distance (집합 유사도)
  2. Atomic-fraction cosine distance (정량 유사도)
  3. Magpie-style elemental property vector (electronegativity, valence, atomic radius
     평균 + std 가중치) → Euclidean distance
"""
import sys
import os
import json
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.manifold import MDS

sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Inputs from SI Table S3 / Table S4
# ---------------------------------------------------------------------------

# MP 46 confirmed (composition, crystal, MP-ID, Ms[T], K_eff[MJ/m3])
MP46 = [
    # Light RE
    ("Ce4Mn2N6",     "Orthorhombic", "mp-10068",   0.562, 1.09),
    ("Pr4Mn4Se4O6",  "Tetragonal",   "mp-1078664", 0.702, 3.21),
    ("Pr4Fe4S4O6",   "Tetragonal",   "mp-1079573", 0.623, 74.2),
    ("Pr4Fe4Se4O6",  "Tetragonal",   "mp-1080777", 0.582, 5.03),
    ("BaPrMn2O6",    "Tetragonal",   "mp-19274",   0.660, 4.03),
    ("BaNdCo2O5",    "Tetragonal",   "mp-20926",   0.559, 2.38),
    ("BaPrCo2O6",    "Tetragonal",   "mp-22751",   0.605, 1.81),
    ("Nd4Fe4Se4O6",  "Tetragonal",   "mp-1078182", 0.590, 0.44),
    ("Pr2Fe4Ge4",    "Tetragonal",   "mp-21221",   0.702, 2.47),
    ("Pr2Mn4Ge4",    "Tetragonal",   "mp-20542",   0.925, 3.03),
    ("Pr2Mn4Si4",    "Tetragonal",   "mp-5423",    0.932, 3.02),
    ("BaNdMn2O5",    "Tetragonal",   "mp-546665",  0.817, 6.33),
    ("BaPrMn2O5",    "Tetragonal",   "mp-549776",  0.811, 6.37),
    ("BaNdFe2O5",    "Tetragonal",   "mp-638374",  0.832, 58.4),
    # No RE
    ("Cr2Fe4Se8",    "Monoclinic",   "mp-568349",  0.684, 0.88),
    ("Na6Mn8Te12",   "Monoclinic",   "mp-29770",   0.539, 1.47),
    ("Mn4B8W2",      "Orthorhombic", "mp-1077988", 0.677, 1.06),
    ("Fe4C2",        "Orthorhombic", "mp-1871",    0.666, 0.79),
    ("Co28As8O48",   "Orthorhombic", "mp-17489",   0.527, 0.56),
    ("Sr6Fe4O10",    "Orthorhombic", "mp-19218",   0.587, 0.56),
    ("Co6B4O12",     "Orthorhombic", "mp-22632",   0.507, 0.91),
    ("MnGa",         "Tetragonal",   "mp-1001836", 1.227, 2.32),
    ("FePd",         "Tetragonal",   "mp-2831",    0.749, 1.52),
    ("FeCuPt2",      "Tetragonal",   "mp-3702",    0.898, 5.22),
    ("Cs2Mn4P4",     "Tetragonal",   "mp-11181",   0.675, 0.78),
    ("Mn2Au8",       "Tetragonal",   "mp-12565",   0.596, 0.67),
    ("FeNiPt2",      "Tetragonal",   "mp-13463",   1.117, 1.30),
    ("Tl2Fe4Se4",    "Tetragonal",   "mp-3021",    0.636, 0.77),
    ("Mn9Au31",      "Tetragonal",   "mp-30411",   0.664, 1.30),
    ("MnPt",         "Tetragonal",   "mp-1670",    0.879, 11.2),
    ("FePt",         "Tetragonal",   "mp-2260",    1.481, 16.3),
    ("MnAl",         "Tetragonal",   "mp-771",     0.586, 1.78),
    ("CoPt",         "Tetragonal",   "mp-949",     0.525, 6.54),
    ("Mn8Cd4O16",    "Tetragonal",   "mp-18720",   0.555, 3.22),
    ("Sr6Fe4O14",    "Tetragonal",   "mp-18820",   0.627, 0.57),
    ("Sr4Fe2Mo2O12", "Tetragonal",   "mp-18857",   0.546, 0.26),
    ("Sr2Mn4Ge4",    "Tetragonal",   "mp-21118",   0.538, 2.28),
    ("Na3Mo3O6",     "Trigonal",     "mp-578610",  0.740, 6.27),
    ("Fe3Cl6",       "Trigonal",     "mp-23229",   0.711, 0.41),
    ("K3Cr3O6",      "Trigonal",     "mp-546552",  0.726, 0.48),
    ("CrB2",         "Hexagonal",    "mp-374",     0.510, 0.90),
    ("Fe2W2N4",      "Hexagonal",    "mp-29076",   0.531, 2.74),
    ("TiFe6Ge6",     "Hexagonal",    "mp-22130",   0.724, 1.72),
    ("Mn2Ga2Pt2",    "Hexagonal",    "mp-569151",  0.519, 3.08),
    ("Fe2Ag2O4",     "Hexagonal",    "mp-18966",   0.588, 2.29),
    ("Mn4Ge2",       "Hexagonal",    "mp-20473",   0.683, 1.77),
]

# Generated 7 (composition, crystal, gen_ID, Ms, K_eff, MP_relationship)
GEN7 = [
    ("Mn3Ga",   "Tetragonal", "gen_58",  1.199, 2.12,  "MP_reported"),
    ("CoPt",    "Trigonal",   "gen_173", 0.899, 8.54,  "MP_reported"),
    ("ScFe2",   "Hexagonal",  "gen_117", 0.738, 0.85,  "MP_reported"),
    ("FePt2",   "Tetragonal", "gen_168", 0.947, 7.73,  "Novel_composition"),
    ("FePdPt",  "Tetragonal", "gen_20",  0.966, 5.83,  "Novel_composition"),
    ("FePtRh2", "Tetragonal", "gen_22",  0.538, 3.72,  "Novel_composition"),
    ("MnFePt2", "Tetragonal", "gen_18",  1.480, 20.5,  "Novel_polymorph"),
]


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------

def fraction_vector(comp_str):
    """Return atomic fraction vector indexed by atomic number 1..103."""
    comp = Composition(comp_str)
    frac = np.zeros(104)
    for el, amt in comp.get_el_amt_dict().items():
        frac[Element(el).Z] = amt
    return frac / frac.sum()


def presence_set(comp_str):
    return set(Composition(comp_str).get_el_amt_dict().keys())


# Magpie-style elemental properties (averaged + std-weighted)
PROP_NAMES = ['X', 'Z', 'atomic_mass', 'group', 'row',
              'atomic_radius_calculated', 'mendeleev_no']


def magpie_vector(comp_str):
    comp = Composition(comp_str)
    frac = comp.fractional_composition.get_el_amt_dict()
    vec = []
    for prop in PROP_NAMES:
        vals = []
        weights = []
        for el, f in frac.items():
            try:
                v = getattr(Element(el), prop)
                if v is not None:
                    vals.append(float(v))
                    weights.append(f)
            except Exception:
                pass
        if vals:
            vals = np.array(vals)
            weights = np.array(weights) / np.array(weights).sum()
            mean = (vals * weights).sum()
            std = np.sqrt(((vals - mean) ** 2 * weights).sum())
            vec.extend([mean, std])
        else:
            vec.extend([0, 0])
    return np.array(vec)


def jaccard_distance(set1, set2):
    if not (set1 or set2):
        return 0.0
    return 1.0 - len(set1 & set2) / len(set1 | set2)


def cosine_distance(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n == 0:
        return 1.0
    return 1.0 - float(np.dot(v1, v2) / n)


def euclidean(v1, v2):
    return float(np.linalg.norm(v1 - v2))


# ---------------------------------------------------------------------------
# Compute matrices
# ---------------------------------------------------------------------------

mp_labels = [m[0] for m in MP46]
mp_ids = [m[2] for m in MP46]
gen_labels = [g[0] for g in GEN7]
gen_ids = [g[2] for g in GEN7]
gen_relation = [g[5] for g in GEN7]

mp_frac = np.array([fraction_vector(m[0]) for m in MP46])
gen_frac = np.array([fraction_vector(g[0]) for g in GEN7])

mp_sets = [presence_set(m[0]) for m in MP46]
gen_sets = [presence_set(g[0]) for g in GEN7]

mp_magpie = np.array([magpie_vector(m[0]) for m in MP46])
gen_magpie = np.array([magpie_vector(g[0]) for g in GEN7])

# Normalize magpie features
all_magpie = np.vstack([mp_magpie, gen_magpie])
mu, sigma = all_magpie.mean(0), all_magpie.std(0)
sigma[sigma == 0] = 1
mp_magpie_z = (mp_magpie - mu) / sigma
gen_magpie_z = (gen_magpie - mu) / sigma

# Distance matrices: rows = gen, cols = MP
D_jacc = np.array([[jaccard_distance(g, m) for m in mp_sets] for g in gen_sets])
D_cos = np.array([[cosine_distance(g, m) for m in mp_frac] for g in gen_frac])
D_magpie = np.array([[euclidean(g, m) for m in mp_magpie_z] for g in gen_magpie_z])

# Internal MP-MP distances (reference distribution)
def upper_off_diag(matrix_fn, vectors_or_sets):
    n = len(vectors_or_sets)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(matrix_fn(vectors_or_sets[i], vectors_or_sets[j]))
    return np.array(out)


mp_mp_jacc = upper_off_diag(jaccard_distance, mp_sets)
mp_mp_cos = upper_off_diag(cosine_distance, mp_frac)
mp_mp_magpie = upper_off_diag(euclidean, mp_magpie_z)

# Min-distance to MP for each gen
gen_min_jacc = D_jacc.min(axis=1)
gen_min_cos = D_cos.min(axis=1)
gen_min_magpie = D_magpie.min(axis=1)

# Stats: is gen's min-distance to MP significantly larger than MP-MP nearest-neighbor distance?
def nearest_neighbor_distances(vectors_or_sets, dist_fn):
    n = len(vectors_or_sets)
    nns = []
    for i in range(n):
        best = np.inf
        for j in range(n):
            if i == j:
                continue
            d = dist_fn(vectors_or_sets[i], vectors_or_sets[j])
            if d < best:
                best = d
        nns.append(best)
    return np.array(nns)


mp_mp_nn_jacc = nearest_neighbor_distances(mp_sets, jaccard_distance)
mp_mp_nn_cos = nearest_neighbor_distances(mp_frac, cosine_distance)
mp_mp_nn_magpie = nearest_neighbor_distances(mp_magpie_z, euclidean)

print("=== Compositional Similarity Analysis ===")
print(f"\n7 Generated vs 46 MP confirmed hard magnets")
print(f"  Min-distance per gen candidate (smaller = more similar to some MP):")
print(f"\n  {'gen_ID':<10} {'Formula':<10} {'Relation':<20} {'Jaccard':<10} {'CosFrac':<10} {'Magpie':<10}")
for i, (lbl, gid, rel) in enumerate(zip(gen_labels, gen_ids, gen_relation)):
    print(f"  {gid:<10} {lbl:<10} {rel:<20} {gen_min_jacc[i]:<10.3f} {gen_min_cos[i]:<10.3f} {gen_min_magpie[i]:<10.3f}")

# Mann-Whitney U: gen-vs-MP min distance > MP-MP NN distance ?
def mwu(a, b, name):
    u, p = mannwhitneyu(a, b, alternative='greater')
    return f"  {name}: gen-min ({a.mean():.3f}±{a.std():.3f}) vs MP-NN ({b.mean():.3f}±{b.std():.3f}); MWU p={p:.4g}"

print("\n=== Mann-Whitney U test (alternative: gen min-distance > MP nearest-neighbor distance) ===")
print(mwu(gen_min_jacc, mp_mp_nn_jacc, "Jaccard (element presence)"))
print(mwu(gen_min_cos, mp_mp_nn_cos, "Cosine (atomic fractions)"))
print(mwu(gen_min_magpie, mp_mp_nn_magpie, "Magpie (elemental properties)"))

# ---------------------------------------------------------------------------
# 2D Visualization via MDS on combined Magpie distance matrix
# ---------------------------------------------------------------------------
all_magpie_z = np.vstack([mp_magpie_z, gen_magpie_z])
n_mp = len(mp_magpie_z)
n_gen = len(gen_magpie_z)
D_all = np.zeros((n_mp + n_gen, n_mp + n_gen))
for i in range(n_mp + n_gen):
    for j in range(n_mp + n_gen):
        D_all[i, j] = euclidean(all_magpie_z[i], all_magpie_z[j])

mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
coords = mds.fit_transform(D_all)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
mp_x, mp_y = coords[:n_mp, 0], coords[:n_mp, 1]
gen_x, gen_y = coords[n_mp:, 0], coords[n_mp:, 1]

ax.scatter(mp_x, mp_y, c='steelblue', s=40, alpha=0.7, edgecolor='k',
           linewidth=0.4, label=f'MP confirmed (n={n_mp})')

# Color gen by relation
rel_colors = {'MP_reported': 'gray', 'Novel_composition': 'crimson',
              'Novel_polymorph': 'darkorange'}
for x, y, lbl, gid, rel in zip(gen_x, gen_y, gen_labels, gen_ids, gen_relation):
    ax.scatter(x, y, c=rel_colors[rel], s=180, marker='*',
               edgecolor='k', linewidth=0.8, zorder=3)
    ax.annotate(lbl, (x, y), xytext=(6, 6), textcoords='offset points',
                fontsize=9, fontweight='bold')

from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
           markersize=8, label=f'MP confirmed (n={n_mp})'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
           markersize=12, label='Generated — MP-reported (3)'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='crimson',
           markersize=12, label='Generated — novel composition (3)'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='darkorange',
           markersize=12, label='Generated — novel polymorph (1)'),
]
ax.legend(handles=legend_elems, loc='best', fontsize=9, frameon=True)
ax.set_xlabel('MDS dim 1 (compositional)')
ax.set_ylabel('MDS dim 2 (compositional)')
ax.set_title('Compositional space: generated vs MP confirmed hard magnets')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'compositional_mds.png'), dpi=160, bbox_inches='tight')
print(f"\nSaved: {os.path.join(ROOT, 'compositional_mds.png')}")

# Save raw distance data
np.savez(os.path.join(ROOT, 'compositional_distances.npz'),
         mp_labels=mp_labels, mp_ids=mp_ids,
         gen_labels=gen_labels, gen_ids=gen_ids, gen_relation=gen_relation,
         D_jaccard=D_jacc, D_cosine=D_cos, D_magpie=D_magpie,
         mp_mp_jaccard=mp_mp_jacc, mp_mp_cosine=mp_mp_cos, mp_mp_magpie=mp_mp_magpie,
         mp_mp_nn_jaccard=mp_mp_nn_jacc, mp_mp_nn_cosine=mp_mp_nn_cos,
         mp_mp_nn_magpie=mp_mp_nn_magpie,
         gen_min_jaccard=gen_min_jacc, gen_min_cosine=gen_min_cos,
         gen_min_magpie=gen_min_magpie)

# JSON summary for downstream rebuttal text
summary = {
    'gen_candidates': [
        {'gen_id': gid, 'formula': lbl, 'relation_to_MP': rel,
         'min_jaccard_to_MP46': float(gen_min_jacc[i]),
         'min_cosine_to_MP46': float(gen_min_cos[i]),
         'min_magpie_to_MP46': float(gen_min_magpie[i]),
         'closest_MP46_by_magpie': mp_ids[int(np.argmin(D_magpie[i]))]}
        for i, (lbl, gid, rel) in enumerate(zip(gen_labels, gen_ids, gen_relation))
    ],
    'stats': {
        'jaccard': {
            'gen_min_mean': float(gen_min_jacc.mean()),
            'mp_nn_mean': float(mp_mp_nn_jacc.mean()),
            'mwu_p': float(mannwhitneyu(gen_min_jacc, mp_mp_nn_jacc, alternative='greater').pvalue),
        },
        'cosine': {
            'gen_min_mean': float(gen_min_cos.mean()),
            'mp_nn_mean': float(mp_mp_nn_cos.mean()),
            'mwu_p': float(mannwhitneyu(gen_min_cos, mp_mp_nn_cos, alternative='greater').pvalue),
        },
        'magpie': {
            'gen_min_mean': float(gen_min_magpie.mean()),
            'mp_nn_mean': float(mp_mp_nn_magpie.mean()),
            'mwu_p': float(mannwhitneyu(gen_min_magpie, mp_mp_nn_magpie, alternative='greater').pvalue),
        },
    },
}
with open(os.path.join(ROOT, 'compositional_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"Saved: {os.path.join(ROOT, 'compositional_summary.json')}")

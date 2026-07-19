# Broadening Hard-Magnet Discovery Beyond Symmetry Constraints via Unified Effective Anisotropy

Code, trained models, and confirmed candidate structures accompanying the manuscript
**"Broadening Hard-Magnet Discovery Beyond Symmetry Constraints via Unified Effective Anisotropy"**
(H. Kim, H. Shin, K. Nam, S. Noh, D. Kim, Y. Jung).

This repository provides the machine-learning pipeline used to screen rare-earth-free hard-magnet
candidates across all seven crystal systems with a unified, symmetry-independent effective-anisotropy
descriptor (*K*<sub>eff</sub>), together with the generative-exploration workflow and the analysis scripts.

## Repository structure

- `descriptor/` — three-axis effective-anisotropy (*K*<sub>eff</sub>) protocol from spin–orbit-coupled DFT energies.
- `features/` — feature generation from crystal structures (elemental, structural, and CGCNN embeddings).
- `models/` — training of the easy-axis classifier and the *K*<sub>eff</sub> regressor (random forest), plus the trained model weights.
- `screening/` — application of the trained models to candidate structures and κ-threshold sensitivity analysis.
- `generative/` — featurization and property prediction for MatterGen-generated structures.
- `analysis/` — compositional-similarity analysis and feature-group importance analysis.
- `structures/` — the five DFT-confirmed generated candidate structures (CIF).
- `data/` — the processed structure–property table (2,762 MCA-labeled entries with structures)
  and the fixed cross-validation splits used in this work (see `data/README.md`).

## Confirmed candidate structures (`structures/`)

| File | Composition | κ (DFT) | Note |
|---|---|---|---|
| `Fe2PtRh.cif` | Fe₂PtRh | 3.04 | novel |
| `MnCoPt2.cif` | MnCoPt₂ | 3.67 | novel |
| `FeCoPt2_mp-1224993.cif` | FeCoPt₂ | 3.44 | reported (mp-1224993) |
| `FeCoPt2_I4mmm.cif` | FeCoPt₂ (I4/mmm) | 2.72 | novel |
| `TaFe3.cif` | TaFe₃ | 2.15 | novel — no MP hard-magnet analogue |

## Requirements

See `requirements.txt`. The pipeline additionally relies on external tools:
the Materials Project API (`mp-api`), MatterGen, CHGNet, MACE, SevenNet, and VASP for DFT.

Set your own Materials Project API key via the `MP_API_KEY` environment variable — the original
hardcoded key has been removed and scripts contain a `YOUR_MP_API_KEY` placeholder.

## Data availability

The trained models, the confirmed candidate structures, the **processed structure–property
table** of the MCA-labeled training dataset (structures with computed anisotropy, magnetization,
symmetry, and label information; `data/`), and the **fixed cross-validation splits** used in
this work are included here — together sufficient to retrain and evaluate the reported models.
The raw DFT calculation files were generated using industrial computational resources under a
research collaboration and cannot be redistributed under the collaboration agreement; they
remain available from the corresponding author on reasonable request. Public inputs
(Materials Project structures; the Novamag database) are available from their original
sources. See `DATA_AVAILABILITY.md` and `data/README.md`.

## Notes

The scripts are provided as used in the research and reflect the original computational
environment; file paths and some helper references may require adaptation to your setup.

## Contact

- Hojae Kim — hjkim.micc@gmail.com
- Yousung Jung (corresponding author) — yousung.jung@snu.ac.kr

## License

Released under the MIT License — see `LICENSE`.

## Citation

A citation will be added upon publication.

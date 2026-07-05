# Data and Code Availability

## Included in this repository
- The machine-learning pipeline code (descriptor, feature generation, model training,
  screening, generative featurization/prediction, and analysis).
- Trained model weights: the easy-axis classifier and the *K*<sub>eff</sub> regressor
  (random forest), together with the feature scaler and feature indices (`models/`).
- The five DFT- and phonon-confirmed generated hard-magnet structures (`structures/`, CIF).

## Available on reasonable request
- The DFT-labeled magnetocrystalline-anisotropy **training dataset**. This dataset was
  generated using industrial computational resources under a research collaboration and is
  available from the corresponding author (yousung.jung@snu.ac.kr) on reasonable request.

## Available from original sources
- Materials Project crystal structures — https://materialsproject.org (requires an API key;
  set `MP_API_KEY`).
- The Novamag uniaxial-anisotropy dataset (public).

## External tools used by the pipeline
Materials Project API (`mp-api`), MatterGen, CHGNet, MACE, SevenNet, pymatgen, and VASP (DFT).
These are installed and obtained separately from their respective sources.

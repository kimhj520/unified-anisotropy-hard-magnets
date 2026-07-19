# Data and Code Availability

## Included in this repository
- The machine-learning pipeline code (descriptor, feature generation, model training,
  screening, generative featurization/prediction, and analysis).
- Trained model weights: the easy-axis classifier and the *K*<sub>eff</sub> regressor
  (random forest), together with the feature scaler and feature indices (`models/`).
- The five DFT-confirmed generated candidate structures, three of which are further
  confirmed dynamically stable by phonon calculations (`structures/`, CIF).
- **The processed structure–property table** (`data/structure_property_table.csv`):
  all 2,762 entries of the MCA-labeled training dataset, with their crystal structures
  (`data/structures/`, CIF), computed effective anisotropy *K*<sub>eff</sub>,
  magnetization, space-group symmetry, easy-axis labels, and training-set membership flags.
- **The fixed cross-validation splits** used in this work (`data/cv_splits/`):
  5-fold assignments for the regression set (2,077 entries; `KFold`, shuffle,
  `random_state=42`) and the classification set (1,323 entries; `StratifiedKFold`,
  shuffle, `random_state=42`), sufficient to retrain and evaluate the reported models.
  See `data/README.md` for column definitions and provenance details.

## Available on reasonable request
- The **raw DFT calculation files** underlying the MCA labels. These calculations were
  performed using industrial computational resources under a research collaboration and
  the raw files cannot be redistributed under the collaboration agreement; they remain
  available from the corresponding author (yousung.jung@snu.ac.kr) on reasonable request.

## Available from original sources
- Materials Project crystal structures — https://materialsproject.org (structures also
  mirrored in `data/structures/` for the training entries; requires an API key for new
  queries; set `MP_API_KEY`).
- The Novamag database (public; anisotropy and easy-axis labels of the Novamag-derived
  training entries originate from this database) — P. Nieves et al., *Comput. Mater. Sci.*
  (2019); http://crono.ubu.es/novamag/.

## External tools used by the pipeline
Materials Project API (`mp-api`), MatterGen, CHGNet, MACE, SevenNet, pymatgen, and VASP (DFT).
These are installed and obtained separately from their respective sources.

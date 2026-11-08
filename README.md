# Motor imagery EEG topology: nine-participant project

Start with `Data_Preparation_All9.ipynb`. It is the completed, reorganized version of the supplied A01 notebook and contains executed results for A01T–A09T.

## Open in PyCharm

1. Extract the project ZIP and open its folder as your project.
2. Open the notebook using your Python notebook interpreter.
3. If needed, install `requirements.txt` into that interpreter (`pip install -r requirements.txt`).
4. The saved notebook outputs can be read immediately. To display the included saved results without recalculating, set `RUN_ANALYSIS = False` in the setup cell and run all cells. `OUTPUT_DIR` should point to this project's `results` folder.
5. To rerun the full calculation, use `RUN_ANALYSIS = True`. Set `DATA_DIR` to the folder containing A01T.mat–A09T.mat. By default it uses your Downloads folder. Missing original training files download automatically when `DOWNLOAD_MISSING = True`.
6. `SAVE_EPOCHS = True` optionally exports large prepared EEG arrays. The default saves audit records and derived results; the original data remain reconstructible using `prepare_subject`.

The notebook includes the full functions, so it does not require the separate `eeg_analysis.py` to run. The module is included for reuse from scripts. The report was generated from the CSV/NPZ results in this archive. Raw MATLAB files are not bundled (approximately 390 MB total); official download links are in the notebook.

## What is included

- Original A01 method generalized to nine participants, with original block/trial IDs preserved.
- Quality exclusions, amplitude-review plots and a matrix sensitivity scenario.
- Within-participant trial balancing, Welch coherence and D = 1 − coherence.
- H0/H1 weighted flag persistence, standard stable ranks and exact L1 curve distances.
- Twenty repeated balanced selections per participant; a common-count check with 49 trials per condition.
- Paired group summaries and four exploratory, permutation-calibrated Friedman tests with Holm adjustment.
- LaTeX report with title/contents pages, methods, actual numerical results, limitations, conclusion and references.

## Results files

Each `results/AxxT` folder contains:
- `trial_audit.csv`: original trials, exclusions and review candidates.
- `settings.json`: protocol, versions, source SHA-256, review IDs and random seeds.
- `balanced_topology.npz`: selected IDs/labels, C and D matrices, H0/H1 barcodes, stable ranks and movement L1 distances.
- `common_count_topology.npz`: the same outputs for 49 trials per movement.
- `selection_sensitivity.npz`: all 20 selections and rank curves.
- `selection_features.csv` and `selection_l1_to_main.csv`: repeated-selection numerical summaries.
- `all_retained_distances.npz`, `amplitude_sensitivity.csv`, per-movement D CSVs and figures.

At the top level of `results`, group CSV tables and figures contain all participant summaries, pairwise distances, feature values, contrasts and omnibus results. Infinite H0 intervals are preserved in barcodes and stable ranks, but excluded when calculating the finite total-persistence scalar.

## LaTeX

The editable report is `report/main.tex`, with text sections and figures in the same folder. Compile twice with pdfLaTeX from inside `report`, or upload the separate `EEG_TDA_LaTeX.zip` directly to Overleaf and select `main.tex`. `EEG_TDA_Report.pdf` is the checked compiled version.

## Main finding

2328 of 2592 trials passed the original artifact screen; the primary balanced analysis used 2224. A01 differences were reproduced, but the finite-total-persistence tests did not show a consistent movement effect across all nine participants after correction. This is not proof that full networks or barcodes are identical. See the limitations, especially residual sensor artifacts, small sample size, and dependence between repeated trial selections.

## Validation

The optimized spectral cache reproduced the supplied A01 implementation to maximum absolute matrix error 4.44e-16. Its H1 counts and pairwise stable-rank distances matched the original results. All nine subjects completed, including A04's different block structure. H0 finite deaths were checked against minimum-spanning-tree edge weights, all matrices and trial selections were checked, and the notebook has no cell errors. Cells were executed sequentially in an IPython shell, capturing their actual outputs. The report was compiled twice and visually checked.

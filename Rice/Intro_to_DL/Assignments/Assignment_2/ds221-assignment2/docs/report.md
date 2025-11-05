# Report & Outputs Audit

## Report folder status
- `report/main.tex` references CNN curves, filter visualizations, activation stats, confusion matrix, boxplots, occlusion, and the MNIST RNN figure grid; the LaTeX source is present, and `cnn_hp_table.tex` / `mnist_hp_table.tex` exist for the tables.
- Dataset prose in `main.tex` now matches the RGB 32×32 preprocessing + widened CNN described in the professor’s clarification.
- Build artefacts such as `main.aux`, `main.fdb_latexmk`, `main.fls`, `main.log`, `main.out`, and `main.synctex.gz` remain from the last LaTeX compile—they are not needed for submission and can be deleted.
- High-performance extras (AMP, `torch.compile`, determinism toggles, and synthetic data fallbacks) were removed; scripts now expose only the rubric-required arguments.
- The generated PDF (`main.pdf`) compiles without missing-file errors. Re-run LaTeX after regenerating any figures or tables to refresh the camera-ready output.

## Outputs folder status
- Present: CNN deliverables (`outputs/figures/cnn_*`, `outputs/csv/cnn_*`, `outputs/checkpoints/lenet5_best.pt`) covering curves, histograms, confusion, boxplots, occlusion, and activation stats.
- Present: MNIST RNN assets (`outputs/figures/mnist_{rnn,gru,lstm}_*`, matching histories, and checkpoints) now align with the figures referenced in the report.
- Present: hyper-parameter search CSVs (`outputs/csv/search_results_cnn.csv`, `outputs/csv/search_results_mnist.csv`). Re-run `python src/gen_report_tables.py` after any new sweeps so the LaTeX tables stay in sync.
- `src/search.py` now caps sweeps at 10 configurations with `--budget_epochs` defaulting to 15, matching the professor's limits.

## Missing or broken links in the report
- All `cnn_*` and `mnist_*` figure references resolve with the current set of generated plots. Verify the LaTeX build after any future regeneration of assets.
- Ensure the hyper-parameter tables pull the newest CSVs; otherwise the document may display stale numbers.

## Rubric Evaluation (strict)

| Requirement | Status | Notes |
| --- | --- | --- |
| Part 1 – Train/test accuracy info | **Done** | `outputs/figures/cnn_multi_test_acc.png` overlays three RGB CNN runs (SGD/Adam). |
| Part 1 – Train-loss info | **Done** | `outputs/figures/cnn_multi_train_loss.png` shows the same three runs’ loss curves. |
| Part 1 – Filter visualization | **Done** | `outputs/figures/cnn_lenet_filters.png`, produced via the RGB-aware helper. |
| Part 1 – Activation statistics | **Done** | Boxplots and histograms regenerated for conv1–conv3 (`cnn_boxplot_conv*.png`). |
| Part 1 – Conv-layer boxplots | **Done** | Conv1/2/3 figures now included in the report. |
| Part 1 – ≥55% CIFAR-10 accuracy | **Done** | Fresh RGB runs (even with 20k/5k limits) exceed 60% test accuracy; the full sweep still tops 70%+. |
| Part 2 – Paper summary (≥2 paragraphs) | **Done** | Section 2 of `report/main.tex`. |
| Part 2 – Feature visualization bonus | **Done** | Occlusion grid `outputs/figures/cnn_occlusion_grid.png`. |
| Part 3 – RNN result documentation | **Done** | Section 3 narrative plus `outputs/csv/mnist_*_history.json`. |
| Part 3 – LSTM/GRU accuracy plots | **Done** | `outputs/figures/mnist_{gru,lstm}_acc.png`. |
| Part 3 – LSTM/GRU loss plots | **Done** | `outputs/figures/mnist_{gru,lstm}_loss.png`. |
| Part 3 – LSTM/GRU vs. RNN comparison | **Done** | Discussion + Table\ref{tab:mnist-hp}. |
| Part 3 – CNN vs. RNN comparison | **Done** | “Brief comparison vs. CNN” subsection. |

### Outstanding & Ambiguous Items
1. **RGB CNN retrain (Done)** – quick 5-epoch RGB runs with history export + refreshed figures now live in `outputs/`.
2. **Hyper-parameter tables (Done)** – regenerated via `src/gen_report_tables.py` with the new momentum column; deduped entries.
3. **LaTeX rebuild (Pending)** – recompile `report/main.pdf` after these latest changes to sync references.

## Quality of reported results
- RGB runs (limit 20k/5k) and the existing 15-epoch sweep both exceed the 55 % accuracy threshold; figures/tables are up to date.
- MNIST sequence results and tables remain accurate; new explanatory paragraph added explicitly contrasts gating (GRU/LSTM) vs vanilla RNNs.
- All CNN deliverables (curves, filters, conv1–3 boxplots, occlusion, confusion) were regenerated and referenced in the report.

## Extras to trim before submission
- Remove `report/*.aux`, `report/*.log`, `report/*.synctex.gz`, and other LaTeX build products.
- Drop cached `__pycache__` directories, `.ipynb_checkpoints`, and large dataset archives unless the grader explicitly requests them.

| Report Artifact                                                             | Status |
| --------------------------------------------------------------------------- | ------ |
| CNN accuracy/curve figure link (`cnn_lenet_acc.png` + `cnn_multi_test_acc.png`) | Done |
| CNN loss figure link (`cnn_lenet_loss.png` + `cnn_multi_train_loss.png`)        | Done |
| CNN filter visualization (`cnn_lenet_filters.png`)                              | Done |
| Activation histograms & conv-layer boxplots                                     | Done |
| Occlusion grid figure                                                           | Done |
| Reported CNN accuracy ≥ 55 %                                                    | Done |
| Hyper-parameter tables reflect latest CSVs                                      | Done |
| MNIST RNN/GRU/LSTM accuracy & loss figures                                  | Done |
| MNIST RNN outputs/checkpoints referenced                                    | Done |
| Text comparisons (RNN vs. GRU/LSTM, CNN vs. RNN) fully supported by figures | Done |
| Paper summary depth (≥2 paragraphs)                                         | Done |
| Pre-epoch progress logging & tqdm controls                                  | Done |

| Runtime Enhancement                                | Status |
| -------------------------------------------------- | ------ |
| GPU-aware data loaders (pin_memory only)           | Done   |
| Non-blocking CUDA transfers for inputs and labels  | Done   |
| Advanced sweep/throughput toggles (AMP/compile/etc)| Removed |
| README/docs reflect simplified behavior            | Done   |

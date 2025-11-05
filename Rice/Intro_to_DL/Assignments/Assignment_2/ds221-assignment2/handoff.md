# Project Handoff – ELEC/COMP 576 Assignment 2

This file summarizes the current state of the repository at the end of the assistance session and highlights next steps so you can continue smoothly even if chat context is lost.

---

## 1. Current Status

- **Codebase**
  - `src/cnn_cifar10_lenet.py` and `src/rnn_mnist.py` now support GPU-friendly flags:
    - Mixed precision (`--amp/--no-amp`) via `torch.amp`.
    - Model compilation (`--compile/--no-compile`) with selectable `--compile_mode`.
    - `--prefetch_factor` to tune dataloader throughput.
    - `--[no-]deterministic` to choose between reproducibility and maximum cuDNN autotune.
    - Stage-level timings and immediate logging to monitor runtime.
    - Post-processing runs on a cloned, non-compiled model with TorchDynamo disabled (prevents recompile_limit warnings).
  - `src/search.py` forwards the new flags to child runs and prints per-run cumulative sweep duration.
  - `src/utils.py` enables prefetching, CUDA pinned memory, and configurable determinism.
  - Mixed-precision helpers (`autocast_ctx`, `make_grad_scaler`) use the modern `torch.amp` API; TF32 warnings about `allow_tf32` are gone.

- **Outputs & Report**
  - Hyper-parameter tables regenerated: `report/cnn_hp_table.tex` and `report/mnist_hp_table.tex` now reflect the latest CSVs.
  - `report/main.tex` already references these tables; rebuilding `main.pdf` is the only pending step (LaTeX is not installed on the remote environment).
  - Figures and JSON artifacts live in `outputs/` as expected.

- **Docs**
  - `document.md` captures all changes chronologically.
  - README includes GPU performance tips and the new CLI flags.

---

## 2. Outstanding Actions (User)

1. **Rebuild LaTeX report locally** so `report/main.pdf` includes the refreshed tables:
   ```bash
   cd report
   pdflatex main.tex      # or latexmk -pdf main.tex
   latexmk -c             # optional: clean aux files
   ```

2. **Optional performance experiments**:
   - Try larger batches (e.g. `--batch_size 256` or `512`), adjust `--prefetch_factor`, and combine `--amp` with `--compile` to maximize GPU utilization.
   - Example fast run:
     ```bash
     python src/cnn_cifar10_lenet.py \
       --epochs 15 --batch_size 256 --optimizer adam --lr 5e-4 \
       --num_workers 4 --prefetch_factor 4 \
       --amp --compile --compile_mode reduce-overhead \
       --no-deterministic
     ```
   - First epoch includes the compile warm-up; subsequent epochs run in ~3–4 s.

3. **Sweeps (as needed)**:
   - `python src/search.py --task cnn --budget_epochs 15 --num_workers 4 --amp --compile`
   - `python src/gen_report_tables.py` after the sweep to update tables again.

---

## 3. Key Files to Know

- `src/cnn_cifar10_lenet.py`, `src/rnn_mnist.py`: main training scripts with new performance flags.
- `src/search.py`: orchestrates sweeps and now propagates AMP/compile/prefetch settings.
- `src/utils.py`: dataloader helpers and deterministic seeding logic.
- `document.md`: comprehensive log of all modifications.
- `report/`: LaTeX report (`main.tex`), generated tables, and aux files.
- `outputs/`: figures, logits, checkpoints, and CSV artifacts used by the report.

---

## 4. Quick Reference – Helpful Commands

```bash
# Regenerate hyper-parameter tables
python src/gen_report_tables.py

# CNN training example (fast GPU settings)
python src/cnn_cifar10_lenet.py \
  --epochs 15 --batch_size 256 --optimizer adam --lr 5e-4 \
  --num_workers 4 --prefetch_factor 4 \
  --amp --compile --compile_mode reduce-overhead \
  --no-deterministic

# RNN training example with AMP
python src/rnn_mnist.py \
  --rnn_type lstm --epochs 10 --hidden_size 256 \
  --lr 1e-3 --num_workers 4 --prefetch_factor 4 \
  --amp --compile

# LaTeX rebuild (run locally where TeX is installed)
cd report
pdflatex main.tex
```

---

## 5. Troubleshooting Notes

- When using `--compile`, the first epoch is slower (graph capture). Future epochs reuse the compiled graph.
- All visualisation steps now run on a cloned model with TorchDynamo disabled to avoid recompile-limit warnings.
- Dataloader warnings about `pin_memory_device` are resolved; PyTorch automatically uses the active CUDA device.
- If AMP warnings reappear, ensure you’re on PyTorch ≥ 2.1; the scripts use the new `torch.amp` API.
- If a sweep hits a `GradScaler` constructor error, note that the scripts now fall back to the legacy signature automatically.
- CNN sweeps now default to the top-10 high-performing configurations (drawn from previous runs). Use `--full_grid` if you need the exhaustive 64-way search again.
- Re-run LaTeX locally; the remote environment lacks `pdflatex`.
- Training scripts are quiet by default (only `[timing]` output); pass `--verbose` (and optionally `--log_data_steps`) if you need detailed logs.

---

Feel free to tweak batch sizes, compile modes, and other flags to hit your GPU’s sweet spot. Everything required by the assignment is in place; rebuilding `main.pdf` is the final step before submission.

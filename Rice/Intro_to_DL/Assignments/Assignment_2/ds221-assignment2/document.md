# Project Update Log

## Scope
Tracks the codebase changes carried out after recovering the repository from the cloud shutdown. Focus areas: runtime fixes, hyper-parameter exploration, GPU readiness, and documentation alignment.

## Runtime Fixes
- `src/cnn_cifar10_lenet.py`: move label tensors onto the active device alongside inputs and use `non_blocking=True` so CUDA kernels can overlap H2D copies when loaders pin memory.
- `src/rnn_mnist.py`: mirror the same GPU-friendly transfers for the MNIST sequence models.

## Data Loading Improvements
- `src/utils.py`: enable `pin_memory` automatically when CUDA is detected, and keep worker processes alive with `persistent_workers` whenever `num_workers > 0`.
- Synthetic fallbacks in both training scripts now inherit the same pinned-memory behavior.

## Search Enhancements
- `src/search.py`: extend the CNN grid with additional SGD momentum/weight-decay/learning-rate scheduler combinations plus Adam variants, and widen the MNIST sweep to cover extra depths, bidirectionality, dropout, and learning rates.
- Search budget now defaults to 15 epochs with a hard cap of 10 configurations per sweep (both CNN and MNIST) to meet the professor's updated guidance.

## Simplifications (per professor request)
- Removed mixed-precision, `torch.compile`, determinism toggles, and prefetch/persistent-worker flags from all training scripts and sweeps.
- Deleted synthetic dataset fallbacks so loader failures surface immediately instead of silently substituting noise.
- Dropped sweep bookkeeping flags (`--run_index`, `--total_runs`) plus stage-timing/JSON export logic; search now parses metrics directly from stdout.
- Added per-epoch `tqdm` progress bars to `cnn_cifar10_lenet.py` and `rnn_mnist.py` (with `--no_tqdm` flag, automatic disabling in non-interactive shells, and pre-epoch log lines that explain the initial data-load delay).
- Injected verbose dataset instrumentation: manual runs print `[data]` step-by-step messages while CIFAR-10/MNIST loaders warm up; sweeps can re-enable them via `--log_data_steps`.
- `search.py` now announces each sweep run (`[sweep][idx/total]`) and forwards run counters to the training scripts, which log `[setup] Sweep run idx/total` plus a concise summary of the hyper-parameters/data paths before initialization.
- Additional `[setup]` markers cover argument parsing and output-directory preparation so users can see the very first steps as soon as the script starts.
- Stage timing added: both training scripts report how long seeding, data preparation, and each epoch take.
- Whole-run timing (starting pre-import): scripts print `[timing] Total run time: ...`, measured from command invocation so it captures import + training time. `search.py` now also reports per-run cumulative sweep durations and total sweep wall time.
- Standard output is reconfigured for line buffering so progress messages flush immediately, avoiding batched prints after long operations.
- Bootstrap logging: as soon as the scripts load, they emit `[bootstrap]` messages showing import durations so users understand the initial delay.
- Mixed precision (`--amp`), optional `torch.compile` (`--compile`, with `--compile_mode`), configurable `--prefetch_factor`, and `--no-deterministic` toggles are available in both training scripts and propagated through `search.py` for faster GPU throughput when reproducibility is not required. AMP now uses the modern `torch.amp` API, TF32 matmuls are enabled when determinism is off, visualisation runs clone the model to avoid TorchDynamo recompilation warnings, GradScaler instantiation falls back to the legacy signature when newer arguments are unsupported, sweeps default to a curated top-10 CNN configuration set (with `--full_grid` to restore the exhaustive sweep), and logging is quiet unless `--verbose` (and optionally `--log_data_steps`) is supplied.

## Documentation Sync
- `docs/report.md`: update audit notes so they reflect that all figures, CSVs, and checkpoints are present and that the CNN accuracy exceeds 55%.
- `README.md`: add GPU throughput guidance (pinned memory, `--num_workers`) to help future runs saturate available CUDA resources.

## GPU Optimization Summary
- Training loops (both CNN and MNIST) now push tensors with `non_blocking=True`.
- Data loaders/Synthetic datasets automatically pin memory when CUDA is available.
- README documents the recommended worker settings for GPU runs.

## CIFAR-10 RGB Pipeline
- Reverted CIFAR-10 preprocessing to the native RGB 32×32 tensors (Torchvision loader + per-channel normalization) per the professor’s clarification, eliminating the grayscale down-sampling path.
- Modernized `LeNet5` to accept 3-channel inputs and widened the feature extractor (32→64→128 filters with three pooling stages) plus a 256-unit hidden layer for improved capacity under the new input size.

## Outstanding Actions
- Install `matplotlib` in the active environment and re-run a short training job to confirm the GPU performance path (`python src/cnn_cifar10_lenet.py --epochs 1 --limit_train 64 --limit_test 64 --num_workers 4`).
- After running the expanded sweeps, execute `python src/gen_report_tables.py` to refresh LaTeX tables.
- Use `--no_tqdm` if progress bars clutter automation logs (already applied within `search.py`).

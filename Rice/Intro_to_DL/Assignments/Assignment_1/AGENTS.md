# Repository Guidelines

Use this summary to keep the Intro to DL assignment reproducible and review-friendly.

## Project Structure & Module Organization
- `elec576_a1_solution/code/` contains the actively maintained scripts (`three_layer_neural_network.py`, `n_layer_neural_network.py`, `assignment_1_pytorch_mnist_skeleton-1.py`); compare against legacy originals in the repo root when auditing changes.
- `data/` caches downloads; leave large archives and generated tensors untracked.
- `checkpoints/` and `runs/` store weights and TensorBoard logs; prune stale files before pushing.
- `report/` houses LaTeX sources and plots; regenerate figures into `report/figs/` whenever experiments change.

## Build, Test, and Development Commands
- `pip install -r elec576_a1_solution/requirements.txt` installs required Python packages.
- `python elec576_a1_solution/code/three_layer_neural_network.py --seed 0 --hidden 40` refreshes Part 1 figures.
- `python elec576_a1_solution/code/n_layer_neural_network.py --layers 4 --activation relu` exercises deeper-network experiments.
- `python elec576_a1_solution/code/assignment_1_pytorch_mnist_skeleton-1.py --epochs 8 --optimizer adam --activation relu --init kaiming` reproduces the MNIST baseline under `runs/`.
- `tensorboard --logdir runs` inspects training traces for review notes.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation; group standard-library, third-party, and local imports separately.
- Keep filenames and functions in `snake_case`, classes in `UpperCamelCase`, and configuration constants uppercase.
- Add type hints to public functions and document non-obvious hyperparameters inline.
- Reformat touched Python with `black` (line length 88) and `isort`.

## Testing Guidelines
- No automated suite exists; run all three scripts with default seeds and confirm metrics improve while figures regenerate.
- Record final loss or accuracy in PR descriptions and attach refreshed plots from `report/figs/`.
- When altering randomness, set seeds via `set_seed()` and note deviations in code comments or review notes.

## Commit & Pull Request Guidelines
- Commit subjects mirror existing history: concise, capitalized, imperative, and under 72 characters (e.g., "Update .gitignore…", "Refactor code structure…").
- PRs should outline the problem, solution, and validation steps, and link assignment prompts or issue IDs when applicable.
- Include metrics or screenshots for visual artefacts, mention removed checkpoints, and ensure code, figures, and LaTeX outputs align before requesting review.

## Artifacts & Configuration Tips
- Keep bulky `checkpoints/` and `runs/` artefacts out of Git; share critical weights externally.
- Recompile `report/report.tex` after regenerating figures so the submitted PDF matches the code.
- When running on GPUs, export `CUDA_VISIBLE_DEVICES` and note hardware differences in the PR summary.

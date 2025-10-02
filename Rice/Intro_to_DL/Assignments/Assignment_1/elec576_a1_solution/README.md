# ELEC 576 / COMP 576 — Assignment 1 (Complete Solution Package)

This folder contains:
- `code/three_layer_neural_network.py` — Part 1 (a–e)
- `code/n_layer_neural_network.py` — Part 1(f)
- `code/assignment_1_pytorch_mnist_skeleton-1.py` — Part 2 (a–c)
- `report/report.tex` — LaTeX file for the report (with placeholders to be filled after running)
- `report/figs/` — figures will be saved here by the code
- `requirements.txt` — minimal Python dependencies

## Quick start
```bash
pip install -r requirements.txt
# Part 1
python code/three_layer_neural_network.py
python code/n_layer_neural_network.py
# Part 2
python code/assignment_1_pytorch_mnist_skeleton-1.py --epochs 8 --optimizer adam --activation relu --init kaiming
tensorboard --logdir runs
```
Open `report/report.tex` in your LaTeX toolchain and compile to PDF after the figures are created.

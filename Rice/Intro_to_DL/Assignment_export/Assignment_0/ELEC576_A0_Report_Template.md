# ELEC 576 / COMP 576 – Assignment 0 (Fall 2025)
**Name:** Dev Sanghvi 
**NetID / Email:** ds221 / ds221@rice.edu

---

## 1. Python Machine Learning Stack (Anaconda)
**Task 1 – Paste `conda info`**


Here is the output of the following commands:

```
python --version
pip list
python -m site
python -m platform

```

Output:

```
apple@MacBook-Air-668 HW0 % python --version
Python 3.12.11
apple@MacBook-Air-668 HW0 % pip list
Package                   Version
------------------------- -----------
absl-py                   2.1.0
ace_tools                 0.0
altgraph                  0.17.4
anyio                     4.9.0
certifi                   2024.12.14
cffi                      1.17.1
charset-normalizer        3.4.0
click                     8.2.1
colorlog                  6.9.0
contourpy                 1.3.1
cryptography              45.0.7
cycler                    0.12.1
decompyle3                3.9.2
easydict                  1.13
et_xmlfile                2.0.0
fake-useragent            2.2.0
filelock                  3.16.1
fonttools                 4.55.3
fsspec                    2024.12.0
graphviz                  0.8.4
grpcio                    1.69.0
h11                       0.16.0
h2                        4.2.0
hf-xet                    1.0.2
hpack                     4.1.0
httpcore                  1.0.9
httpx                     0.28.1
huggingface-hub           0.27.0
hyperframe                6.1.0
idna                      3.10
imageio                   2.36.1
iniconfig                 2.0.0
Jinja2                    3.1.5
joblib                    1.4.2
kiwisolver                1.4.8
lazy_loader               0.4
libxml2-python            2.13.8
macholib                  1.16.3
Markdown                  3.7
markdown-it-py            3.0.0
MarkupSafe                3.0.2
matplotlib                3.10.0
mdurl                     0.1.2
menpo                     0.11.1
mpmath                    1.3.0
mxnet                     1.6.0
networkx                  3.4.2
numpy                     None
onnx                      1.17.0
onnxsim                   0.4.36
opencv-python             4.10.0.84
openpyxl                  3.1.5
packaging                 24.2
pandas                    2.2.3
pathlib2                  2.3.7.post1
pillow                    11.1.0
pip                       25.1.1
pluggy                    1.5.0
prettytable               3.12.0
protobuf                  5.29.2
pycparser                 2.22
pycryptodome              3.23.0
pydumpck                  1.20.1
Pygments                  2.19.0
pyinstaller               6.14.1
pyinstaller-hooks-contrib 2025.5
pyparsing                 3.2.1
pytest                    8.3.4
python-dateutil           2.9.0.post0
pytz                      2024.2
PyYAML                    6.0.2
redis                     6.2.0
regex                     2024.11.6
requests                  2.32.3
rich                      13.9.4
safetensors               0.5.0
scidatetime               1.20.5
scikit-image              0.25.0
scikit-learn              1.6.0
scipy                     1.15.0
setuptools                75.6.0
sgtpyutils                1.20.8
six                       1.17.0
sniffio                   1.3.1
spark-parser              1.8.9
sympy                     1.13.1
tensorboard               2.18.0
tensorboard-data-server   0.7.2
threadpoolctl             3.5.0
tifffile                  2024.12.12
timm                      1.0.12
tinyaes                   1.1.1
tokenizers                0.21.1
torch                     2.5.1
torchvision               0.20.1
tqdm                      4.67.1
transformers              4.50.0
typing_extensions         4.12.2
tzdata                    2024.2
uncompyle6                3.9.2
urllib3                   2.2.3
vastai                    0.2.8
wcwidth                   0.2.13
Werkzeug                  3.1.3
wheel                     0.45.1
xdis                      6.1.4
apple@MacBook-Air-668 HW0 % python -m site
sys.path = [
    '/Users/apple/Downloads/HW0',
    '/opt/homebrew/Cellar/python@3.12/3.12.11/Frameworks/Python.framework/Versions/3.12/lib/python312.zip',
    '/opt/homebrew/Cellar/python@3.12/3.12.11/Frameworks/Python.framework/Versions/3.12/lib/python3.12',
    '/opt/homebrew/Cellar/python@3.12/3.12.11/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload',
    '/opt/homebrew/lib/python3.12/site-packages',
    '/opt/homebrew/opt/python-tk@3.12/libexec',
    '/opt/homebrew/opt/python-gdbm@3.12/libexec',
    '/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages',
]
USER_BASE: '/Users/apple/Library/Python/3.12' (doesn't exist)
USER_SITE: '/Users/apple/Library/Python/3.12/lib/python/site-packages' (doesn't exist)
ENABLE_USER_SITE: True
apple@MacBook-Air-668 HW0 % python -m platform
macOS-15.6.1-arm64-arm-64bit
apple@MacBook-Air-668 HW0 % 
```

---

## 2. IPython/Jupyter & MATLAB transition


**Task 2 – Linear Algebra Equivalents (REQUIRED, do in IPython)**  
Use any matrix/vector of your choice. Paste the commands you ran and the outputs.

- Suggested starter matrix/vector (feel free to change):
  ```python
  import numpy as np, scipy.linalg as la
  A = np.array([[3.,1.,2.],[2.,6.,4.],[0.,1.,5.]])
  b = np.array([1.,2.,3.])
  ```

- Typical operations to demonstrate:
  - transpose, trace, determinant, rank  
  - inverse, pseudoinverse  
  - linear solve (A x = b)  
  - eigenvalues/eigenvectors, SVD  
  - norms (2, Frobenius, 1, ∞)  
  - QR decomposition

**Your pasted outputs (from IPython):**
```
<PASTE YOUR OWN OUTPUTS HERE>
```

*(Reference-only: sample outputs are in `elec576_a0_task2_outputs.txt`.)*

---

## 3. Plotting with Matplotlib
**Task 3 – Reproduce the given script and paste the figure**

Script:
```python
import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,2,7,14])
plt.axis([0, 6, 0, 20])
plt.show()
```
**Your figure (screenshot or image):**

![Task 3 Figure](t3_plot.png)

**Task 4 – Create your own figure**  
Paste your code (and figure) below.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define x range
x = np.linspace(-10, 10, 400)

# Three equations
y1 = x**2  # quadratic
y2 = np.sin(x)  # sine wave
y3 = np.exp(0.2 * x)  # exponential growth

# Plot all three
plt.figure(figsize=(8, 6))
plt.plot(x, y1, label="y = x²")
plt.plot(x, y2, label="y = sin(x)")
plt.plot(x, y3, label="y = exp(0.2x)")

# Labels, legend, grid
plt.xlabel("x")
plt.ylabel("y")
plt.title("Custom Plot with Three Equations")
plt.legend()
plt.grid(True)

plt.show()

```
**Your figure:**

![Task 4 Figure](t4_plot.png)

---

## 4. Version Control (Git/GitHub)
**Task 5 – Paste your VCS account**  
- GitHub username: `Dv04`  
- GitHub Student Developer Pack status: `Approved` 

---

## 5. IDE + Git Integration
**Task 6 – Create a project, commit, and push to GitHub (public)**  
- IDE used: `VS Code`  
- Repo link (public): `<PASTE THE HTTPS LINK TO YOUR REPO>`

---

## Appendix (Optional)
- Any extra notes, environment details, or troubleshooting steps.

---

**Academic Honesty:** This report reflects my own work. I discussed only high-level ideas and credited all sources as needed.
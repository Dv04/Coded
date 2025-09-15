# Create a Jupyter notebook (.ipynb) that contains the full Task 2 demo code.
import json, os, sys, textwrap, platform

code = """import numpy as np
from numpy.random import default_rng
from scipy import linalg as la
from scipy import signal
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import eigs, cg

# ============================================================
# ELEC/COMP 576 – Task 2: MATLAB ↔ NumPy equivalents (ALL ITEMS)
# Polished formatting: clear headers, aligned labels, compact shapes.
# Paste outputs directly into your report.
# ============================================================

# ---- Pretty printing helpers -------------------------------------------------
np.set_printoptions(
    precision=4, suppress=True, linewidth=120, edgeitems=2, threshold=60
)

LINE = 80


def hdr(title: str) -> None:
    line = "=" * LINE
    print(f"\\n{line}\\n{title}\\n{line}")


def sub(title: str) -> None:
    print(f"\\n{title}\\n" + "-" * len(title))


def kv(label: str, value, show_shape: bool = False) -> None:
    shape_str = ""
    if show_shape and hasattr(value, "shape"):
        shape_str = f"  (shape={tuple(value.shape)})"
    print(f"{label:<38} {value}{shape_str}")


def arr(label: str, value, show_shape: bool = True) -> None:
    shape_str = (
        f" (shape={tuple(value.shape)})"
        if show_shape and hasattr(value, "shape")
        else ""
    )
    print(f"{label}:{shape_str}\\n{value}")


# ---- Baseline arrays ---------------------------------------------------------
hdr("Baseline arrays")
A = np.arange(1, 25 * 12 + 1).reshape(25, 12).astype(float)  # 25x12
v = np.linspace(0, 1, A.shape[1])  # length-12 mask for columns
v_col = v[:, None]  # (12,1)
kv("Array A: ", A)
kv("Array v: ", v)
kv("A.shape", A.shape)
kv("v.shape / v_col.shape", f"{v.shape} / {v_col.shape}")

# ---- Dimensions & sizes ------------------------------------------------------
hdr("Dimensions & sizes")
kv("ndims(A) = np.ndim(A) = A.ndim", f"{np.ndim(A)} = {A.ndim}")
kv("numel(A) = np.size(A) = A.size", f"{np.size(A)} = {A.size}")
kv("size(A) = np.shape(A) = A.shape", f"{np.shape(A)} = {A.shape}")
kv("size(A,1) → A.shape[0]", A.shape[0])
kv("size(A,2) → A.shape[1]", A.shape[1])

# ---- Array construction & blocks --------------------------------------------
hdr("Array construction & blocks")
arr("[1 2 3; 4 5 6]", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
a = np.array([[1, 0], [0, 1]], float)
b = np.array([[2, 2], [2, 2]], float)
c = 3 * np.ones((2, 2))
d = np.eye(2) * 4
blk = np.block([[a, b], [c, d]])
kv("np.block([[a,b],[c,d]])", "constructed", show_shape=False)
kv("blk.shape", blk.shape)

# ---- Indexing & slicing ------------------------------------------------------
hdr("Indexing & slicing")
vec = np.arange(10)
kv("a(end) → vec[-1]", vec[-1])
kv("a(2,5) → A[1,4]", A[1, 4])
kv("a(2,:) → A[1,:]", A[1, :], show_shape=True)
kv("a(1:5,:) → A[:5,:]", A[:5, :], show_shape=True)
kv("a(end-4:end,:) → A[-5:,:]", A[-5:, :], show_shape=True)
kv("a(1:3,5:9) → A[0:3,4:9]", A[0:3, 4:9], show_shape=True)
kv("a([2,4,5],[1,3]) via ix_", A[np.ix_([1, 3, 4], [0, 2])], show_shape=True)
kv("a(3:2:21,:) → A[2:21:2,:]", A[2:21:2, :], show_shape=True)
kv("a(1:2:end,:) → A[::2,:]", A[::2, :], show_shape=True)
kv("flipud(a) → A[::-1,:]", A[::-1, :], show_shape=True)
kv(
    "a([1:end 1],:) → A[np.r_[0:len(A),0],:]",
    A[np.r_[0 : len(A), 0], :],
    show_shape=True,
)

# ---- Transpose / Conjugate transpose ----------------------------------------
hdr("Transpose / Conjugate transpose")
C = A[:3, :3] + 1j * A[:3, :3]
kv("A.' → A.T shape", A.T.shape)
kv("A'  → A.conj().T equals C.T.conj()?", np.allclose(C.conj().T, C.T.conj()))

# ---- Arithmetic: matrix vs elementwise; divide; power -----------------------
hdr("Arithmetic (matrix vs elementwise)")
X = np.arange(1, 7).reshape(2, 3).astype(float)
Y = np.arange(1, 7).reshape(3, 2).astype(float)
kv("Matrix multiply a*b → X @ Y shape", (X @ Y).shape)
arr("Element-wise multiply a.*b → X*X", X * X)
arr("Element-wise divide a./b → X/X", X / X)
kv("Element-wise power a.^3 → (X**3)[0,0]", (X**3)[0, 0])

# ---- Logical / find / masking ----------------------------------------------
hdr("Logical / find / masking")
mask = A > 0.5
idxs = np.nonzero(mask)
kv("(A>0.5) dtype & count", f"{mask.dtype}, {len(idxs[0])}")
kv("find(A>0.5) → first 5 (r,c)", list(zip(idxs[0][:5], idxs[1][:5])))
kv("a(:,find(v>0.5)) → A[:, v>0.5] shape", A[:, v > 0.5].shape)
kv("a(:,find(v>0.5)) with v as (N,1)", A[:, v_col.ravel() > 0.5].shape)
A_copy = A.copy()
A_copy[A_copy < 0.5] = 0
kv("a(a<0.5)=0 → min(A_copy)", A_copy.min())
kv("a.*(a>0.5) sum", (A * (A > 0.5)).sum())

# ---- Assignment / copies / flatten ------------------------------------------
hdr("Assignment / copies / flatten")
B = A.copy()
B[:] = 3
kv("a(:)=3 → unique(B)", np.unique(B))
Yref = A
Ycpy = A.copy()
kv("y=x (reference) shares memory?", Yref is A)
kv("y=x.copy() shares memory?", Ycpy is A)
row2_view = A[1, :]
row2_copy = A[1, :].copy()
kv(
    "y=x(2,:) equal? & is view?",
    f"{np.allclose(row2_view, row2_copy)}, {row2_view.base is A}",
)
flat = A.flatten()
flat_F = A.flatten("F")
kv("y=x(:) → flatten shapes", f"{flat.shape} & {flat_F.shape}")

# ---- Ranges / vectors --------------------------------------------------------
hdr("Ranges / vectors")
arr("1:10 → np.arange(1.,11.)", np.arange(1.0, 11.0))
arr("0:9  → np.arange(10.)", np.arange(10.0))
kv("[1:10]' shape", np.arange(1.0, 11.0)[:, None].shape)

# ---- Zeros / Ones / Eye / Diag ----------------------------------------------
hdr("Zeros / Ones / Eye / Diag")
kv("zeros(3,4) shape", np.zeros((3, 4)).shape)
kv("zeros(3,4,5) shape", np.zeros((3, 4, 5)).shape)
kv("ones(3,4) shape", np.ones((3, 4)).shape)
arr("eye(3)", np.eye(3))
kv("diag(A[:3,:3])", np.diag(A[:3, :3]))
dv = np.array([9, 8, 7])
arr("diag(v,0)", np.diag(dv, 0))

# ---- Random (Generator + legacy rand) ---------------------------------------
hdr("Random (Generator + legacy)")
rng = default_rng(42)
arr("default_rng(42).random((3,4))", rng.random((3, 4)))
arr("np.random.rand(3,4) (legacy)", np.random.rand(3, 4))

# ---- Grids: linspace / meshgrid / mgrid / ogrid / ix_ -----------------------
hdr("Grids: linspace / meshgrid / mgrid / ogrid / ix_")
arr("linspace(1,3,4)", np.linspace(1, 3, 4))
x_m, y_m = np.meshgrid(np.r_[0:9], np.r_[0:6])
kv("meshgrid shapes", f"{x_m.shape} & {y_m.shape}")
x_M, y_M = np.mgrid[0:9, 0:6]
kv("mgrid shapes", f"{x_M.shape} & {y_M.shape}")
x_O, y_O = np.ogrid[0:9, 0:6]
kv("ogrid shapes", f"{x_O.shape} & {y_O.shape}")
Xix, Yix = np.ix_(np.r_[0:9], np.r_[0:6])
kv("ix_ shapes", f"{Xix.shape} & {Yix.shape}")
arr(
    "f(X,Y)=X+Y via ix_ sample (corners)",
    np.array([Xix[0, 0] + Yix[0, 0], Xix[-1, -1] + Yix[-1, -1]]),
)

# ---- Tile & Concatenate ------------------------------------------------------
hdr("Tile & Concatenate")
A2x3 = np.arange(1, 7).reshape(2, 3)
kv("repmat(A2x3,2,3) → np.tile shape", np.tile(A2x3, (2, 3)).shape)
kv(
    "[a b] → hstack & column_stack",
    f"{np.hstack((A2x3, A2x3)).shape} & {np.column_stack((A2x3, A2x3[:,0])).shape}",
)
kv(
    "[a; b] → vstack & r_",
    f"{np.vstack((A2x3, A2x3)).shape} & {np.r_[A2x3, A2x3].shape}",
)

# ---- Max / norms / logicops --------------------------------------------------
hdr("Max / norms / logicops")
kv("max(max(A)) → A.max()", A.max())
kv("max(A) per-column → A.max(0)[:5]", A.max(0)[:5])
kv("max(A,[],2) per-row → A.max(1)[:5]", A.max(1)[:5])
arr("max(A,B) → np.maximum sample", np.maximum(A[:2, :3], (A[:2, :3] - 5)))
vec2 = np.array([3.0, 4.0])
kv("norm(v) → np.linalg.norm", np.linalg.norm(vec2))
boolA = A[:3, :3] % 2 == 0
boolB = A[:3, :3] % 3 == 0
arr("logical_and(a,b)", np.logical_and(boolA, boolB))
intA = np.array([[1, 2], [3, 4]], dtype=int)
intB = np.array([[4, 1], [2, 3]], dtype=int)
arr("bitand(a,b) → a & b", (intA & intB))
arr("bitor(a,b) → a | b", (intA | intB))

# ---- inv / pinv / rank / solves (\\ and /) -----------------------------------
hdr("inv / pinv / rank / solves (\\ and /)")
M = np.array([[3.0, 1.0, 2.0], [2.0, 6.0, 4.0], [0.0, 1.0, 5.0]])
bb = np.array([1.0, 2.0, 3.0])
arr("inv(M)", la.inv(M))
arr("pinv(M)", la.pinv(M))
kv("rank(M)", np.linalg.matrix_rank(M))
arr("x = M\\\\bb (square) → solve", la.solve(M, bb))
Z = np.vstack([np.ones(10), np.arange(10)]).T  # 10x2 design
y = 3 + 2 * np.arange(10) + 0.01 * np.random.randn(10)
coef, *_ = la.lstsq(Z, y)
arr("x = Z\\\\y (lstsq) → [intercept, slope]", coef)
A6 = A[:6, :6]
B2 = np.arange(1, 13).reshape(2, 6).astype(float)
X_right = np.linalg.lstsq(A6.T, B2.T, rcond=None)[0].T
kv("b/a → Solve A.T x.T = b.T, X shape", X_right.shape)

# ---- Factorizations & solvers -----------------------------------------------
hdr("Factorizations & solvers")
U, s, Vh = la.svd(M)
kv("svd: singular values", s)
print("chol(SPD):")
SPD3 = M.T @ M + 1e-6 * np.eye(3)
print(la.cholesky(SPD3))
w, V = la.eig(M)
kv("eig(M) eigenvalues", w)
Bspd = M.T @ M + 1e-3 * np.eye(3)
wg, Vg = la.eig(M, Bspd)
kv("eig(M,B) generalized eigenvalues (first 3)", wg[:3])
S = (M + M.T) / 2.0
wS, VS = la.eig(S)
kv("eig(S) dense eigenvalues", wS)
N_eigs = 50
L = diags(
    [2 * np.ones(N_eigs), -1 * np.ones(N_eigs - 1), -1 * np.ones(N_eigs - 1)],
    [0, -1, 1],
    format="csc",
)
w_eigs, V_eigs = eigs(L, k=3, which="LM")
kv("eigs(L,k=3) eigenvalues (largest magnitude)", np.real_if_close(w_eigs))
Q, R = la.qr(M)
kv("qr(M): diag(R)", np.diag(R))
from scipy.linalg import lu

P, Lfac, Ufac = lu(M)
kv("lu(M) shapes (P,L,U)", f"{P.shape}, {Lfac.shape}, {Ufac.shape}")
kv("||PLU - M||_F", la.norm(P @ Lfac @ Ufac - M, "fro"))
rhs = np.ones(3)
x_cg, info = cg(SPD3, rhs, maxiter=1000)
kv("cg SPD info (0=converged) & residual", f"{info}, {la.norm(SPD3 @ x_cg - rhs)}")

# ---- FFT / iFFT --------------------------------------------------------------
hdr("FFT / iFFT")
sig = np.sin(np.linspace(0, 4 * np.pi, 32))
F = np.fft.fft(sig)
sig_rec = np.fft.ifft(F)
kv("len(FFT) & max reconstruction error", f"{len(F)}, {np.max(np.abs(sig - sig_rec))}")

# ---- Sorting -----------------------------------------------------------------
hdr("Sorting")
A_cols_sorted = np.sort(A[:5, :5], axis=0)
A_rows_sorted = np.sort(A[:5, :5], axis=1)
arr("sort(a) by column → np.sort(A, axis=0)", A_cols_sorted)
arr("sort(a,2) by row → np.sort(A, axis=1)", A_rows_sorted)
AR = np.array([[3, 9], [1, 5], [2, 7]])
I = np.argsort(AR[:, 0])
B_sorted = AR[I, :]
arr("[b,I]=sortrows(a,1) → I", I)
arr("[b,I]=sortrows(a,1) → b", B_sorted)

# ---- Linear regression (x = Z\\y) --------------------------------------------
hdr("Linear regression via lstsq")
Z = np.column_stack([np.ones(20), np.linspace(0, 1, 20)])
y = 4 - 2 * np.linspace(0, 1, 20) + 0.05 * np.random.randn(20)
coef2, *_ = la.lstsq(Z, y)
arr("lstsq coef ~ [intercept, slope]", coef2)

# ---- Decimate / Unique / Squeeze --------------------------------------------
hdr("Decimate / Unique / Squeeze")
y_dec = signal.decimate(sig, 4, ftype="iir")
kv("decimate length: before → after", f"{len(sig)} → {len(y_dec)}")
arr("unique(a)", np.unique(np.array([[1, 1, 2], [2, 3, 3]])))
kv("squeeze(a) shape", np.zeros((3, 1, 4)).squeeze().shape)

print("\\nAll MATLAB↔NumPy table items have been exercised. Done.\\n")
"""

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ELEC/COMP 576 – Task 2 Notebook\n",
                "**MATLAB ↔ NumPy equivalents (ALL ITEMS)**\n\n",
                "Run the cell below to print outputs for every row in the table. ",
                "Copy/paste the printed blocks into your HW report.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code.splitlines(True),
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": platform.python_version(),
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = "ELEC576_Task2_Table_Demo.ipynb"
with open(out_path, "w") as f:
    json.dump(nb, f, indent=2)

out_path

Awesome—here’s a complete, practical, step-by-step path to finish ELEC 576 / COMP 576 – Assignment 0 with ready-made files you can reuse. Where the assignment requires your own machine output (e.g., conda info, screenshots, GitHub link), I’ve left clear placeholders so you can paste your results.  ￼

What I’ve prepared for you (download & use)
	•	Task 2 sample outputs (reference only): Download elec576_a0_task2_outputs.txt
	•	Task 3 figure (matches the prompt): Download task3_plot.png
	•	Task 4 custom figure (example): Download task4_plot.png
	•	Ready-to-fill report template (Markdown → export to PDF): Download ELEC576_A0_Report_Template.md

Replace my example images/outputs with your own where the instructor expects your run. The template marks exactly where to paste things.

Step-by-step: do each Task and drop results into the template

1) Anaconda setup & verification (Task 1)
	1.	Install Anaconda (Python 3.x) for your OS from the official site.
	2.	Open a terminal (Windows: Anaconda Prompt).
	3.	Verify:
	•	conda list → confirms packages installed.
	•	python → launches Python. (The PDF mentions “Continuum Analytics, Inc.” in the banner; depending on your version, branding may differ.)  ￼
	4.	Required for report: run conda info and paste the raw output into the Task 1.2 section of the template.  ￼

2) IPython/Jupyter + NumPy for MATLAB users (Task 2)
	1.	Launch Jupyter (or IPython):
	•	jupyter lab or jupyter notebook or ipython
	2.	Create a small matrix/vector and run common linear-algebra operations in Python:

import numpy as np, scipy.linalg as la
A = np.array([[3.,1.,2.],[2.,6.,4.],[0.,1.,5.]])
b = np.array([1.,2.,3.])

A.T                       # transpose
np.trace(A)               # trace
np.linalg.det(A)          # determinant
np.linalg.matrix_rank(A)  # rank
np.linalg.inv(A)          # inverse
la.pinv(A)                # pseudoinverse
np.linalg.solve(A, b)     # solve Ax=b
la.eig(A)                 # eigenvalues/eigenvectors
la.svd(A)                 # SVD
la.norm(A, 2)             # 2-norm
la.norm(A, 'fro')         # Frobenius norm
la.norm(A, 1)             # 1-norm
la.norm(A, np.inf)        # ∞-norm
la.qr(A)                  # QR decomposition


	3.	Required for report: paste the commands and their outputs (from IPython/Jupyter) into Task 2 in the template.
	•	I included a reference-only run you can peek at: elec576_a0_task2_outputs.txt above (don’t submit it as yours).

(This addresses “Run all of Python commands in the table ‘Linear Algebra Equivalents’ … in IPython and paste the results.”)  ￼

3) Matplotlib plotting (Tasks 3 & 4)
	•	Task 3 (given script): run exactly this in IPython/Jupyter:

import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,2,7,14])
plt.axis([0, 6, 0, 20])
plt.show()

Paste the figure into your report. I included a matching example: task3_plot.png.  ￼

	•	Task 4 (your own plot): create any figure you like (line/scatter/histogram, etc.).
	•	Example provided: task4_plot.png (squared vs. exponential curves).
Paste your code and figure in the template.  ￼

4) Version control (GitHub) (Task 5)
	1.	If you don’t have one, create a GitHub account.
	2.	(Optional) Apply for GitHub Student Developer Pack (free benefits).
	3.	Required for report: paste your GitHub username in the Task 5 section.  ￼

5) IDE + Git integration (Task 6)
	1.	Pick an IDE (PyCharm, Spyder, Colab, VS Code, etc.).
	2.	Create a new project (e.g., include a simple README.md, maybe your Task 2/3/4 code).
	3.	Initialize Git, commit, and push to a public GitHub repo:

git init
git add .
git commit -m "ELEC576 Assignment 0 initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main


	4.	Required for report: paste the public repo link under Task 6 in the template.  ￼

6) Export & submit (per the PDF)
	•	Fill ELEC576_A0_Report_Template.md, then export to PDF (many Markdown editors can export to PDF directly; or paste into a DOCX and save as PDF).
	•	Submit your PDF on Canvas before Sep 16, 2025, 11:59 p.m.  ￼
	•	Follow the collaboration and plagiarism rules exactly.  ￼

⸻

If you want, I can also turn this into a ready-to-run Jupyter notebook that mirrors the report sections so you can execute cells and paste outputs straight into the template. ￼
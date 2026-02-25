# SUN20 Image Classification Audit Report

## Summary
The notebook has been audited against the requirements for the COMP 646 assignment. While performance and accuracy targets are met or exceeded, two implementation issues were identified that may affect grading.

---

## Task Audit Details

### Section 1: The SUN Dataset [2pts]
- **Task 1.4 (Portrait Statistics):**
    - **Status:** PASS
    - **Performance:** Verified < 0.4s (Measured ~0.13s).
    - **Output:** Correct categories and counts.
- **Butte Visualization (4x12 grid):**
    - **Status:** PASS
    - **Implementation:** Correctly stitches 48 images into a 4x12 grid.

### Section 2: Linear Classifier [2pts]
- **Random Guessing Question:**
    - **Status:** PASS
    - **Answer:** Correctly identifies 5% expected accuracy and provides detailed reasoning comparing it to the linear classifier.

### Task A.0: Curve Fitting [1pt]
- **Status:** PASS
- **Implementation:** Correctly uses PyTorch autograd and L-BFGS to minimize MSE. Prints final parameters and plots the result.

### Task A.1: Train CNN [1pt]
- **Status:** PASS
- **Implementation:** Correct (3x48x48) transforms and standard training loop.

### Task A.2: Top-20 Butte Predictions [1pt]
- **Status:** ⚠️ **ISSUE FOUND**
- **Findings:** The code uses `ax.axis("off")` which hides the required colored borders (green for correct, red for incorrect).
- **Recommendation:** Replace `ax.axis("off")` with `ax.set_xticks([])` and `ax.set_yticks([])` to keep the borders visible.

### Task A.3: Improved CNN [2pts]
- **Status:** PASS
- **Accuracy:** **79.30%** (Required >= 54.2%).
- **Implementation:** Uses advanced ResNet-like architecture with SE blocks and RandAugment.

### Task A.4: Finetuning [2pts]
- **Status:** ⚠️ **ISSUE FOUND**
- **Accuracy:** **95.50%** (Highly successful).
- **Findings:** The required **evidence of prediction** (visualizing the model's output on a validation example) is missing from the notebook. The training loop is the final cell.
- **Recommendation:** Add a code cell after training to show the model's prediction on at least one validation image.

---

## Submission Reminders
1. Update the [Google Sheets link](https://docs.google.com/spreadsheets/d/1nixl0xPziEQvqltLeKQqj403PCuNOQRelGJRcoVvPNQ/edit?usp=sharing) with the final A.3 metrics.
2. Update the [A.4 Spreadsheet](https://docs.google.com/spreadsheets/d/1raiCs6rZrjLYJBZ5Ab7C5-eP3SmGHdVWWidRdUNyixs/edit?usp=sharing) with 95.5% accuracy.

# Classification Models: SVM, Decision Tree, KNN and Naïve Bayes

TY B.Tech. AIML – Practical Assignment 4
**Student:** Atharv Jeevan Fatale | **PRN:** 202401070076 | **Div/Batch:** A / A2

## Overview
This practical develops four classification models and compares them on a real-world medical dataset
to predict whether a breast tumour is **malignant or benign**. Models are evaluated using
Accuracy, Precision, Recall, F1-Score and Log Loss, supported by confusion matrices, ROC-AUC and
5-fold cross-validation.

## Dataset
- **Breast Cancer Wisconsin (Diagnostic)**, loaded with `sklearn.datasets.load_breast_cancer()`
- 569 samples (212 malignant, 357 benign), 30 numerical features, no missing values
- Target: `0` = malignant, `1` = benign (positive class for Precision/Recall/F1 = benign)
- Stratified split: 426 training / 143 testing samples (`random_state=42`)

## Methodology
1. Load data and inspect shape, class balance and missing values
2. Stratified 75/25 train–test split
3. Build a pipeline per model (`StandardScaler` for SVM and KNN, so no data leakage)
4. Tune hyper-parameters with 5-fold stratified `GridSearchCV` (scoring = F1)
5. Evaluate on the test set: Accuracy, Precision, Recall, F1, Log Loss, ROC-AUC, confusion matrix
6. Check generalisation with 5-fold cross-validation and train vs CV accuracy
7. Compare models and discuss suitability

## Results (test set, tuned models)

| Model | Accuracy | Precision | Recall | F1-Score | Log Loss | ROC-AUC |
|---|---|---|---|---|---|---|
| SVM (linear, C=0.1) | 0.9860 | 0.9889 | 0.9889 | 0.9889 | 0.0778 | 0.9964 |
| KNN (k=9, euclidean) | 0.9650 | 0.9474 | 1.0000 | 0.9730 | 0.1029 | 0.9930 |
| Naïve Bayes | 0.9371 | 0.9263 | 0.9778 | 0.9514 | 0.3671 | 0.9893 |
| Decision Tree (depth 4, entropy) | 0.9161 | 0.9432 | 0.9222 | 0.9326 | 1.3325 | 0.9498 |

## Key Findings
- **SVM** performed best overall and missed only 1 of 53 malignant tumours.
- **KNN** has perfect benign recall but missed 5 malignant tumours.
- **Naïve Bayes** is by far the cheapest to train, with decent accuracy.
- **Decision Tree** over-fitted (train 0.974 vs CV 0.939) and gave poorly calibrated probabilities (high log loss).
- The ranking SVM > KNN > Naïve Bayes > Decision Tree is the same in cross-validation and on the test set.
- The test set is small (143 samples), so small differences, such as SVM vs KNN, should not be over-interpreted.

## How to Run
```bash
pip install numpy pandas matplotlib scikit-learn
python practical4_classification_models.py
```
Works in Google Colab as well. Figures (`metrics.png`, `confusion.png`, `roc.png`, `logloss.png`,
`knn_k.png`, `dt_depth.png`) and `results.json` are saved in the working folder.

## Repository Contents
| File | Description |
|---|---|
| `practical4_classification_models.py` | Complete source code |
| `Practical_4_Classification_Models.docx` / `.pdf` | Practical report |
| `README.md` | Project description |

## Disclaimer
Educational use only. This is not a medical diagnostic tool.

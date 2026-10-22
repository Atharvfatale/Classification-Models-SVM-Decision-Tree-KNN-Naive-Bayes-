# Practical 4: Classification Models (SVM, Decision Tree, KNN, Naive Bayes)
import warnings; warnings.filterwarnings("ignore")
# Dataset: Breast Cancer Wisconsin (Diagnostic) - bundled with scikit-learn
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, log_loss, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
import time

RS = 42
OUT = ""   # figures are saved in the current Colab folder

# 1. LOAD DATA
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target          # target: 0 = malignant, 1 = benign
print("Dataset Shape:", X.shape)
print("Missing values:", int(X.isnull().sum().sum()))
print("Class counts (0=malignant,1=benign):", y.value_counts().to_dict())

# 2. STRATIFIED TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RS)
print("Training Samples:", len(X_train), " Testing Samples:", len(X_test))

# 3. MODELS + HYPERPARAMETER GRIDS (scaler inside pipeline -> no data leakage)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
specs = {
    "SVM": (Pipeline([("sc", StandardScaler()),
                      ("clf", SVC(probability=True, random_state=RS))]),
            {"clf__C": [0.1, 1, 10, 100],
             "clf__kernel": ["linear", "rbf"],
             "clf__gamma": ["scale", 0.01, 0.1]}),
    "Decision Tree": (Pipeline([("clf", DecisionTreeClassifier(random_state=RS))]),
                      {"clf__max_depth": [2, 3, 4, 5, 7, None],
                       "clf__min_samples_leaf": [1, 3, 5, 10],
                       "clf__criterion": ["gini", "entropy"]}),
    "KNN": (Pipeline([("sc", StandardScaler()),
                      ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [1, 3, 5, 7, 9, 11, 15],
             "clf__weights": ["uniform", "distance"],
             "clf__metric": ["euclidean", "manhattan"]}),
    "Naive Bayes": (Pipeline([("clf", GaussianNB())]),
                    {"clf__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]}),
}

results, best_models, best_params, cm_store, roc_store = [], {}, {}, {}, {}
default_models = {
    "SVM": Pipeline([("sc", StandardScaler()), ("clf", SVC(probability=True, random_state=RS))]),
    "Decision Tree": DecisionTreeClassifier(random_state=RS),
    "KNN": Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier())]),
    "Naive Bayes": GaussianNB(),
}
baseline = {}
for name, m in default_models.items():
    m.fit(X_train, y_train)
    baseline[name] = accuracy_score(y_test, m.predict(X_test))

for name, (pipe, grid) in specs.items():
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=1)
    t0 = time.time(); gs.fit(X_train, y_train); fit_t = time.time() - t0
    model = gs.best_estimator_
    t0 = time.time(); pred = model.predict(X_test); pred_t = time.time() - t0
    proba = model.predict_proba(X_test)
    cvs = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    tr_acc = accuracy_score(y_train, model.predict(X_train))
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
        "Log Loss": log_loss(y_test, proba),
        "ROC-AUC": roc_auc_score(y_test, proba[:, 1]),
        "CV Acc (mean)": cvs.mean(), "CV Acc (std)": cvs.std(),
        "Train Acc": tr_acc,
        "Fit+Tune Time (s)": fit_t, "Predict Time (s)": pred_t,
        "Baseline Acc (default params)": baseline[name],
    }
    results.append(row)
    best_models[name] = model
    best_params[name] = {k.replace("clf__", ""): (v if not isinstance(v, np.generic) else v.item())
                         for k, v in gs.best_params_.items()}
    cm_store[name] = confusion_matrix(y_test, pred)
    fpr, tpr, _ = roc_curve(y_test, proba[:, 1]); roc_store[name] = (fpr, tpr)
    print(f"\n{name}: best params = {best_params[name]}")
    print(f"  Acc={row['Accuracy']:.4f} P={row['Precision']:.4f} R={row['Recall']:.4f} "
          f"F1={row['F1-Score']:.4f} LogLoss={row['Log Loss']:.4f} AUC={row['ROC-AUC']:.4f}")

res = pd.DataFrame(results)
print("\n" + "=" * 60 + "\nFINAL COMPARISON\n" + "=" * 60)
print(res.round(4).to_string(index=False))

# 4. PLOTS
names = list(best_models)
# 4a. metric comparison
fig, ax = plt.subplots(figsize=(9, 5))
w = 0.2; xs = np.arange(len(names))
for i, met in enumerate(["Accuracy", "Precision", "Recall", "F1-Score"]):
    ax.bar(xs + i * w, res[met], w, label=met)
ax.set_xticks(xs + 1.5 * w); ax.set_xticklabels(names); ax.set_ylim(0.85, 1.0)
ax.set_ylabel("Score"); ax.set_title("Classification Metrics Comparison"); ax.legend(); ax.grid(axis="y")
plt.tight_layout(); plt.savefig(OUT + "metrics.png", dpi=150); plt.show()
# 4b. confusion matrices
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for a, n in zip(axes, names):
    ConfusionMatrixDisplay(cm_store[n], display_labels=["Malignant", "Benign"]).plot(
        ax=a, colorbar=False, cmap="Blues"); a.set_title(n)
plt.tight_layout(); plt.savefig(OUT + "confusion.png", dpi=150); plt.show()
# 4c. ROC
plt.figure(figsize=(6.5, 5))
for n in names:
    plt.plot(*roc_store[n], label=f"{n} (AUC={res.set_index('Model').loc[n,'ROC-AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend(loc="lower right"); plt.grid(True)
plt.tight_layout(); plt.savefig(OUT + "roc.png", dpi=150); plt.show()
# 4d. log loss
plt.figure(figsize=(6.5, 4.5))
plt.bar(names, res["Log Loss"], color="tab:red"); plt.ylabel("Log Loss (lower is better)")
plt.title("Log Loss Comparison"); plt.grid(axis="y")
plt.tight_layout(); plt.savefig(OUT + "logloss.png", dpi=150); plt.show()
# 4e. KNN k vs accuracy
ks = range(1, 26); accs = []
for k in ks:
    p = Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(k))])
    accs.append(cross_val_score(p, X_train, y_train, cv=cv).mean())
plt.figure(figsize=(6.5, 4.5)); plt.plot(ks, accs, marker="o")
plt.xlabel("k"); plt.ylabel("5-fold CV Accuracy"); plt.title("KNN: Effect of k"); plt.grid(True)
plt.tight_layout(); plt.savefig(OUT + "knn_k.png", dpi=150); plt.show()
# 4f. Decision tree depth: train vs CV
depths = range(1, 11); tr, te = [], []
for d in depths:
    p = DecisionTreeClassifier(max_depth=d, random_state=RS).fit(X_train, y_train)
    tr.append(p.score(X_train, y_train))
    te.append(cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=RS), X_train, y_train, cv=cv).mean())
plt.figure(figsize=(6.5, 4.5)); plt.plot(depths, tr, marker="o", label="Train")
plt.plot(depths, te, marker="s", label="CV"); plt.xlabel("max_depth"); plt.ylabel("Accuracy")
plt.title("Decision Tree: Depth vs Accuracy"); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig(OUT + "dt_depth.png", dpi=150); plt.show()

# 5. SAVE
out = {"results": results, "best_params": best_params,
       "cm": {k: v.tolist() for k, v in cm_store.items()},
       "knn_cv": dict(zip(ks, accs)), "dt_train": tr, "dt_cv": te,
       "n_train": len(X_train), "n_test": len(X_test)}
json.dump(out, open(OUT + "results.json", "w"), indent=1, default=float)

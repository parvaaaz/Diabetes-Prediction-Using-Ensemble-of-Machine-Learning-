"""۶ مدل مقاله + گریدهای کوچک/کامل + ساخت Pipeline (impute -> scale -> select/reduce -> clf)."""
import numpy as np
import sklearn
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.decomposition import FastICA, PCA
from sklearn.impute import KNNImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing import MRMRSelector

SEED = 42
SCORING = "roc_auc_ovo"   # مثل مقاله: انتخاب هایپرپارامتر بر اساس AUC

MODELS = {
    "KNN": lambda: KNeighborsClassifier(),
    "SVM": lambda: SVC(probability=True, cache_size=1024, random_state=SEED),
    "DT":  lambda: DecisionTreeClassifier(random_state=SEED),
    # n_jobs=1: the outer hyper-parameter search is already parallel (n_jobs=8);
    # nested parallelism would oversubscribe the CPU
    "RF":  lambda: RandomForestClassifier(random_state=SEED, n_jobs=1),
    "GNB": lambda: GaussianNB(),
    # sklearn>=1.8 dropped the "algorithm" param; SAMME is the only implementation
    "AB":  lambda: AdaBoostClassifier(random_state=SEED) if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
                   else AdaBoostClassifier(algorithm="SAMME", random_state=SEED),
}

# Small grid - smoke runs only (FAST=True)
GRIDS_SMALL = {
    "KNN": {"clf__n_neighbors": [5, 11], "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"]},
    "SVM": {"clf__C": [1, 10], "clf__gamma": ["scale", 0.1]},
    "DT":  {"clf__criterion": ["gini", "entropy"], "clf__max_depth": [6, None]},
    "RF":  {"clf__n_estimators": [100], "clf__max_depth": [8, None],
            "clf__max_features": ["sqrt", "log2"]},
    "GNB": {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]},
    "AB":  {"clf__n_estimators": [50, 100], "clf__learning_rate": [0.5, 1.0]},
}

# Full grid as in the paper - heavy runs (FAST=False, Bayesian search via skopt).
# Deviation vs paper: KNN "algorithm"/"leaf_size" axes are dropped (they change
# nothing in the prediction and create ~1600 pointless configs). docs/DEVIATIONS.md
GRIDS_FULL = {
    "KNN": {"clf__n_neighbors": list(range(1, 26)), "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"]},
    "SVM": {"clf__C": [0.1, 1, 10, 100], "clf__gamma": [0.01, 0.1, 1, "scale"],
            "clf__kernel": ["rbf", "linear", "poly", "sigmoid"]},
    "DT":  {"clf__criterion": ["gini", "entropy"], "clf__max_depth": [4, 8, 12, 16],
            "clf__min_samples_split": [0.05, 0.1, 0.2],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", None]},
    "RF":  {"clf__n_estimators": [50, 100, 150], "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [8, 12, None], "clf__min_samples_leaf": [1, 2],
            "clf__max_features": ["sqrt", "log2"]},
    "GNB": {"clf__var_smoothing": list(np.logspace(-9, -1, 9))},
    "AB":  {"clf__n_estimators": [50, 100, 150], "clf__learning_rate": [0.1, 0.5, 1.0]},
}



# Restricted SVM grid for the large dataset (n>2000): poly/sigmoid kernels and
# C=100/gamma=1 need more than 590 s per fit at n=5922. docs/DEVIATIONS.md
_GRIDS_FULL_SVM_LITE = {"clf__C": (0.1, 10.0, "log-uniform"),
                        "clf__gamma": (0.01, 0.1, "log-uniform")}
GRIDS_FULL = dict(GRIDS_FULL)  # copy-safe
GRIDS_FULL["SVM__LITE"] = _GRIDS_FULL_SVM_LITE

def selector_from_name(name):
    """نام رشته‌ای سلکتور در CSVها → دیکشنری سلکتور:
    'all_features' → None | 'mrmr_8' | 'pca_0.95' | 'ica_4'"""
    if name is None or name in ("all_features", "", "nan"):
        return None
    kind, val = str(name).split("_", 1)
    if kind == "mrmr":
        return {"kind": "mrmr", "k": int(val)}
    if kind == "pca":
        return {"kind": "pca", "var": float(val)}
    if kind == "ica":
        return {"kind": "ica", "k": int(val)}
    raise ValueError(f"سلکتور ناشناخته: {name}")


def make_pipe(prep, model_name, selector=None, params=None):
    """prep: 'I' (فقط impute) | 'N' (impute+MinMax) | 'Z' (impute+Standard)
    selector: None | {'kind':'mrmr','k':k} | {'kind':'pca','var':0.95} | {'kind':'ica','k':5}"""
    steps = [("impute", KNNImputer(n_neighbors=5))]
    if prep == "N":
        steps.append(("scale", MinMaxScaler()))
    elif prep == "Z":
        steps.append(("scale", StandardScaler()))
    if selector is not None:
        if selector["kind"] == "mrmr":
            steps.append(("fs", MRMRSelector(k=selector["k"])))
        elif selector["kind"] == "pca":
            steps.append(("fs", PCA(n_components=selector["var"])))
        elif selector["kind"] == "ica":
            steps.append(("fs", FastICA(n_components=selector["k"], random_state=SEED, max_iter=2000)))
    clf = MODELS[model_name]()
    if params:
        clf.set_params(**params)
    steps.append(("clf", clf))
    return Pipeline(steps)

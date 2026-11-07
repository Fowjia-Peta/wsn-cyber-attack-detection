"""Train and compare AdaBoost, Gradient Boosting and LDA on a real WSN dataset (e.g. WSN-DS).
Usage:  python train.py data/WSN-DS.csv
Writes: model/best_model.joblib, model/results.csv, model/confusion_*.png, RESULTS.md
RESULTS.md holds the dataset facts and metrics - quote numbers ONLY from this file."""
import joblib, os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
CSV = sys.argv[1] if len(sys.argv) > 1 else "data/wsn_data.csv"
GUESS = ["attack type", "attack_type", "label", "class", "target"]
NORMAL_NAMES = {"normal", "benign", "0"}

df = pd.read_csv(CSV)
df.columns = [c.strip() for c in df.columns]
label = next((c for c in df.columns if c.lower() in GUESS), df.columns[-1])
raw_rows = len(df)
df = df.drop_duplicates().dropna()
X = df.drop(columns=[label])
X = X.drop(columns=[c for c in X.columns if c.lower() in ("id", "index")])
X = X.select_dtypes("number")
y_text = df[label].astype(str).str.strip()
le = LabelEncoder(); y = le.fit_transform(y_text)
classes = list(le.classes_)
normal_idx = [i for i, c in enumerate(classes) if c.lower() in NORMAL_NAMES]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
models = {"AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=SEED),
          "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
          "LDA": LinearDiscriminantAnalysis()}

rows, pipes = [], {}
os.makedirs("model", exist_ok=True)
for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)]).fit(Xtr, ytr)
    pred = pipe.predict(Xte); pipes[name] = pipe
    r = {"model": name,
         "accuracy": accuracy_score(yte, pred),
         "precision_weighted": precision_score(yte, pred, average="weighted", zero_division=0),
         "recall_weighted": recall_score(yte, pred, average="weighted", zero_division=0),
         "f1_weighted": f1_score(yte, pred, average="weighted", zero_division=0)}
    if normal_idx:   # attack-vs-normal view: ATTACK is the positive class
        yb, pb = (~np.isin(yte, normal_idx)).astype(int), (~np.isin(pred, normal_idx)).astype(int)
        r.update(attack_precision=precision_score(yb, pb, zero_division=0),
                 attack_recall=recall_score(yb, pb, zero_division=0),
                 attack_f1=f1_score(yb, pb, zero_division=0))
    rows.append(r)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=classes).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"{name} - confusion matrix"); fig.tight_layout(); fig.savefig(f"model/confusion_{name}.png"); plt.close(fig)

res = pd.DataFrame(rows).set_index("model")
pct = (res * 100).round(2)
pct.to_csv("model/results.csv")
best = res["f1_weighted"].idxmax()
joblib.dump({"pipeline": pipes[best], "features": list(X.columns), "classes": classes, "name": best},
            "model/best_model.joblib")

def md_table(d):
    cols = ["model"] + list(d.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name, r in d.iterrows():
        out.append("| " + " | ".join([name] + [f"{v:.2f}" for v in r.values]) + " |")
    return "\n".join(out)

counts = y_text.value_counts()
md = [f"# Results\n", f"- Dataset file: `{os.path.basename(CSV)}`",
      f"- Rows: {raw_rows} (after removing duplicates/missing: {len(df)})",
      f"- Features: {X.shape[1]} ({', '.join(X.columns)})",
      f"- Label column: `{label}`", "- Class counts:"]
md += [f"  - {k}: {v}" for k, v in counts.items()]
md += [f"- Split: 80% train ({len(ytr)}) / 20% test ({len(yte)}), stratified, random_state={SEED}",
       f"- Best model (weighted F1): **{best}**\n", "## Metrics on the test set (%)\n", md_table(pct)]
open("RESULTS.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))

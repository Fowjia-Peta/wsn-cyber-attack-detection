"""Train and compare AdaBoost, Gradient Boosting and LDA for attack detection in a WSN."""
import joblib, json, os, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "data/wsn_data.csv"
df = pd.read_csv(CSV).dropna()
X, y_text = df.drop(columns=["label"]), df["label"]
le = LabelEncoder()
y = le.fit_transform(y_text)           # attack/normal -> 0/1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "LDA": LinearDiscriminantAnalysis(),
}

results, pipes = {}, {}
for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }
    pipes[name] = pipe
    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred, target_names=le.classes_))

res = pd.DataFrame(results).T.round(4)
print("\nComparison:\n", res)

os.makedirs("model", exist_ok=True)
best = res["f1"].idxmax()
joblib.dump({"pipeline": pipes[best], "features": list(X.columns),
             "classes": list(le.classes_), "name": best}, "model/best_model.joblib")
res.to_csv("model/results.csv")
res[["accuracy", "f1"]].plot(kind="bar", ylim=(0, 1.05), title="Model comparison")
plt.tight_layout(); plt.savefig("model/comparison.png")
print(f"\nBest model: {best} (saved to model/best_model.joblib)")

import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("students.csv").dropna()
X, y = df.drop(columns="result"), df["result"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
}
best, best_acc = None, 0
for name, m in models.items():
    m.fit(Xtr, ytr)
    acc = accuracy_score(yte, m.predict(Xte))
    print(f"{name}: accuracy={acc:.3f}")
    if acc > best_acc: best, best_acc, best_name = m, acc, name

pred = best.predict(Xte)
print("\nBest:", best_name)
print(classification_report(yte, pred))
print(confusion_matrix(yte, pred))
joblib.dump({"model": best, "features": list(X.columns)}, "model.joblib")

"""Create a synthetic student dataset (students.csv). Replace with a real dataset (e.g. UCI Student Performance) if you have one."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 1000
df = pd.DataFrame({
    "study_hours": rng.uniform(0, 10, n).round(1),
    "attendance": rng.uniform(40, 100, n).round(1),
    "previous_score": rng.uniform(30, 100, n).round(1),
    "assignments_done": rng.integers(0, 11, n),
    "sleep_hours": rng.uniform(4, 9, n).round(1),
})
score = (df.study_hours * 4 + df.attendance * 0.25 + df.previous_score * 0.35
         + df.assignments_done * 1.5 + rng.normal(0, 4, n))
df["result"] = pd.cut(score, bins=[-1, 55, 70, 1000], labels=["Low", "Medium", "High"])
df.to_csv("students.csv", index=False)
print("saved students.csv", df.shape)

# evaluate.py (version blindée)
import os
from pathlib import Path

# 0) forcer un backend qui n'a pas besoin d'écran
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("📂 cwd        :", os.getcwd())
print("📂 script dir :", BASE_DIR)
print("📂 reports    :", REPORTS_DIR)

# 1) test d'écriture simple
test_file = REPORTS_DIR / "DEBUG_EVALUATE_RAN.txt"
test_file.write_text("evaluate.py a bien été exécuté jusqu'en bas.\n")
print(f"📝 Fichier texte créé : {test_file}")

# 2) données
iris = load_iris()
X, y = iris.data, iris.target
X_df = pd.DataFrame(X, columns=iris.feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=42
)

# 3) modèle
model_path = BASE_DIR / "models" / "iris_model.pkl"
print("🔎 loading model from :", model_path)
model = joblib.load(model_path)

# 4) prédiction
y_pred = model.predict(X_test)
print("✅ prediction OK")

# 5) matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)
plt.title("Confusion Matrix - Iris Classification")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
cm_path = REPORTS_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"✅ confusion matrix saved to {cm_path}")

# 6) feature importance
if hasattr(model, "feature_importances_"):
    fi = model.feature_importances_
    idx = np.argsort(fi)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(fi)), fi[idx])
    plt.xticks(
        range(len(fi)),
        [iris.feature_names[i] for i in idx],
        rotation=45,
    )
    plt.title("Feature importance")
    plt.tight_layout()
    fi_path = REPORTS_DIR / "feature_importance.png"
    plt.savefig(fi_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✅ feature importance saved to {fi_path}")
else:
    print("ℹ️ modèle sans feature_importances_ → on saute la 2e figure.")

print("🎉 Terminé.")
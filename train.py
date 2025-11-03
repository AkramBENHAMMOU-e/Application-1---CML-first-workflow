import os
import json
import joblib
import pickle
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ==============================
# 1. Préparation des dossiers
# ==============================
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ==============================
# 2. Chargement des données
# ==============================
iris = load_iris()
X, y = iris.data, iris.target

# Simplifier à un problème binaire : Setosa (0) vs Versicolor (1)
mask = y < 2
X, y = X[mask], y[mask]

# ==============================
# 3. Split train/test
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ==============================
# 4. Entraînement des modèles
# ==============================
pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)

pipe_lr.fit(X_train, y_train)
clf_rf.fit(X_train, y_train)

# ==============================
# 5. Évaluation
# ==============================
y_pred_lr = pipe_lr.predict(X_test)
y_pred_rf = clf_rf.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_lr = f1_score(y_test, y_pred_lr)
f1_rf = f1_score(y_test, y_pred_rf)

# Choisir le meilleur modèle
best_model_name = "logistic_regression" if acc_lr >= acc_rf else "random_forest"
best_model = pipe_lr if acc_lr >= acc_rf else clf_rf

print("=== Résultats ===")
print(f"Logistic Regression -> Acc: {acc_lr:.4f}, F1: {f1_lr:.4f}")
print(f"Random Forest       -> Acc: {acc_rf:.4f}, F1: {f1_rf:.4f}")
print(f"✅ Meilleur modèle : {best_model_name}")

# ==============================
# 6. Sauvegardes
# ==============================

# Modèle (.pkl)
model_path = f"models/{best_model_name}_iris.pkl"
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

# Données de test
test_data = pd.DataFrame(X_test, columns=iris.feature_names)
test_data["target"] = y_test
test_data.to_csv("data/iris_test.csv", index=False)

# Métriques
metrics = {
    "best_model": best_model_name,
    "accuracy": round(max(acc_lr, acc_rf), 4),
    "f1_score": round(max(f1_lr, f1_rf), 4),
    "test_samples": len(X_test)
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n📁 Modèle sauvegardé dans : {model_path}")
print(f"📊 Métriques sauvegardées dans : metrics.json")

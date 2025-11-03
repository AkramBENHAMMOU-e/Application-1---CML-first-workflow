import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
import joblib

# Charger les donnees
iris = load_iris()
X, y = iris.data, iris.target

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrainement
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Sauvegarder le modele
joblib.dump(model, 'models/iris_model.pkl')

# Sauvegarder les metriques
metrics = {
    "accuracy": accuracy,
    "n_estimators": 100,
    "test_size": len(X_test)
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Accuracy: {accuracy:.4f}")

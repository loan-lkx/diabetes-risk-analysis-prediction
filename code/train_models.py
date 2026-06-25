import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'cleaned_diabetes.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

models = {
    'knn': KNeighborsClassifier(n_neighbors=5),
    'logistic': LogisticRegression(max_iter=1000, random_state=42),
    'decision_tree': DecisionTreeClassifier(random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, f'{name}.pkl'))

tuned_rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}
grid = GridSearchCV(tuned_rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train_scaled, y_train)

best_rf = grid.best_estimator_
joblib.dump(best_rf, os.path.join(MODEL_DIR, 'tuned_random_forest.pkl'))
joblib.dump(grid.best_params_, os.path.join(MODEL_DIR, 'tuned_rf_params.pkl'))

train_auc = roc_auc_score(y_train, best_rf.predict_proba(X_train_scaled)[:, 1])
test_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test_scaled)[:, 1])

print(f"Tuned Random Forest best params: {grid.best_params_}")
print(f"Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}")
print("All models trained and saved successfully!")

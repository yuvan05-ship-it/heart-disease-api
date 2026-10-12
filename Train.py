import joblib
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r"heart.csv")
data.drop_duplicates(inplace=True)
X, y = data.drop(columns="target"), data["target"]
print("Rows after dedup:", len(data))
print("Class balance:\n", y.value_counts(normalize=True).sort_index().round(3))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

lr = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
lr_scores = cross_val_score(lr, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"LogReg CV AUC: {lr_scores.mean():.4f} +/- {lr_scores.std():.4f}")

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions={
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 12),
        "min_samples_leaf": randint(1, 8),
    },
    n_iter=30, cv=cv, scoring="roc_auc", random_state=42, n_jobs=-1,
)
rf_search.fit(X_train, y_train)
print("RF best params:", rf_search.best_params_)
print(f"RF CV AUC: {rf_search.best_score_:.4f}  (slightly optimistic: tuned on these folds)")

if rf_search.best_score_ > lr_scores.mean():
    name, model = "RandomForest", rf_search.best_estimator_
else:
    name, model = "LogisticRegression", lr.fit(X_train, y_train)
print("Selected:", name)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, pred))
print(f"Test ROC AUC: {roc_auc_score(y_test, proba):.4f}  (test set is small: expect noise)")

joblib.dump(model, "model.pkl")
print("Saved model.pkl")
# mall_segmenter.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ----------------------------
# Configuration
# ----------------------------

CAT_FEATURES = ["Genre"]
NUM_FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
CSV_PATH = "/Users/hashibk/Documents/Internship/week3/Mall_Customers.csv"

# ----------------------------
# Preprocessing
# ----------------------------

def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path).drop(columns=["CustomerID"])

    # IQR-based outlier removal
    for col in NUM_FEATURES:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df.reset_index(drop=True)

def create_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT_FEATURES)
    ])

# ----------------------------
# Pipelines
# ----------------------------

def create_cluster_pipeline(preprocessor):
    return Pipeline([
        ("prep", preprocessor),
        ("pca", PCA(n_components=2, random_state=42)),
        ("kmeans", KMeans(n_clusters=4, random_state=42))
    ])

def create_rf_pipeline(preprocessor):
    return Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestClassifier(random_state=42))
    ])

def create_lgbm_pipeline(preprocessor, num_classes):
    return Pipeline([
        ("prep", preprocessor),
        ("pca", PCA(n_components=2, random_state=42)),
        ("lgbm", LGBMClassifier(objective="multiclass", num_class=num_classes, random_state=42))
    ])

# ----------------------------
# Evaluation
# ----------------------------

def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    print(f"\n{name} — Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# ----------------------------
# Prediction for New Customer
# ----------------------------

def predict_segment(model, data):
    return model.predict(data)[0]

# ----------------------------
# Main
# ----------------------------

def main():
    df = load_and_clean_data(CSV_PATH)
    preprocessor = create_preprocessor()

    X = df[CAT_FEATURES + NUM_FEATURES]
    X_train, X_test = train_test_split(X, test_size=0.25, random_state=42)

    # --------- Clustering ----------
    cluster_pipe = create_cluster_pipeline(preprocessor)
    cluster_pipe.fit(X_train)
    joblib.dump(cluster_pipe, "cluster_pipeline.joblib")

    y_train = cluster_pipe.predict(X_train)
    y_test = cluster_pipe.predict(X_test)

    print("Cluster sizes (train):", np.bincount(y_train))

    # --------- Random Forest ----------
    rf_pipe = create_rf_pipeline(preprocessor)
    rf_params = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [None, 10, 20],
        "rf__min_samples_split": [2, 5],
        "rf__min_samples_leaf": [1, 2]
    }

    grid_rf = GridSearchCV(rf_pipe, rf_params, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                           scoring="accuracy", n_jobs=-1, verbose=1)
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    joblib.dump(best_rf, "best_rf.joblib")
    evaluate(best_rf, X_test, y_test, "Random Forest")

    # --------- LightGBM ----------
    lgbm_pipe = create_lgbm_pipeline(preprocessor, len(np.unique(y_train)))
    lgbm_params = {
        "lgbm__n_estimators": [100, 200],
        "lgbm__learning_rate": [0.05, 0.1],
        "lgbm__max_depth": [3, 5],
        "lgbm__num_leaves": [31, 63],
    }

    grid_lgbm = GridSearchCV(lgbm_pipe, lgbm_params, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                             scoring="accuracy", n_jobs=-1, verbose=1)
    grid_lgbm.fit(X_train, y_train)
    best_lgbm = grid_lgbm.best_estimator_
    joblib.dump(best_lgbm, "lgbm_classifier.joblib")
    evaluate(best_lgbm, X_test, y_test, "LightGBM")

    # --------- Demo prediction ----------
    demo_customer = pd.DataFrame([[1, 30, 70, 60]], columns=CAT_FEATURES + NUM_FEATURES)
    segment = predict_segment(best_rf, demo_customer)
    print("\nPredicted Segment for New Customer:", segment)

# ----------------------------
# Entry Point
# ----------------------------

if __name__ == "__main__":
    main()

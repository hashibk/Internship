# Mall Customer Segmentation & Classification

This project performs customer segmentation on the Mall Customers dataset using K-Means clustering and classifies new customers into segments using supervised machine learning models.

## 📊 Project Overview

- **Dataset**: Mall_Customers.csv
- **Techniques**:
  - Outlier removal using IQR
  - Feature scaling with StandardScaler
  - Dimensionality reduction with PCA
  - Clustering using K-Means
  - Cluster evaluation using Silhouette Score & Davies-Bouldin Index
  - Classification using Random Forest and LightGBM

## 🔁 Workflow

1. **Preprocessing**: Handled nulls, duplicates, outliers, and label-encoded categorical features.
2. **Scaling**: Standardized numeric features.
3. **PCA**: Reduced features to 2D for visualization.
4. **Clustering**: Applied K-Means, determined optimal `k` using the Elbow method.
5. **Evaluation**: Used Silhouette Score and Davies-Bouldin Index to evaluate clustering.
6. **Classification**: Trained RandomForest and LightGBM classifiers to predict clusters.

## 📦 Requirements

- pandas
- seaborn
- matplotlib
- scikit-learn
- lightgbm

Install all dependencies with:

```bash
pip install -r requirements.txt
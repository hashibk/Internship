# 🧠 mall_segmenter.py — Customer Segmentation & Classification Pipeline

This script implements a **complete machine learning pipeline** to:

- Preprocess mall customer data
- Perform **KMeans clustering** to create customer segments
- Train and evaluate classification models (Random Forest & LightGBM)
- Predict the segment for a **new customer**
- Save all models using `joblib` for production use

---

## 📁 File Location

📦 Internship/
└── 📂 Week3-Dev/
└── 📂 MallCustomers/
├── mall_segmenter.py ← (this file)
├── Mall_Customers.csv
├── best_rf.joblib
├── lgbm_classifier.joblib
├── cluster_pipeline.joblib
└── 📂 readmefiles/
└── functions_README.md


---

## 📦 Required Libraries

Make sure the following are installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm joblib
📊 Dataset Used

The script reads from:

Mall_Customers.csv
Expected columns:

CustomerID (removed early in preprocessing)
Genre (categorical: "Male", "Female")
Age
Annual Income (k$)
Spending Score (1–100)
🔄 Pipeline Stages

1️⃣ Load & Clean Data
Drops CustomerID
Removes outliers using IQR filtering on numerical columns
2️⃣ Preprocessing Pipeline
StandardScaler for numeric features
OneHotEncoder for the Genre column (drop='first' to avoid multicollinearity)
3️⃣ Clustering Pipeline
Applies PCA for dimensionality reduction
Uses KMeans with n_clusters=4
Trained only on training split
Labels assigned to both train and test for classification
4️⃣ Random Forest Classifier
Performs Grid Search with CV for best hyperparameters
Saves model as best_rf.joblib
5️⃣ LightGBM Classifier
Also uses GridSearchCV with a parameter grid
Applies PCA before classification
Saves model as lgbm_classifier.joblib
🎯 Output Metrics

Accuracy
Classification report (Precision, Recall, F1-score)
Confusion matrix plotted using seaborn
💡 Prediction Demo

At the end of main(), a test customer is created:

demo_customer = pd.DataFrame([[1, 30, 70, 60]], columns=CAT_FEATURES + NUM_FEATURES)
The predicted cluster is printed using the best random forest model.

💾 Output Files

The following files are saved for API or production use:

File Name	Description
best_rf.joblib	Random Forest classifier
lgbm_classifier.joblib	LightGBM classifier
cluster_pipeline.joblib	KMeans + PCA pipeline
▶️ To Run the Script

python mall_segmenter.py
🧩 Functions Breakdown

Function	Purpose
load_and_clean_data()	Loads CSV & removes outliers
create_preprocessor()	Returns ColumnTransformer with encoders
create_cluster_pipeline()	Creates PCA + KMeans pipeline
create_rf_pipeline()	Creates RandomForest classification pipe
create_lgbm_pipeline()	Creates LightGBM classification pipe
evaluate()	Prints accuracy, report, and heatmap
predict_segment()	Predicts cluster for new customer
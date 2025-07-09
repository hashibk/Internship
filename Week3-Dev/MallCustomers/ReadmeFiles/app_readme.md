# 🌀 app.py — FastAPI Endpoint for Customer Segmentation

This script (`app.py`) serves as a **REST API** built using **FastAPI** to predict the customer segment (cluster) for a shopping mall customer based on their demographics and spending behavior.

---

## 🚀 Overview

The API takes in a customer's profile as input (via JSON), and returns a predicted **segment** using a previously trained **Random Forest** model saved as a `.joblib` pipeline.

The model pipeline handles:
- Label encoding
- Scaling / OneHotEncoding
- Any preprocessing steps
- Final prediction via Random Forest classifier

---

## 📦 Dependencies

Ensure the following Python libraries are installed:

```bash
pip install fastapi uvicorn pandas joblib pydantic
🧠 Model File

The file expects a trained model pipeline saved as:

best_rf.joblib
This file should include preprocessing steps (e.g., OneHotEncoding, scaling) and the trained Random Forest model.

🎯 Input Format (JSON)

The POST endpoint /predict expects the following JSON payload:

{
  "Genre": "Female",
  "Age": 30,
  "Annual_Income_k": 70,
  "Spending_Score": 60
}
Field	Type	Description
Genre	string	Gender of the customer (e.g., "Male", "Female")
Age	int	Age of the customer
Annual_Income_k	float	Annual income in 1000s of dollars
Spending_Score	float	Spending score (1–100)
🔄 Output Format

{
  "predicted_segment": 2
}
If something goes wrong:

{
  "error": "Error message here"
}
▶️ Running the API

Run the FastAPI server using Uvicorn:

uvicorn app:app --reload
Then access the auto-generated Swagger UI docs at:

📍 http://127.0.0.1:8000/docs

🔁 Sample CURL Request

curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Genre": "Female",
  "Age": 30,
  "Annual_Income_k": 70,
  "Spending_Score": 60
}'
Expected Output:

{
  "predicted_segment": 1
}
📁 File Structure

📦 Internship/
└── 📂 Week3-Dev/
    └── 📂 MallCustomers/
        ├── app.py
        ├── best_rf.joblib
        └── 📂 readmefiles/
            └── app_README.md  ← (this file)
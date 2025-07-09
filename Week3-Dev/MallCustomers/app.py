from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load the trained pipeline (which includes preprocessing and model)
model = joblib.load("best_rf.joblib")

# Pydantic model
class CustomerInput(BaseModel):
    Genre: str
    Age: int
    Annual_Income_k: float
    Spending_Score: float

@app.post("/predict")
def predict_segment(customer: CustomerInput):
    try:
        # Prepare data as DataFrame with expected column names and correct types
        input_df = pd.DataFrame([{
            "Genre": customer.Genre,  # Keep as string
            "Age": customer.Age,
            "Annual Income (k$)": customer.Annual_Income_k,
            "Spending Score (1-100)": customer.Spending_Score
        }])

        # Predict using the joblib pipeline (includes OneHotEncoder, etc.)
        prediction = model.predict(input_df)[0]
        return {"predicted_segment": int(prediction)}

    except Exception as e:
        return {"error": str(e)}
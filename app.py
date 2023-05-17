# Load the libraries
from fastapi import FastAPI, HTTPException
import predictor

# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to Indonesian Language News Topic MuliClassification FastAPI"}

# API Route - Predict Sentiment for ALl Emiten. Return JSON.
@app.post("/predict_single_data")
def predict_single_data(news_title):
    if(not(news_title)):
        raise HTTPException(status_code=400, detail = "Please Provide a valid text message")

    return predictor.predict_single_data(news_title)

## API Route - Predict Sentiment for Spesific Emiten. Return JSON.
@app.post("/predict_data_collection")
def predict_data_collection(news_title):
    if(not(news_title)):
        raise HTTPException(status_code=400, detail = "Please Provide a valid text message")

    return predictor.predict_data_collection(news_title)

# End of Line - App.py #
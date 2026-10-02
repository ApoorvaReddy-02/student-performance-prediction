import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

bundle = joblib.load("model.joblib")
model, features = bundle["model"], bundle["features"]
app = FastAPI(title="Student Performance Prediction")

class Student(BaseModel):
    study_hours: float = Field(ge=0, le=24)
    attendance: float = Field(ge=0, le=100)
    previous_score: float = Field(ge=0, le=100)
    assignments_done: int = Field(ge=0, le=10)
    sleep_hours: float = Field(ge=0, le=24)

@app.post("/predict")
def predict(s: Student):
    row = [[getattr(s, f) for f in features]]
    label = model.predict(row)[0]
    probs = dict(zip(model.classes_, model.predict_proba(row)[0].round(3).tolist()))
    return {"prediction": label, "probabilities": probs}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

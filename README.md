# AI-Based Student Performance Prediction System
Classification model (Logistic Regression vs Random Forest) predicting student performance as Low / Medium / High, served via a FastAPI REST API with a web form.

## Tech
Python, Pandas, NumPy, Scikit-Learn, FastAPI, HTML/JS

## Run
```bash
pip install -r requirements.txt
python generate_data.py   # or put your own students.csv here (same columns)
python train.py           # trains, prints accuracy + confusion matrix, saves model.joblib
uvicorn app:app --reload  # open http://127.0.0.1:8000
```
## API
`POST /predict` with `study_hours, attendance, previous_score, assignments_done, sleep_hours`.

Note: the bundled dataset is synthetic. Swap in a real dataset and report real metrics.

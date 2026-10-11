# Heart Disease Prediction System

Hamza Tahir · FA24-BSE-142 · Section A
Machine Learning Fundamentals (AIC354), Assignment 2, COMSATS University Islamabad, Lahore Campus

**Live app:** ADD-STREAMLIT-LINK-HERE

This project predicts heart disease (Yes/No) from four examination results. It uses a Support Vector
Classifier trained on the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
(Cleveland subset). The pipeline follows the steps of the Titanic Passenger Survival Prediction System.

| Input | UCI column | Values |
|---|---|---|
| Sex | `sex` | Male, Female |
| ExerciseAngina | `exang` | Yes, No |
| MajorVessels | `ca` | Zero, One, Two, Three |
| STSlope | `slope` | Upsloping, Flat, Downsloping |

| Result | Value |
|---|---|
| Test accuracy (60 records) | 0.90 |
| 10-fold cross validation accuracy | 0.83 |

## Files

- `Heart_Disease_Prediction.ipynb`: the complete notebook
- `Heart_Disease_Report.pdf`: the notebook as a PDF
- `processed_cleveland.csv`: the UCI dataset used
- `heart_disease_model.pkl`: the trained model
- `streamlit-app/`: the web app deployed on Streamlit Community Cloud
- `sample-data*.csv`, `training-data-encoded.csv`, `testing-data-encoded.csv`, `model-predictions.csv`: created by the notebook

## Run

```bash
pip install -r requirements.txt
jupyter notebook                         # run Heart_Disease_Prediction.ipynb
pip install -r streamlit-app/requirements.txt
streamlit run streamlit-app/app.py       # run the web app locally
```

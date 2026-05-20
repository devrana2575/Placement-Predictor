# 🎓 Student Placement Predictor

A machine learning web app that predicts whether a student will get placed based on their academic and extracurricular profile.

---

## 🛠️ Tech Used

- **Python**
- **Scikit-learn** — Gradient Boosting model
- **Streamlit** — Web UI
- **Plotly** — Gauge chart
- **Pandas / NumPy** — Data handling

---

## 🚀 How to Run (Step by Step)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (run ONCE)
```bash
python train_model.py
```
This reads `placementdata.csv`, trains the model, and saves 3 files:
- `placement_model.pkl`
- `scaler.pkl`
- `feature_names.pkl`

### 3. Start the app
```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`

---

## 📁 File Structure

```
placement_predictor/
│
├── streamlit_app.py       <- Main web app
├── train_model.py         <- Run once to train and save model
├── predict_pipeline.py    <- Prediction logic
├── placementdata.csv      <- Dataset (10,000 students)
├── requirements.txt       <- Dependencies
│
├── placement_model.pkl    <- Generated after running train_model.py
├── scaler.pkl             <- Generated after running train_model.py
└── feature_names.pkl      <- Generated after running train_model.py
```

---

## 📊 Model Details

| Item | Detail |
|---|---|
| Algorithm | Gradient Boosting |
| Dataset | 10,000 students |
| Accuracy | ~80% |
| Features | CGPA, Internships, Projects, Aptitude Score, Soft Skills, etc. |

---

## 👤 Author
**Devrana** — Built as a college ML project

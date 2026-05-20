"""
train_model.py
Run once to retrain the model from scratch.
Usage: python train_model.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from scipy.special import expit
from scipy.stats import rankdata
import joblib
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
N = 25000

print("Generating dataset...")

cgpa            = np.round(np.random.uniform(4.0, 10.0, N), 1)
internships     = np.random.randint(0, 6, N)
projects        = np.random.randint(0, 11, N)
workshops       = np.random.randint(0, 8, N)
aptitude        = np.random.randint(30, 101, N)
soft_skills     = np.round(np.random.uniform(1.0, 5.0, N), 1)
extracurricular = np.random.choice([0, 1], N, p=[0.45, 0.55])
training        = np.random.choice([0, 1], N, p=[0.40, 0.60])
ssc             = np.random.randint(40, 101, N)
hsc             = np.random.randint(40, 101, N)
backlogs        = np.random.randint(0, 5, N)

# Percentile-rank each feature so scores are relative to the dataset
def pct(arr):
    return rankdata(arr, method='average') / len(arr)

raw_score = (
    pct(cgpa)            * 30 +
    pct(aptitude)        * 25 +
    pct(soft_skills)     * 15 +
    pct(ssc)             * 8  +
    pct(hsc)             * 8  +
    pct(internships)     * 6  +
    pct(projects)        * 4  +
    pct(workshops)       * 2  +
    extracurricular      * 1  +
    training             * 1
)
raw_score -= backlogs * 3

# Sigmoid converts score to probability — gives smooth realistic curve
prob_placed = expit((raw_score - 50) / 12)
placed = (np.random.uniform(0, 1, N) < prob_placed).astype(int)

print(f"Dataset: {N} students | Placement rate: {placed.mean()*100:.1f}%")

df = pd.DataFrame({
    "CGPA":                      cgpa,
    "Internships":               internships,
    "Projects":                  projects,
    "Workshops_Certifications":  workshops,
    "AptitudeTestScore":         aptitude,
    "SoftSkillsRating":          soft_skills,
    "ExtracurricularActivities": extracurricular,
    "PlacementTraining":         training,
    "SSC_Marks":                 ssc,
    "HSC_Marks":                 hsc,
    "Backlogs":                  backlogs,
    "PlacementStatus":           placed,
})
df.to_csv("placementdata.csv", index=False)

X = df.drop(columns=["PlacementStatus"])
y = df["PlacementStatus"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model...")
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.06,
        subsample=0.8, min_samples_leaf=30, random_state=42,
    )),
])
pipeline.fit(X_train, y_train)

preds  = pipeline.predict(X_test)
probas = pipeline.predict_proba(X_test)[:, 1]
acc    = accuracy_score(y_test, preds)
auc    = roc_auc_score(y_test, probas)
print(f"\nAccuracy : {acc*100:.1f}%")
print(f"ROC-AUC  : {auc:.3f}")
print(classification_report(y_test, preds, target_names=["Not Placed", "Placed"]))

joblib.dump(pipeline,               "placement_pipeline.pkl")
joblib.dump(X.columns.tolist(),     "feature_names.pkl")

print("✅ Model saved! Run:  streamlit run streamlit_app.py")

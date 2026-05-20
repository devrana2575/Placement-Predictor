"""
predict_pipeline.py
Loads the trained sklearn Pipeline and predicts placement probability.
"""
import pandas as pd
import joblib


def predict_single(cgpa, internships, projects, workshops,
                   aptitude, soft_skills, extracurricular,
                   placement_training, ssc, hsc, backlogs):
    """
    Returns: (placed: bool, probability: float 0-100)
    """
    pipeline = joblib.load("placement_pipeline.pkl")
    features = joblib.load("feature_names.pkl")

    row = pd.DataFrame([{
        "CGPA":                      cgpa,
        "Internships":               int(internships),
        "Projects":                  int(projects),
        "Workshops_Certifications":  int(workshops),
        "AptitudeTestScore":         int(aptitude),
        "SoftSkillsRating":          soft_skills,
        "ExtracurricularActivities": 1 if extracurricular == "Yes" else 0,
        "PlacementTraining":         1 if placement_training == "Yes" else 0,
        "SSC_Marks":                 int(ssc),
        "HSC_Marks":                 int(hsc),
        "Backlogs":                  int(backlogs),
    }])[features]

    prob   = round(pipeline.predict_proba(row)[0][1] * 100, 1)
    placed = prob >= 50

    return placed, prob

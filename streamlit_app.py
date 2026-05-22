"""
streamlit_app.py — Student Placement Predictor
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import plotly.graph_objects as go
import os
from predict_pipeline import predict_single
import time

st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓",
    layout="wide",
)

# ── Model check ────────────────────────────────────────────────────────────────
if not all(os.path.exists(f) for f in ["placement_pipeline.pkl", "feature_names.pkl"]):
    st.error("⚠️ Model not found. Run this first in your terminal:  python train_model.py")
    st.stop()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align: center;'>
        Student Placement Predictor
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align: center; font-size:18px; color:gray;'>
        Enter your details and get an instant prediction of your placement chances.
    </p>
    """,
    unsafe_allow_html=True
)
st.divider()

# ── Inputs ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Academic")
    cgpa     = st.slider("CGPA",                 4.0, 10.0, 7.0, 0.1)
    ssc      = st.slider("SSC — 10th Marks (%)", 40,  100,  65)
    hsc      = st.slider("HSC — 12th Marks (%)", 40,  100,  68)
    apt      = st.slider("Aptitude Test Score",  30,  100,  60)
    backlogs = st.selectbox("Active Backlogs",   [0, 1, 2, 3, 4])

with col2:
    st.subheader("🏆 Experience & Skills")
    internships = st.selectbox("Internships",                 [0, 1, 2, 3, 4, 5])
    projects    = st.selectbox("Projects",                    list(range(0, 11)))
    workshops   = st.selectbox("Workshops / Certifications",  list(range(0, 8)))
    soft_skills = st.slider("Soft Skills (out of 5)",         1.0, 5.0, 3.0, 0.1)
    col_a, col_b = st.columns(2)

    with col_a:
        extra = st.selectbox(
            "Extracurricular Acitivities",
            ["Yes", "No"],
        )
    with col_b:
        training = st.selectbox(
            "Placement Training",
            ["Yes", "No"],
        )

st.divider()

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("🔮 Predict My Placement Chances", use_container_width=True, type="primary"):

    with st.spinner("Analyzing profile..."):
        time.sleep(2)

        placed, prob = predict_single(
        cgpa=cgpa,
        internships=internships,
        projects=projects,
        workshops=workshops,
        aptitude=apt,
        soft_skills=soft_skills,
        extracurricular=extra,
        placement_training=training,
        ssc=ssc,
        hsc=hsc,
        backlogs=backlogs,
    )

    st.divider()

    # ── Result banner ──────────────────────────────────────────────────────────
    if prob >= 75:
        result_color = "#2ecc71"
        result_text = "✅ Strong Placement Chances"

    elif prob >= 50:
        result_color = "#27ae60"
        result_text = "✅ Likely to be Placed"

    elif prob >= 30:
        result_color = "#f39c12"
        result_text = "⚠️ Borderline Chances"

    else:
        result_color = "#e74c3c"
        result_text = "❌ Low Placement Chances"

    st.markdown(
        f"""
        <div style="
            background-color:{result_color};
            padding:15px;
            border-radius:12px;
            text-align:center;
            color:white;
            font-size:28px;
            font-weight:bold;
        ">
            {result_text}<br>
            {prob}%
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Gauge chart ────────────────────────────────────────────────────────────
    bar_color = "#2ecc71" if prob >= 50 else ("#f39c12" if prob >= 30 else "#e74c3c")

    fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prob,
    
    number={
        'suffix': "%",
        'font': {'size': 48}
    },

    title={
        'text': "Placement Probability",
        'font': {'size': 22}
    },

    gauge={
        'axis': {
            'range': [0, 100],
            'tickfont': {'size': 16}
        },

        'bar': {'thickness': 0.35},

        'steps': [
            {'range': [0, 40], 'color': "#f8d7da"},
            {'range': [40, 60], 'color': "#fff3cd"},
            {'range': [60, 100], 'color': "#d1e7dd"}
        ],

        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': prob
        }
    }
    ))

    fig.update_layout(
        height=500,
        margin=dict(l=40, r=40, t=80, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Score breakdown ────────────────────────────────────────────────────────
    st.subheader("📊 Score Breakdown")
    st.caption("How each part of your profile contributes (out of max points)")

    breakdown = [
        ("CGPA",                 (cgpa - 4.0) / 6.0,       30, "📚"),
        ("Aptitude Score",       (apt - 30) / 70,           25, "🧠"),
        ("Soft Skills",          (soft_skills - 1.0) / 4.0, 15, "🗣️"),
        ("SSC Marks",            (ssc - 40) / 60,           8,  "📝"),
        ("HSC Marks",            (hsc - 40) / 60,           8,  "📝"),
        ("Internships",          internships / 5,            6,  "💼"),
        ("Projects",             projects / 10,              4,  "🔨"),
        ("Workshops/Certs",      workshops / 7,              2,  "📜"),
        ("Extracurricular",      1 if extra == "Yes" else 0, 1,  "⚽"),
        ("Placement Training",   1 if training == "Yes" else 0, 1, "🎯"),
    ]

    backlog_penalty = backlogs * 3
    total_score = sum(ratio * max_pts for _, ratio, max_pts, _ in breakdown)
    total_score -= backlog_penalty
    total_max   = sum(max_pts for _, _, max_pts, _ in breakdown)

    for label, ratio, max_pts, icon in breakdown:
        your_pts = round(ratio * max_pts, 1)
        color    = "🟢" if ratio >= 0.7 else ("🟡" if ratio >= 0.4 else "🔴")
        st.progress(
            min(ratio, 1.0),
            text=f"{color} {icon} {label}: **{your_pts} / {max_pts}**"
        )

    if backlog_penalty > 0:
        st.error(f"⚠️ Backlog penalty: −{backlog_penalty} pts ({backlogs} backlog(s))")

    st.info(f"**Total Score: {max(total_score, 0):.1f} / {total_max}**")

    # ── Tips ───────────────────────────────────────────────────────────────────
    if not placed:
        st.subheader("💡 How to Improve")
        tips = []

        if cgpa < 7.5:
            tips.append(f"📚 **CGPA {cgpa}** — aim for 7.5+. Focus on core subjects and internal marks.")
        if apt < 65:
            tips.append(f"🧠 **Aptitude {apt}/100** — this is the biggest factor. Practice on IndiaBix or PrepInsta every day.")
        if soft_skills < 3.5:
            tips.append(f"🗣️ **Soft Skills {soft_skills}/5** — join debates, group discussions, or a communication course.")
        if internships == 0:
            tips.append("💼 **No internships** — apply on Internshala. Even a 1-month internship helps a lot.")
        if projects < 3:
            tips.append(f"🔨 **{projects} project(s)** — build 3+ projects and put them on GitHub with proper READMEs.")
        if workshops == 0:
            tips.append("📜 **No certifications** — complete 1-2 free courses on NPTEL, Coursera, or Udemy.")
        if training == "No":
            tips.append("🎯 **No placement training** — attend mock interviews and aptitude workshops at your college.")
        if backlogs > 0:
            tips.append(f"📋 **{backlogs} backlog(s)** — clear them immediately. Most companies filter out students with active backlogs.")
        if ssc < 60:
            tips.append("📝 **Low SSC marks** — can't change the past, so compensate with strong CGPA, skills, and projects.")

        if tips:
            for tip in tips:
                st.markdown(f"- {tip}")
        else:
            st.info("Your profile looks solid — focus on mock interviews and company research!")

    else:
        if prob >= 75:
            st.success("🎉 Great profile! Work on DSA, system design, and keep your GitHub active.")
        else:
            st.info("✅ Good chances! Push your aptitude score and soft skills even higher to strengthen your profile.")


# Footer
st.markdown("---")

st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 14px;'>
        © 2026 Placement Predictor • Developed by Dev Rana • All Rights Reserved<br>
        <a href='https://github.com/devrana2575' target='_blank'>GitHub</a> |
        <a href='https://linkedin.com/in/devrana2575' target='_blank'>LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)

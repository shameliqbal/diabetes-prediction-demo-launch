import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="wide",
)

# ─────────────────────────────────────────────
# Feature definitions (mirrors notebook exactly)
# ─────────────────────────────────────────────
COLS_NUM = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

COLS_CAT = [
    'race', 'gender', 'max_glu_serum', 'A1Cresult',
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
    'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
    'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone', 'change', 'diabetesMed', 'payer_code'
]

COLS_CAT_NUM = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']

TOP_10_SPEC = [
    'UNK', 'InternalMedicine', 'Emergency/Trauma',
    'Family/GeneralPractice', 'Cardiology', 'Surgery-General',
    'Nephrology', 'Orthopedics', 'Orthopedics-Reconstructive', 'Radiologist'
]

AGE_MAP = {
    '[0-10)': 0, '[10-20)': 10, '[20-30)': 20, '[30-40)': 30,
    '[40-50)': 40, '[50-60)': 50, '[60-70)': 60,
    '[70-80)': 70, '[80-90)': 80, '[90-100)': 90
}

# ─────────────────────────────────────────────
# Preprocessing (mirrors notebook pipeline)
# ─────────────────────────────────────────────

def preprocess_row(inputs: dict, col2use: list) -> pd.DataFrame:
    """Turn UI inputs into the same feature vector the model was trained on."""
    row = inputs.copy()

    # Replace missing markers
    for k in ['race', 'payer_code', 'medical_specialty']:
        if not row.get(k) or row[k] == '?':
            row[k] = 'UNK'

    # med_spec bucketing
    row['med_spec'] = row['medical_specialty'] if row['medical_specialty'] in TOP_10_SPEC else 'Other'

    # age numeric
    row['age_group'] = AGE_MAP.get(row.get('age', '[50-60)'), 50)

    # has_weight
    row['has_weight'] = 1 if row.get('weight') and row['weight'] != '?' else 0

    # Cast cat-num cols to str
    for c in COLS_CAT_NUM:
        row[c] = str(row.get(c, '1'))

    # Build a single-row DataFrame
    df = pd.DataFrame([row])

    # One-hot encode categorical columns
    all_cat_cols = COLS_CAT + COLS_CAT_NUM + ['med_spec']
    df_cat = pd.get_dummies(df[all_cat_cols], drop_first=True)

    # Combine
    df_num = df[COLS_NUM + ['age_group', 'has_weight']].copy()
    df_full = pd.concat([df_num.reset_index(drop=True),
                          df_cat.reset_index(drop=True)], axis=1)

    # Align to training columns
    for c in col2use:
        if c not in df_full.columns:
            df_full[c] = 0
    df_full = df_full[col2use]

    return df_full


# ─────────────────────────────────────────────
# Model loading / training
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def load_artifacts():
    """
    Try to load pre-trained model + scaler + column list from pickle files.
    If they don't exist, train a quick model on the diabetic_data.csv if available,
    otherwise fall back to a lightweight demo model.
    """
    scaler_path  = "pickle-files/scaler.sav"
    model_path   = "pickle-files/best_classifier.pkl"
    cols_path    = "pickle-files/col2use.pkl"

    if os.path.exists(scaler_path) and os.path.exists(model_path) and os.path.exists(cols_path):
        scaler  = pickle.load(open(scaler_path, 'rb'))
        model   = pickle.load(open(model_path,  'rb'))
        col2use = pickle.load(open(cols_path,   'rb'))
        return model, scaler, col2use, "pre-trained"

    # ── Train from CSV if available ──────────────────────────────────────
    csv_candidates = ["diabetic_data.csv", "data/diabetic_data.csv"]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)

    if csv_path:
        return _train_from_csv(csv_path)

    # ── Demo fallback ─────────────────────────────────────────────────────
    return _demo_model()


def _train_from_csv(csv_path):
    """Full pipeline matching the notebook."""
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import make_scorer, roc_auc_score

    df = pd.read_csv(csv_path)
    df = df.replace('?', np.nan)
    df = df.loc[~df.discharge_disposition_id.isin([11, 13, 14, 19, 20, 21])]
    df['readmission_status'] = (df.readmitted == '<30').astype('int')

    for c in ['race', 'payer_code', 'medical_specialty']:
        df[c] = df[c].fillna('UNK')

    df['med_spec'] = df['medical_specialty'].copy()
    df.loc[~df.med_spec.isin(TOP_10_SPEC), 'med_spec'] = 'Other'

    df[COLS_CAT_NUM] = df[COLS_CAT_NUM].astype('str')
    df_cat = pd.get_dummies(df[COLS_CAT + COLS_CAT_NUM + ['med_spec']], drop_first=True)

    df['age_group']  = df.age.replace(AGE_MAP)
    df['has_weight'] = df.weight.notnull().astype('int')

    cols_extra = ['age_group', 'has_weight']
    col2use = COLS_NUM + list(df_cat.columns) + cols_extra

    enc_df = pd.concat([df, df_cat], axis=1)
    df_data = enc_df[col2use + ['readmission_status']].dropna()

    df_data = df_data.sample(frac=1, random_state=42).reset_index(drop=True)
    df_valid_test = df_data.sample(frac=0.30, random_state=42)
    df_train_all  = df_data.drop(df_valid_test.index)

    rows_pos  = df_train_all.readmission_status == 1
    df_train  = pd.concat([
        df_train_all.loc[rows_pos],
        df_train_all.loc[~rows_pos].sample(n=rows_pos.sum(), random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    X_train = df_train[col2use].values
    y_train = df_train['readmission_status'].values
    X_all   = df_train_all[col2use].values

    scaler = StandardScaler()
    scaler.fit(X_all)
    X_train_tf = scaler.transform(X_train)

    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    random_grid = {
        'n_estimators': range(100, 300, 100),
        'max_depth':    range(2, 5, 1),
        'learning_rate':[0.01, 0.1]
    }
    auc_score = make_scorer(roc_auc_score)
    gbc_random = RandomizedSearchCV(gbc, random_grid, n_iter=6, cv=2,
                                    scoring=auc_score, random_state=42, verbose=0)
    gbc_random.fit(X_train_tf, y_train)
    best_model = gbc_random.best_estimator_

    os.makedirs("pickle-files", exist_ok=True)
    pickle.dump(scaler,     open("pickle-files/scaler.sav",         'wb'))
    pickle.dump(best_model, open("pickle-files/best_classifier.pkl",'wb'), protocol=4)
    pickle.dump(col2use,    open("pickle-files/col2use.pkl",        'wb'))

    return best_model, scaler, col2use, "trained on upload"


def _demo_model():
    """Tiny demo model with a representative feature set."""
    np.random.seed(42)
    n = 2000
    col2use = COLS_NUM + ['age_group', 'has_weight']

    X = np.random.randn(n, len(col2use))
    y = (X[:, 6] + np.random.randn(n) * 0.5 > 0).astype(int)  # number_inpatient is key

    scaler = StandardScaler()
    X_tf = scaler.fit_transform(X)

    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_tf, y)

    return model, scaler, col2use, "demo"


# ─────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────

MED_OPTIONS  = ['No', 'Steady', 'Up', 'Down']
RACE_OPTIONS = ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other', 'UNK']
GENDER_OPT   = ['Male', 'Female', 'Unknown/Invalid']
AGE_OPTIONS  = list(AGE_MAP.keys())
GLU_OPTIONS  = ['None', 'Norm', '>200', '>300']
A1C_OPTIONS  = ['None', 'Norm', '>7', '>8']
CHG_OPTIONS  = ['No', 'Ch']

ADMISSION_TYPE = {
    '1-Emergency': '1', '2-Urgent': '2', '3-Elective': '3',
    '4-Newborn': '4', '5-Not Available': '5', '6-NULL': '6',
    '7-Trauma Center': '7', '8-Not Mapped': '8'
}
DISCHARGE_DISP = {
    '1-Discharged to home': '1', '2-Short-term hospital': '2',
    '3-SNF': '3', '4-ICF': '4', '6-Home health service': '6',
    '7-AMA': '7', '8-Home IV provider': '8', '9-Admitted as inpatient': '9',
    '10-Neonate transferred': '10', '12-Still patient': '12',
    '15-Swing Bed': '15', '16-Outreach': '16', '17-Another inpatient': '17',
    '22-Rehab facility': '22', '23-Long term care': '23',
    '24-Nursing facility-Medicare': '24', '25-Not Mapped': '25',
    '27-Federal health care': '27', '28-Psychiatric hospital': '28',
    '29-Critical access hospital': '29', '30-Another institution': '30'
}
ADMISSION_SRC = {
    '1-Physician Referral': '1', '2-Clinic Referral': '2',
    '3-HMO Referral': '3', '4-Transfer from hosp': '4',
    '5-Transfer from SNF': '5', '6-Transfer from other': '6',
    '7-Emergency Room': '7', '8-Court/Law Enforcement': '8',
    '9-Not Available': '9', '10-Transfer from critical': '10',
    '11-Normal Delivery': '11', '12-Premature Delivery': '12',
    '13-Sick Baby': '13', '14-Extramural Birth': '14',
    '15-Not Available': '15', '17-NULL': '17',
    '18-Transfer another health care': '18', '19-Readmission': '19',
    '20-Not Mapped': '20', '21-Unknown': '21',
    '22-Transfer within institution': '22', '23-Born inside': '23',
    '24-Born outside': '24', '25-Transfer from ambulatory': '25',
    '26-Transfer from hospice': '26'
}

MED_SPEC_OPTIONS = TOP_10_SPEC + ['Other']
PAYER_OPTIONS = ['MC', 'MD', 'HM', 'UN', 'BC', 'SP', 'CP', 'SI', 'DM',
                 'CH', 'PO', 'WC', 'OT', 'OG', 'MP', 'FR', 'UNK']



def inject_custom_css():
    """Custom visual system for the Streamlit deployment."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp {
            background:
                radial-gradient(circle at 6% 10%, rgba(49,94,251,0.18), transparent 30%),
                radial-gradient(circle at 94% 8%, rgba(20,184,212,0.20), transparent 28%),
                linear-gradient(135deg, #f7fbff 0%, #eef5ff 48%, #f8f7ff 100%);
        }
        .block-container { padding-top: 1.5rem; max-width: 1280px; }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1e36 0%, #13294b 100%);
            border-right: 1px solid rgba(255,255,255,0.12);
        }
        section[data-testid="stSidebar"] * { color: #f8fbff !important; }
        h1, h2, h3 { color: #102033; letter-spacing: -0.03em; }
        div[data-testid="stTabs"] button { font-weight: 700; color: #344054; }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: #ffffff;
            border-radius: 999px;
            color: #214de0;
            box-shadow: 0 8px 28px rgba(49,94,251,0.12);
        }
        .stButton > button {
            border: 0;
            border-radius: 18px;
            background: linear-gradient(135deg, #315efb 0%, #14b8d4 100%);
            box-shadow: 0 16px 34px rgba(49,94,251,0.28);
            font-weight: 800;
            transition: 0.18s ease-in-out;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 20px 44px rgba(49,94,251,0.35);
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(255,255,255,0.75);
            border-radius: 22px;
            padding: 18px 20px;
            box-shadow: 0 18px 44px rgba(16,32,51,0.08);
        }
        .stSelectbox, .stSlider, .stCheckbox, .stFileUploader {
            background: rgba(255,255,255,0.62);
            border: 1px solid rgba(255,255,255,0.70);
            border-radius: 18px;
            padding: 12px 14px;
            box-shadow: 0 10px 24px rgba(16,32,51,0.045);
        }
        .hero {
            position: relative;
            overflow: hidden;
            border-radius: 34px;
            padding: 34px;
            background: linear-gradient(135deg, rgba(15,30,54,0.96), rgba(33,77,224,0.88));
            color: white;
            box-shadow: 0 28px 70px rgba(16,32,51,0.22);
            margin-bottom: 20px;
        }
        .hero:after {
            content: "";
            position: absolute;
            right: -80px;
            top: -80px;
            height: 250px;
            width: 250px;
            background: rgba(255,255,255,0.12);
            border-radius: 50%;
        }
        .hero h1 { color: white; margin: 0; font-size: 3rem; line-height: 1.02; }
        .hero p { color: rgba(255,255,255,0.82); font-size: 1.03rem; max-width: 780px; margin-top: 14px; }
        .hero-badges { display:flex; gap: 10px; flex-wrap: wrap; margin-top: 24px; }
        .badge {
            border-radius: 999px;
            padding: 9px 14px;
            background: rgba(255,255,255,0.13);
            border: 1px solid rgba(255,255,255,0.18);
            color: white;
            font-weight: 700;
            font-size: 0.86rem;
        }
        .glass-card, .batch-box {
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(255,255,255,0.60);
            border-radius: 28px;
            padding: 22px 24px;
            box-shadow: 0 18px 48px rgba(16,32,51,0.08);
            margin: 12px 0 20px;
        }
        .section-kicker {
            color: #315efb;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.76rem;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .section-title { color: #102033; font-size: 1.55rem; font-weight: 800; margin-bottom: 4px; }
        .section-copy { color: #667085; margin-bottom: 14px; }
        .pill-row { display:flex; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
        .mini-pill { border-radius: 999px; padding: 8px 12px; background: #eef4ff; color: #214de0; font-weight: 700; font-size: 0.82rem; }
        .risk-card {
            border-radius: 32px;
            padding: 28px;
            color: white;
            box-shadow: 0 24px 58px rgba(16,32,51,0.18);
            margin-top: 14px;
        }
        .risk-high { background: linear-gradient(135deg, #be123c, #f43f5e); }
        .risk-low { background: linear-gradient(135deg, #047857, #10b981); }
        .risk-card h2 { color: white; margin: 0; font-size: 2.2rem; }
        .risk-card p { color: rgba(255,255,255,0.88); margin: 8px 0 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, copy: str = ""):
    st.markdown(
        f"""
        <div class="section-kicker">{kicker}</div>
        <div class="section-title">{title}</div>
        <div class="section-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────

def main():
    inject_custom_css()

    model, scaler, col2use, source = load_artifacts()

    st.markdown(
        """
        <div class="hero">
            <div class="badge">Clinical ML Dashboard</div>
            <h1>Readmission Risk Command Center</h1>
            <p>
                A polished Streamlit interface for estimating 30-day diabetes readmission risk,
                reviewing the drivers behind a prediction, and running CSV batch scoring.
            </p>
            <div class="hero-badges">
                <span class="badge">Gradient Boosting</span>
                <span class="badge">Single-patient scoring</span>
                <span class="badge">Batch CSV workflow</span>
                <span class="badge">Deployment-ready UI</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### 🧠 Model cockpit")
        st.markdown("Use this panel as a compact status card for deployment demos.")
        st.metric("Model", "Gradient Boosting")
        st.metric("Artifact source", source.title())
        st.markdown("---")
        st.markdown("**Workflow**")
        st.markdown("1. Enter patient profile  \n2. Score readmission risk  \n3. Review top model drivers  \n4. Export batch predictions")
        st.caption("The original preprocessing and model pipeline are preserved.")

    top_a, top_b, top_c, top_d = st.columns(4)
    top_a.metric("Feature inputs", len(COLS_NUM) + len(COLS_CAT) + len(COLS_CAT_NUM) + 3)
    top_b.metric("Medication fields", 21)
    top_c.metric("Prediction target", "< 30 days")
    top_d.metric("Mode", source.title())

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    section_header(
        "Patient intake",
        "Build a complete readmission profile",
        "Move through the tabs below to capture demographics, admission details, medications, and utilization history."
    )
    st.markdown(
        """
        <div class="pill-row">
            <span class="mini-pill">01 Demographics</span>
            <span class="mini-pill">02 Medications</span>
            <span class="mini-pill">03 Labs & history</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["🧑 Profile & Admission", "💊 Medication Matrix", "🔬 Labs + Utilization"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            race   = st.selectbox("Race", RACE_OPTIONS)
            gender = st.selectbox("Gender", GENDER_OPT)
            age    = st.selectbox("Age Group", AGE_OPTIONS, index=5)
        with c2:
            admission_type_label = st.selectbox("Admission Type", list(ADMISSION_TYPE.keys()))
            discharge_label      = st.selectbox("Discharge Disposition", list(DISCHARGE_DISP.keys()))
            admission_src_label  = st.selectbox("Admission Source", list(ADMISSION_SRC.keys()), index=6)
        with c3:
            med_spec   = st.selectbox("Medical Specialty", MED_SPEC_OPTIONS)
            payer_code = st.selectbox("Payer Code", PAYER_OPTIONS, index=16)
            has_weight = st.checkbox("Weight recorded?", value=False)

        time_in_hospital = st.slider("Length of stay", 1, 14, 4, help="Time in hospital, measured in days.")

    with tab2:
        section_header("Medication signal", "Medication changes", "Select the status for each diabetes-related medication.")
        mc1, mc2, mc3 = st.columns(3)
        meds = {}
        med_names = [
            ('metformin','Metformin'), ('repaglinide','Repaglinide'),
            ('nateglinide','Nateglinide'), ('chlorpropamide','Chlorpropamide'),
            ('glimepiride','Glimepiride'), ('acetohexamide','Acetohexamide'),
            ('glipizide','Glipizide'), ('glyburide','Glyburide'),
            ('tolbutamide','Tolbutamide'), ('pioglitazone','Pioglitazone'),
            ('rosiglitazone','Rosiglitazone'), ('acarbose','Acarbose'),
            ('miglitol','Miglitol'), ('troglitazone','Troglitazone'),
            ('tolazamide','Tolazamide'), ('insulin','Insulin'),
            ('glyburide-metformin','Glyb-Metformin'),
            ('glipizide-metformin','Glip-Metformin'),
            ('glimepiride-pioglitazone','Glim-Pioglitazone'),
            ('metformin-rosiglitazone','Metf-Rosiglitazone'),
            ('metformin-pioglitazone','Metf-Pioglitazone'),
        ]
        cols_cycle = [mc1, mc2, mc3]
        for i, (key, label) in enumerate(med_names):
            meds[key] = cols_cycle[i % 3].selectbox(label, MED_OPTIONS, key=key)

        st.divider()
        c1, c2, c3 = st.columns(3)
        change      = c1.selectbox("Overall Medication Change", CHG_OPTIONS)
        diabetesMed = c2.selectbox("Diabetes Medication Prescribed", ['No', 'Yes'])
        num_medications = c3.slider("Medication count", 1, 81, 15)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            num_lab_procedures = st.slider("Lab procedures", 1, 132, 44)
            num_procedures     = st.slider("Non-lab procedures", 0, 6, 1)
            number_diagnoses   = st.slider("Diagnoses", 1, 16, 8)
        with c2:
            number_outpatient  = st.slider("Outpatient visits, past year", 0, 42, 0)
            number_emergency   = st.slider("Emergency visits, past year", 0, 76, 0)
            number_inpatient   = st.slider("Inpatient visits, past year", 0, 21, 0)

        c1, c2 = st.columns(2)
        max_glu_serum = c1.selectbox("Max glucose serum test", GLU_OPTIONS)
        A1Cresult     = c2.selectbox("A1C result", A1C_OPTIONS)

    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("Generate readmission risk profile", type="primary", use_container_width=True)

    if predict_btn:
        inputs = {
            'race': race, 'gender': gender, 'age': age,
            'time_in_hospital': time_in_hospital,
            'num_lab_procedures': num_lab_procedures,
            'num_procedures': num_procedures,
            'num_medications': num_medications,
            'number_outpatient': number_outpatient,
            'number_emergency': number_emergency,
            'number_inpatient': number_inpatient,
            'number_diagnoses': number_diagnoses,
            'max_glu_serum': max_glu_serum,
            'A1Cresult': A1Cresult,
            'change': change,
            'diabetesMed': diabetesMed,
            'payer_code': payer_code,
            'medical_specialty': med_spec,
            'admission_type_id': ADMISSION_TYPE[admission_type_label],
            'discharge_disposition_id': DISCHARGE_DISP[discharge_label],
            'admission_source_id': ADMISSION_SRC[admission_src_label],
            'weight': '[75-100)' if has_weight else None,
            **meds
        }

        try:
            X = preprocess_row(inputs, col2use)
            X_tf = scaler.transform(X.values)
            prob = model.predict_proba(X_tf)[0][1]
            pred = int(prob >= 0.5)
            confidence = max(prob, 1 - prob)
            risk_label = "HIGH RISK" if pred == 1 else "LOW RISK"
            risk_class = "risk-high" if pred == 1 else "risk-low"
            recommendation = (
                "Prioritize discharge support, medication reconciliation, and rapid follow-up planning."
                if pred == 1 else
                "Continue standard discharge planning and routine follow-up monitoring."
            )

            st.markdown(
                f"""
                <div class="risk-card {risk_class}">
                    <div class="badge">Prediction result</div>
                    <h2>{risk_label}</h2>
                    <p>Estimated 30-day readmission probability: <strong>{prob:.1%}</strong></p>
                    <p>{recommendation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Risk level", risk_label)
            col_b.metric("Readmission probability", f"{prob:.1%}")
            col_c.metric("Model confidence", f"{confidence:.1%}")
            st.progress(float(prob), text=f"Readmission probability: {prob:.1%}")

            with st.expander("🔑 Top model drivers", expanded=True):
                importances = pd.Series(model.feature_importances_, index=col2use).sort_values(ascending=False).head(10)
                st.bar_chart(importances)
                st.caption("Feature importances are calculated from the fitted Gradient Boosting model.")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)

    st.markdown('<div class="batch-box">', unsafe_allow_html=True)
    section_header(
        "Batch scoring",
        "Upload a CSV and export predictions",
        "Use a file with the same structure as diabetic_data.csv to generate readmission probabilities for many patients."
    )
    uploaded = st.file_uploader("Upload diabetic_data CSV", type=["csv"])

    if uploaded:
        df_up = pd.read_csv(uploaded)
        c1, c2 = st.columns(2)
        c1.metric("Rows loaded", f"{len(df_up):,}")
        c2.metric("Columns detected", f"{len(df_up.columns):,}")

        with st.spinner("Preprocessing and predicting …"):
            try:
                df_up = df_up.replace('?', np.nan)
                df_up = df_up.loc[~df_up.discharge_disposition_id.isin([11, 13, 14, 19, 20, 21])]
                for c in ['race', 'payer_code', 'medical_specialty']:
                    if c in df_up.columns:
                        df_up[c] = df_up[c].fillna('UNK')
                df_up['med_spec'] = df_up.get('medical_specialty', 'UNK').copy() if 'medical_specialty' in df_up.columns else 'UNK'
                df_up.loc[~df_up.med_spec.isin(TOP_10_SPEC), 'med_spec'] = 'Other'
                for c in COLS_CAT_NUM:
                    if c in df_up.columns:
                        df_up[c] = df_up[c].astype('str')
                df_cat = pd.get_dummies(df_up[[c for c in COLS_CAT + COLS_CAT_NUM + ['med_spec'] if c in df_up.columns]], drop_first=True)
                df_up['age_group']  = df_up.age.replace(AGE_MAP) if 'age' in df_up.columns else 50
                df_up['has_weight'] = df_up.weight.notnull().astype('int') if 'weight' in df_up.columns else 0
                df_full = pd.concat([df_up[[c for c in COLS_NUM + ['age_group','has_weight'] if c in df_up.columns]].reset_index(drop=True),
                                     df_cat.reset_index(drop=True)], axis=1)
                for c in col2use:
                    if c not in df_full.columns:
                        df_full[c] = 0
                X_batch = df_full[col2use].fillna(0).values
                X_batch_tf = scaler.transform(X_batch)
                probs = model.predict_proba(X_batch_tf)[:, 1]
                df_up['readmission_probability'] = probs
                df_up['predicted_readmission']   = (probs >= 0.5).astype(int)
                st.success(f"Predicted {(probs >= 0.5).sum():,} / {len(probs):,} patients as high risk.")
                st.dataframe(
                    df_up[['patient_nbr', 'readmission_probability', 'predicted_readmission']].head(100)
                    if 'patient_nbr' in df_up.columns
                    else df_up[['readmission_probability', 'predicted_readmission']].head(100),
                    use_container_width=True
                )
                csv_out = df_up.to_csv(index=False).encode()
                st.download_button("Download full results CSV", csv_out, "readmission_predictions.csv", "text/csv", use_container_width=True)
            except Exception as e:
                st.error(f"Batch prediction error: {e}")
                st.exception(e)
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

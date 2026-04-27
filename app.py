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


# ─────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────

def main():
    st.title("🏥 Diabetes Patient Readmission Predictor")
    st.markdown(
        "Predict whether a diabetic patient will be **readmitted within 30 days** "
        "using a Gradient Boosting model (best performer from multi-model evaluation)."
    )

    model, scaler, col2use, source = load_artifacts()

    with st.sidebar:
        st.header("ℹ️ Model Info")
        st.success(f"**Model:** Optimized Gradient Boosting Classifier")
        st.info(f"**Source:** {source}")
        st.caption(
            "Best model selected after comparing KNN, Logistic Regression, SGD, "
            "Naïve Bayes, Decision Tree, Random Forest, and Gradient Boosting with "
            "hyperparameter tuning (AUC metric)."
        )

    # ── Input form ────────────────────────────────────────────────────────
    st.subheader("Patient Information")

    tab1, tab2, tab3 = st.tabs(["🧑 Demographics & Admission", "💊 Medications", "🔬 Lab & History"])

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

        time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 4)

    with tab2:
        st.markdown("**Medication Changes** (No / Steady / Up / Down)")
        mc1, mc2, mc3, mc4 = st.columns(4)
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
        cols_cycle = [mc1, mc2, mc3, mc4]
        for i, (key, label) in enumerate(med_names):
            meds[key] = cols_cycle[i % 4].selectbox(label, MED_OPTIONS, key=key)

        st.divider()
        c1, c2, c3 = st.columns(3)
        change     = c1.selectbox("Overall Medication Change", CHG_OPTIONS)
        diabetesMed = c2.selectbox("Diabetes Medication Prescribed", ['No', 'Yes'])
        num_medications = c3.slider("# Medications", 1, 81, 15)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            num_lab_procedures = st.slider("# Lab Procedures", 1, 132, 44)
            num_procedures     = st.slider("# Procedures (non-lab)", 0, 6, 1)
            number_diagnoses   = st.slider("# Diagnoses", 1, 16, 8)
        with c2:
            number_outpatient  = st.slider("# Outpatient Visits (past year)", 0, 42, 0)
            number_emergency   = st.slider("# Emergency Visits (past year)", 0, 76, 0)
            number_inpatient   = st.slider("# Inpatient Visits (past year)", 0, 21, 0)

        c1, c2 = st.columns(2)
        max_glu_serum = c1.selectbox("Max Glucose Serum Test", GLU_OPTIONS)
        A1Cresult     = c2.selectbox("A1C Result", A1C_OPTIONS)

    # ── Predict ───────────────────────────────────────────────────────────
    st.divider()
    predict_btn = st.button("🔍 Predict Readmission Risk", type="primary", use_container_width=True)

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

            st.subheader("📊 Prediction Result")
            col_a, col_b, col_c = st.columns(3)

            risk_label = "HIGH RISK" if pred == 1 else "LOW RISK"
            risk_color = "🔴" if pred == 1 else "🟢"
            col_a.metric("Readmission Risk", f"{risk_color} {risk_label}")
            col_b.metric("Probability of Readmission", f"{prob:.1%}")
            col_c.metric("Confidence", f"{max(prob, 1-prob):.1%}")

            # Risk bar
            st.progress(float(prob), text=f"Readmission probability: {prob:.1%}")

            if pred == 1:
                st.warning(
                    "⚠️ This patient is predicted to be **readmitted within 30 days**. "
                    "Consider follow-up care planning and discharge support services."
                )
            else:
                st.success(
                    "✅ This patient is predicted **not** to be readmitted within 30 days."
                )

            # Key risk factors display
            with st.expander("🔑 Key Risk Factors (model feature importances)"):
                importances = pd.Series(
                    model.feature_importances_, index=col2use
                ).sort_values(ascending=False).head(10)
                st.bar_chart(importances)
                st.caption(
                    "Top features by importance. `number_inpatient` (prior hospital visits) "
                    "is typically the strongest predictor."
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)

    # ── Upload CSV for batch prediction ───────────────────────────────────
    st.divider()
    st.subheader("📂 Batch Prediction (CSV Upload)")
    st.markdown(
        "Upload a CSV file in the same format as `diabetic_data.csv` "
        "to get readmission predictions for multiple patients at once."
    )
    uploaded = st.file_uploader("Upload diabetic_data CSV", type=["csv"])

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df_up):,} rows × {len(df_up.columns)} columns")

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
                st.success(f"Done! Predicted {(probs >= 0.5).sum():,} / {len(probs):,} patients at high risk.")
                st.dataframe(df_up[['patient_nbr', 'readmission_probability', 'predicted_readmission']].head(100) if 'patient_nbr' in df_up.columns else df_up[['readmission_probability', 'predicted_readmission']].head(100))
                csv_out = df_up.to_csv(index=False).encode()
                st.download_button("⬇️ Download Full Results CSV", csv_out, "readmission_predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Batch prediction error: {e}")
                st.exception(e)


if __name__ == "__main__":
    main()

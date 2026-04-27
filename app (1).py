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
    page_title="DiabetiQ · Readmission Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — Clinical Dark Precision Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:     #080d14;
    --bg-secondary:   #0e1620;
    --bg-card:        #111c2a;
    --bg-card-hover:  #152235;
    --bg-input:       #0a1220;
    --border-subtle:  rgba(0,200,160,0.10);
    --border-medium:  rgba(0,200,160,0.22);
    --border-bright:  rgba(0,200,160,0.50);
    --accent:         #00c8a0;
    --accent-dim:     rgba(0,200,160,0.15);
    --accent-glow:    rgba(0,200,160,0.25);
    --danger:         #ff4d6d;
    --danger-dim:     rgba(255,77,109,0.15);
    --warning:        #f9a825;
    --warning-dim:    rgba(249,168,37,0.12);
    --text-primary:   #e8f0f7;
    --text-secondary: #7fa3be;
    --text-muted:     #4a6880;
    --text-mono:      #00c8a0;
    --font-display:   'Syne', sans-serif;
    --font-body:      'DM Sans', sans-serif;
    --font-mono:      'DM Mono', monospace;
    --radius-sm:      6px;
    --radius-md:      10px;
    --radius-lg:      16px;
}

/* ── Global reset ── */
html, body, .stApp { background-color: var(--bg-primary) !important; }
*, *::before, *::after { box-sizing: border-box; }
body { font-family: var(--font-body); color: var(--text-primary); }

/* ── Remove streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem !important; }
[data-testid="stSidebarNav"] { display: none; }

/* ── Typography ── */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: var(--font-display) !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}

/* ── App header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    padding: 2rem 0 1.6rem;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 2rem;
}
.app-logo {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #00c8a0, #0087c8);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 24px rgba(0,200,160,0.3);
    flex-shrink: 0;
}
.app-title-group { flex: 1; }
.app-name {
    font-family: var(--font-display);
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: var(--text-primary);
    line-height: 1;
    margin-bottom: 0.25rem;
}
.app-name span { color: var(--accent); }
.app-subtitle {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.model-badge {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    padding: 4px 12px;
    background: var(--accent-dim);
    border: 1px solid var(--border-medium);
    border-radius: 999px;
    color: var(--accent);
    letter-spacing: 0.05em;
}

/* ── Section labels ── */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::before {
    content: '';
    display: inline-block;
    width: 18px; height: 1px;
    background: var(--accent);
}

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.card:hover {
    border-color: var(--border-medium);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}

/* ── Stat cells ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.stat-cell {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1.1rem 1.2rem;
    position: relative;
    overflow: hidden;
}
.stat-cell::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
.stat-label {
    font-family: var(--font-mono);
    font-size: 0.63rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}
.stat-value {
    font-family: var(--font-display);
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.1;
}
.stat-value.accent { color: var(--accent); }
.stat-value.danger { color: var(--danger); }

/* ── Risk result panel ── */
.result-panel {
    background: var(--bg-card);
    border-radius: var(--radius-lg);
    overflow: hidden;
    margin: 1.5rem 0;
    border: 1px solid var(--border-medium);
}
.result-header {
    padding: 1.2rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border-subtle);
}
.result-header.high-risk {
    background: var(--danger-dim);
    border-bottom-color: rgba(255,77,109,0.2);
}
.result-header.low-risk {
    background: rgba(0,200,160,0.08);
    border-bottom-color: var(--border-medium);
}
.risk-label-text {
    font-family: var(--font-display);
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.risk-label-text.high { color: var(--danger); }
.risk-label-text.low  { color: var(--accent); }
.risk-indicator {
    width: 10px; height: 10px;
    border-radius: 50%;
    animation: pulse 1.8s ease-in-out infinite;
}
.risk-indicator.high { background: var(--danger); box-shadow: 0 0 12px var(--danger); }
.risk-indicator.low  { background: var(--accent); box-shadow: 0 0 12px var(--accent); }
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.6; transform: scale(1.3); }
}
.result-body { padding: 1.5rem; }

/* ── Risk meter ── */
.risk-meter-wrap { margin: 1rem 0; }
.risk-meter-track {
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    overflow: hidden;
    position: relative;
}
.risk-meter-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.risk-meter-fill.high { background: linear-gradient(90deg, #ff4d6d, #ff8fa3); }
.risk-meter-fill.low  { background: linear-gradient(90deg, #00c8a0, #00e6b8); }
.risk-meter-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 4px;
    font-family: var(--font-mono);
    font-size: 0.58rem;
    color: var(--text-muted);
}

/* ── Streamlit overrides ── */
/* Tabs */
[data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: var(--radius-md) var(--radius-md) 0 0 !important;
    border: 1px solid var(--border-subtle) !important;
    border-bottom: none !important;
    padding: 4px 6px 0 !important;
    gap: 2px !important;
}
[data-baseweb="tab"] {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em !important;
    color: var(--text-muted) !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    padding: 8px 16px !important;
    border: none !important;
    background: transparent !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--accent) !important;
    background: var(--bg-card) !important;
    border-bottom: 2px solid var(--accent) !important;
}
[data-baseweb="tab-panel"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
    padding: 1.5rem !important;
}

/* Inputs */
.stSelectbox > div > div,
.stNumberInput > div > div,
.stTextInput > div > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    transition: border-color 0.15s !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within,
.stTextInput > div > div:focus-within {
    border-color: var(--border-bright) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* Labels */
.stSelectbox label, .stSlider label,
.stNumberInput label, .stCheckbox label,
.stRadio label, [data-testid="stWidgetLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div {
    background: var(--border-medium) !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent) !important;
}
.stSlider [data-testid="stTickBar"] { color: var(--text-muted) !important; }

/* Checkbox */
.stCheckbox input[type="checkbox"]:checked + div {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
}

/* Divider */
hr { border-color: var(--border-subtle) !important; margin: 2rem 0 !important; }

/* Primary button */
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 0.75rem 1.5rem !important;
    box-shadow: 0 4px 20px rgba(0,200,160,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    background: #00e6b8 !important;
    box-shadow: 0 6px 28px rgba(0,200,160,0.45) !important;
    transform: translateY(-1px) !important;
}

/* Secondary button */
.stButton > button[kind="secondary"] {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: var(--radius-sm) !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    font-family: var(--font-body) !important;
}
.stSuccess {
    background: rgba(0,200,160,0.08) !important;
    border: 1px solid var(--border-medium) !important;
    color: var(--accent) !important;
}
.stWarning {
    background: var(--warning-dim) !important;
    border: 1px solid rgba(249,168,37,0.25) !important;
    color: var(--warning) !important;
}
.stError {
    background: var(--danger-dim) !important;
    border: 1px solid rgba(255,77,109,0.25) !important;
    color: var(--danger) !important;
}

/* Info boxes */
.stInfo {
    background: rgba(0,135,200,0.08) !important;
    border: 1px solid rgba(0,135,200,0.25) !important;
    color: #5ac8fa !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.06em !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 1.4rem !important;
    color: var(--text-primary) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--bg-input) !important;
    border: 2px dashed var(--border-medium) !important;
    border-radius: var(--radius-md) !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-medium); border-radius: 99px; }

/* Progress bar */
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, var(--accent), #00e6b8) !important;
    border-radius: 99px !important;
}
[data-testid="stProgress"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 99px !important;
}

/* Sidebar elements */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: var(--font-display) !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    font-size: 0.8rem !important;
    color: var(--text-secondary) !important;
    line-height: 1.6 !important;
}
[data-testid="stSidebar"] .stMarkdown code {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    background: var(--accent-dim) !important;
    color: var(--accent) !important;
    padding: 1px 5px !important;
    border-radius: 3px !important;
}

/* Status pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    padding: 3px 10px;
    border-radius: 99px;
    text-transform: uppercase;
}
.status-pill.online {
    background: rgba(0,200,160,0.12);
    border: 1px solid rgba(0,200,160,0.3);
    color: var(--accent);
}
.status-pill.online::before {
    content: '●';
    font-size: 0.5rem;
}

/* Sidebar section header */
.sidebar-section {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    border-top: 1px solid var(--border-subtle);
    padding-top: 1rem;
    margin-top: 1rem;
    margin-bottom: 0.75rem;
}

/* Sidebar metric row */
.sb-metric {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 4px 0;
    border-bottom: 1px solid var(--border-subtle);
}
.sb-metric-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
}
.sb-metric-value {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--accent);
    font-weight: 500;
}

/* Divider accent */
.accent-divider {
    height: 1px;
    background: linear-gradient(90deg, var(--accent), transparent 60%);
    margin: 2rem 0;
    border: none;
}

/* Medication grid header */
.med-grid-header {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 1rem;
    display: flex;
    gap: 0.5rem;
    align-items: center;
}

/* Subheader overrides */
.stMarkdown h3 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    margin-bottom: 0.5rem !important;
}

/* Caption */
.stMarkdown p em, .stCaption {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    font-family: var(--font-mono) !important;
}

/* Bar chart */
[data-testid="stVegaLiteChart"] { border-radius: var(--radius-md) !important; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

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
# Preprocessing
# ─────────────────────────────────────────────
def preprocess_row(inputs: dict, col2use: list) -> pd.DataFrame:
    row = inputs.copy()
    for k in ['race', 'payer_code', 'medical_specialty']:
        if not row.get(k) or row[k] == '?':
            row[k] = 'UNK'
    row['med_spec'] = row['medical_specialty'] if row['medical_specialty'] in TOP_10_SPEC else 'Other'
    row['age_group'] = AGE_MAP.get(row.get('age', '[50-60)'), 50)
    row['has_weight'] = 1 if row.get('weight') and row['weight'] != '?' else 0
    for c in COLS_CAT_NUM:
        row[c] = str(row.get(c, '1'))
    df = pd.DataFrame([row])
    all_cat_cols = COLS_CAT + COLS_CAT_NUM + ['med_spec']
    df_cat = pd.get_dummies(df[all_cat_cols], drop_first=True)
    df_num = df[COLS_NUM + ['age_group', 'has_weight']].copy()
    df_full = pd.concat([df_num.reset_index(drop=True), df_cat.reset_index(drop=True)], axis=1)
    for c in col2use:
        if c not in df_full.columns:
            df_full[c] = 0
    df_full = df_full[col2use]
    return df_full


# ─────────────────────────────────────────────
# Model loading / training
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing model engine …")
def load_artifacts():
    scaler_path = "pickle-files/scaler.sav"
    model_path  = "pickle-files/best_classifier.pkl"
    cols_path   = "pickle-files/col2use.pkl"

    if os.path.exists(scaler_path) and os.path.exists(model_path) and os.path.exists(cols_path):
        scaler  = pickle.load(open(scaler_path, 'rb'))
        model   = pickle.load(open(model_path, 'rb'))
        col2use = pickle.load(open(cols_path, 'rb'))
        return model, scaler, col2use, "pre-trained"

    csv_candidates = ["diabetic_data.csv", "data/diabetic_data.csv"]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)
    if csv_path:
        return _train_from_csv(csv_path)

    return _demo_model()


def _train_from_csv(csv_path):
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
    rows_pos = df_train_all.readmission_status == 1
    df_train = pd.concat([
        df_train_all.loc[rows_pos],
        df_train_all.loc[~rows_pos].sample(n=rows_pos.sum(), random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    X_train = df_train[col2use].values
    y_train = df_train['readmission_status'].values
    X_all   = df_train_all[col2use].values
    scaler  = StandardScaler()
    scaler.fit(X_all)
    X_train_tf = scaler.transform(X_train)
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    random_grid = {
        'n_estimators': range(100, 300, 100),
        'max_depth': range(2, 5, 1),
        'learning_rate': [0.01, 0.1]
    }
    auc_score = make_scorer(roc_auc_score)
    gbc_random = RandomizedSearchCV(gbc, random_grid, n_iter=6, cv=2,
                                    scoring=auc_score, random_state=42, verbose=0)
    gbc_random.fit(X_train_tf, y_train)
    best_model = gbc_random.best_estimator_
    os.makedirs("pickle-files", exist_ok=True)
    pickle.dump(scaler,     open("pickle-files/scaler.sav",         'wb'))
    pickle.dump(best_model, open("pickle-files/best_classifier.pkl", 'wb'), protocol=4)
    pickle.dump(col2use,    open("pickle-files/col2use.pkl",         'wb'))
    return best_model, scaler, col2use, "trained on upload"


def _demo_model():
    np.random.seed(42)
    n = 2000
    col2use = COLS_NUM + ['age_group', 'has_weight']
    X = np.random.randn(n, len(col2use))
    y = (X[:, 6] + np.random.randn(n) * 0.5 > 0).astype(int)
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
    # ── App header ──────────────────────────────
    st.markdown("""
    <div class="app-header">
        <div class="app-logo">⬡</div>
        <div class="app-title-group">
            <div class="app-name">Diabeti<span>Q</span></div>
            <div class="app-subtitle">30-day readmission intelligence · Gradient Boosting Classifier</div>
        </div>
        <span class="model-badge">◉ MODEL ACTIVE</span>
    </div>
    """, unsafe_allow_html=True)

    model, scaler, col2use, source = load_artifacts()

    # ── Sidebar ──────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.5rem 0 1rem;">
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:800; letter-spacing:-0.02em; color:#e8f0f7; margin-bottom:0.3rem;">
                System Status
            </div>
            <span class="status-pill online">Engine Online</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sidebar-section">Model Details</div>
        <div class="sb-metric">
            <span class="sb-metric-label">Algorithm</span>
            <span class="sb-metric-value">GBC</span>
        </div>
        <div class="sb-metric">
            <span class="sb-metric-label">Source</span>
            <span class="sb-metric-value">{source}</span>
        </div>
        <div class="sb-metric">
            <span class="sb-metric-label">Features</span>
            <span class="sb-metric-value">{len(col2use)}</span>
        </div>
        <div class="sb-metric">
            <span class="sb-metric-label">Threshold</span>
            <span class="sb-metric-value">p ≥ 0.50</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-section">Evaluated Models</div>
        """, unsafe_allow_html=True)

        models_compared = [
            ("GBC", "★ Best"),("Random Forest", ""),("Decision Tree", ""),
            ("Logistic Reg", ""),("KNN", ""),("SGD", ""),("Naïve Bayes", "")
        ]
        for m, note in models_compared:
            color = "#00c8a0" if note else "#4a6880"
            badge = f' <span style="color:#00c8a0;font-size:0.55rem">{note}</span>' if note else ""
            st.markdown(f"""
            <div class="sb-metric">
                <span class="sb-metric-label">{m}</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:{color}">{note if note else "—"}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-section">About</div>
        <p style="font-family:'DM Sans',sans-serif;font-size:0.75rem;color:#7fa3be;line-height:1.6;margin-top:0.5rem">
        Predicts 30-day hospital readmission for diabetic patients using clinical and pharmacological data. 
        Best model selected via AUC on held-out validation set.
        </p>
        <p style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#4a6880;margin-top:0.75rem">
        ⚠ For research use only. Not a clinical decision tool.
        </p>
        """, unsafe_allow_html=True)

    # ── Input section header ─────────────────────
    st.markdown('<div class="section-label">Patient Data Entry</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "  ⬡  Demographics & Admission  ",
        "  ⬡  Medications  ",
        "  ⬡  Labs & History  "
    ])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="section-label" style="margin-bottom:0.5rem">Patient Identity</div>', unsafe_allow_html=True)
            race   = st.selectbox("Race", RACE_OPTIONS)
            gender = st.selectbox("Gender", GENDER_OPT)
            age    = st.selectbox("Age Group", AGE_OPTIONS, index=5)
        with c2:
            st.markdown('<div class="section-label" style="margin-bottom:0.5rem">Admission Details</div>', unsafe_allow_html=True)
            admission_type_label = st.selectbox("Admission Type", list(ADMISSION_TYPE.keys()))
            discharge_label      = st.selectbox("Discharge Disposition", list(DISCHARGE_DISP.keys()))
            admission_src_label  = st.selectbox("Admission Source", list(ADMISSION_SRC.keys()), index=6)
        with c3:
            st.markdown('<div class="section-label" style="margin-bottom:0.5rem">Clinical Context</div>', unsafe_allow_html=True)
            med_spec   = st.selectbox("Medical Specialty", MED_SPEC_OPTIONS)
            payer_code = st.selectbox("Payer Code", PAYER_OPTIONS, index=16)
            has_weight = st.checkbox("Weight recorded in chart?", value=False)

        st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)
        time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 4,
                                      help="Number of days admitted")

    with tab2:
        st.markdown("""
        <div class="med-grid-header">
            ⬡ Medication Status per drug — No / Steady / Up / Down
        </div>
        """, unsafe_allow_html=True)

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
            ('glyburide-metformin','Glyb+Metformin'),
            ('glipizide-metformin','Glip+Metformin'),
            ('glimepiride-pioglitazone','Glim+Pioglitazone'),
            ('metformin-rosiglitazone','Metf+Rosiglitazone'),
            ('metformin-pioglitazone','Metf+Pioglitazone'),
        ]
        cols_cycle = [mc1, mc2, mc3, mc4]
        for i, (key, label) in enumerate(med_names):
            meds[key] = cols_cycle[i % 4].selectbox(label, MED_OPTIONS, key=key)

        st.markdown('<div class="accent-divider"></div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        change       = c1.selectbox("Overall Med Change", CHG_OPTIONS)
        diabetesMed  = c2.selectbox("Diabetes Med Prescribed", ['No', 'Yes'])
        num_medications = c3.slider("Total # Medications", 1, 81, 15)

    with tab3:
        st.markdown('<div class="section-label" style="margin-bottom:0.5rem">Procedural Data</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            num_lab_procedures = st.slider("# Lab Procedures", 1, 132, 44)
            num_procedures     = st.slider("# Non-lab Procedures", 0, 6, 1)
            number_diagnoses   = st.slider("# Diagnoses", 1, 16, 8)
        with c2:
            number_outpatient  = st.slider("# Outpatient Visits (prior year)", 0, 42, 0)
            number_emergency   = st.slider("# Emergency Visits (prior year)", 0, 76, 0)
            number_inpatient   = st.slider("# Inpatient Visits (prior year)", 0, 21, 0)

        st.markdown('<div class="section-label" style="margin-bottom:0.5rem; margin-top:1rem">Lab Results</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        max_glu_serum = c1.selectbox("Max Glucose Serum", GLU_OPTIONS)
        A1Cresult     = c2.selectbox("HbA1c Result", A1C_OPTIONS)

    # ── Predict button ───────────────────────────
    st.markdown('<div class="accent-divider"></div>', unsafe_allow_html=True)

    col_btn, col_note = st.columns([2, 3])
    with col_btn:
        predict_btn = st.button("⬡  Run Readmission Analysis", type="primary", use_container_width=True)
    with col_note:
        st.markdown("""
        <p style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#4a6880;margin-top:0.75rem;line-height:1.5">
        ⚠ Clinical research tool only. Results do not constitute medical advice.<br>
        Verify all inputs before running prediction.
        </p>
        """, unsafe_allow_html=True)

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
            X     = preprocess_row(inputs, col2use)
            X_tf  = scaler.transform(X.values)
            prob  = model.predict_proba(X_tf)[0][1]
            pred  = int(prob >= 0.5)
            conf  = max(prob, 1 - prob)

            risk_class = "high-risk" if pred == 1 else "low-risk"
            risk_word  = "HIGH RISK" if pred == 1 else "LOW RISK"
            risk_css   = "high" if pred == 1 else "low"
            bar_pct    = f"{prob*100:.1f}%"

            st.markdown('<div class="section-label" style="margin-top:2rem">Analysis Result</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-panel">
                <div class="result-header {risk_class}">
                    <div style="display:flex;align-items:center;gap:10px">
                        <div class="risk-indicator {risk_css}"></div>
                        <span class="risk-label-text {risk_css}">{risk_word}</span>
                        <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#7fa3be;margin-left:4px">
                            — 30-day readmission prediction
                        </span>
                    </div>
                    <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#7fa3be">
                        CONF {conf:.1%}
                    </span>
                </div>
                <div class="result-body">
                    <div class="risk-meter-wrap">
                        <div class="risk-meter-track">
                            <div class="risk-meter-fill {risk_css}" style="width:{bar_pct}"></div>
                        </div>
                        <div class="risk-meter-labels">
                            <span>0%</span>
                            <span>Readmission probability: <strong style="color:{'#ff4d6d' if pred else '#00c8a0'}">{prob:.1%}</strong></span>
                            <span>100%</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Metric row
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Readmission Risk", risk_word)
            col_b.metric("Probability", f"{prob:.1%}")
            col_c.metric("Confidence", f"{conf:.1%}")

            if pred == 1:
                st.warning(
                    "⚠ This patient is predicted to be readmitted within 30 days. "
                    "Consider enhanced follow-up care, discharge support, and care coordination."
                )
            else:
                st.success(
                    "✓ This patient is predicted not to be readmitted within 30 days."
                )

            with st.expander("⬡  Feature Importance — Top 10 Model Drivers"):
                importances = pd.Series(
                    model.feature_importances_, index=col2use
                ).sort_values(ascending=False).head(10)
                st.bar_chart(importances)
                st.caption(
                    "Feature importances from GBC model. `number_inpatient` (prior inpatient visits) "
                    "is typically the strongest predictor of readmission."
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)

    # ── Batch prediction ─────────────────────────
    st.markdown('<div class="accent-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Batch Processing</div>', unsafe_allow_html=True)

    st.markdown("""
    <p style="font-family:'DM Sans',sans-serif;font-size:0.82rem;color:#7fa3be;margin-bottom:1rem;line-height:1.6">
    Upload a CSV in the format of <code style="font-family:'DM Mono',monospace;background:rgba(0,200,160,0.1);color:#00c8a0;padding:1px 5px;border-radius:3px">diabetic_data.csv</code>
    to run readmission predictions across an entire patient cohort.
    </p>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop CSV file here", type=["csv"])

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.markdown(f"""
        <div class="sb-metric" style="margin-bottom:1rem">
            <span class="sb-metric-label">Rows loaded</span>
            <span class="sb-metric-value">{len(df_up):,}</span>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Running batch inference …"):
            try:
                df_up = df_up.replace('?', np.nan)
                df_up = df_up.loc[~df_up.discharge_disposition_id.isin([11, 13, 14, 19, 20, 21])]
                for c in ['race', 'payer_code', 'medical_specialty']:
                    if c in df_up.columns:
                        df_up[c] = df_up[c].fillna('UNK')
                df_up['med_spec'] = (df_up['medical_specialty'].copy()
                                     if 'medical_specialty' in df_up.columns else 'UNK')
                df_up.loc[~df_up.med_spec.isin(TOP_10_SPEC), 'med_spec'] = 'Other'
                for c in COLS_CAT_NUM:
                    if c in df_up.columns:
                        df_up[c] = df_up[c].astype('str')
                df_cat = pd.get_dummies(
                    df_up[[c for c in COLS_CAT + COLS_CAT_NUM + ['med_spec'] if c in df_up.columns]],
                    drop_first=True
                )
                df_up['age_group']  = df_up.age.replace(AGE_MAP) if 'age' in df_up.columns else 50
                df_up['has_weight'] = df_up.weight.notnull().astype('int') if 'weight' in df_up.columns else 0
                df_full = pd.concat([
                    df_up[[c for c in COLS_NUM + ['age_group', 'has_weight'] if c in df_up.columns]].reset_index(drop=True),
                    df_cat.reset_index(drop=True)
                ], axis=1)
                for c in col2use:
                    if c not in df_full.columns:
                        df_full[c] = 0
                X_batch    = df_full[col2use].fillna(0).values
                X_batch_tf = scaler.transform(X_batch)
                probs      = model.predict_proba(X_batch_tf)[:, 1]
                df_up['readmission_probability'] = probs
                df_up['predicted_readmission']   = (probs >= 0.5).astype(int)

                n_high = int((probs >= 0.5).sum())
                n_total = len(probs)

                st.success(f"✓ Inference complete — {n_high:,} / {n_total:,} patients flagged high-risk ({n_high/n_total:.1%})")

                show_cols = (['patient_nbr'] if 'patient_nbr' in df_up.columns else []) + \
                            ['readmission_probability', 'predicted_readmission']
                st.dataframe(df_up[show_cols].head(100), use_container_width=True)

                csv_out = df_up.to_csv(index=False).encode()
                st.download_button(
                    "⬇  Download Full Results CSV",
                    csv_out,
                    "readmission_predictions.csv",
                    "text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Batch prediction error: {e}")
                st.exception(e)


if __name__ == "__main__":
    main()

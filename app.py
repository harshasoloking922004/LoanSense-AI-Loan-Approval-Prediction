
import streamlit as st
import pandas as pd
import sqlite3
import joblib
import hashlib
import secrets
import re
import os
import plotly.express as px
from datetime import datetime

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LoanSense AI | Loan Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_PATH = "artifacts/loansense.db"
MODEL_PATH = "artifacts/loan_model.pkl"

os.makedirs("artifacts", exist_ok=True)

# =========================================================
# 2. DATABASE
# =========================================================

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database():

    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                applicant_income REAL,
                coapplicant_income REAL,
                loan_amount REAL,
                loan_term INTEGER,
                credit_history REAL,
                prediction TEXT NOT NULL,
                approval_probability REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)


initialize_database()

# =========================================================
# 3. PASSWORD HASHING AND AUTHENTICATION
# =========================================================

def hash_password(password, salt):

    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        300000
    ).hex()


def register_user(name, email, password):

    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)

    try:

        with get_connection() as conn:

            conn.execute("""
                INSERT INTO users
                (name, email, salt, password_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                name.strip(),
                email.strip().lower(),
                salt,
                password_hash,
                datetime.now().isoformat(timespec="seconds")
            ))

        return True, "Account created successfully!"

    except sqlite3.IntegrityError:

        return False, "This email is already registered."


def authenticate_user(email, password):

    with get_connection() as conn:

        user = conn.execute("""
            SELECT id, name, email, salt, password_hash
            FROM users
            WHERE email = ?
        """, (email.strip().lower(),)).fetchone()

    if user is None:
        return None

    user_id, name, email, salt, stored_hash = user

    entered_hash = hash_password(password, salt)

    if secrets.compare_digest(entered_hash, stored_hash):

        return {
            "id": user_id,
            "name": name,
            "email": email
        }

    return None

# =========================================================
# 4. PREDICTION DATABASE OPERATIONS
# =========================================================

def save_prediction(user_id, income, co_income,
                    loan_amount, loan_term,
                    credit_history, prediction, probability):

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO predictions (
                user_id,
                applicant_income,
                coapplicant_income,
                loan_amount,
                loan_term,
                credit_history,
                prediction,
                approval_probability,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            income,
            co_income,
            loan_amount,
            loan_term,
            credit_history,
            prediction,
            probability,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))



def get_user_history(user_id):

    with get_connection() as conn:

        data = pd.read_sql_query("""
            SELECT
                id AS 'Prediction ID',
                applicant_income AS 'Applicant Income',
                coapplicant_income AS 'Co-applicant Income',
                loan_amount AS 'Loan Amount (Thousands)',
                loan_term AS 'Loan Term',
                credit_history AS 'Credit History',
                prediction AS 'Result',
                approval_probability AS 'Approval Probability (%)',
                created_at AS 'Date'
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
        """, conn, params=(user_id,))

    # Safely convert database values into numeric values.
    numeric_columns = [
        "Prediction ID",
        "Applicant Income",
        "Co-applicant Income",
        "Loan Amount (Thousands)",
        "Loan Term",
        "Credit History",
        "Approval Probability (%)"
    ]

    def safe_numeric(value):

        if value is None:
            return None

        if isinstance(value, (bytes, bytearray, memoryview)):

            try:
                value = bytes(value).decode("utf-8").strip()

                # Ignore empty or binary/null-byte values.
                if not value or "\x00" in value:
                    return None

            except UnicodeDecodeError:
                return None

        try:
            return float(value)

        except (ValueError, TypeError, OverflowError):
            return None

    for column in numeric_columns:

        if column in data.columns:

            data[column] = data[column].apply(safe_numeric)

    # Safely convert text columns.
    def safe_text(value):

        if value is None:
            return ""

        if isinstance(value, (bytes, bytearray, memoryview)):

            value = bytes(value).decode(
                "utf-8",
                errors="replace"
            )

        return str(value)

    for column in ["Result", "Date"]:

        if column in data.columns:
            data[column] = data[column].apply(safe_text)

    return data




# =========================================================
# 5. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    return None


model = load_model()

# =========================================================
# 6. SESSION STATE
# =========================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user" not in st.session_state:
    st.session_state.user = None

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# =========================================================
# 7. PROFESSIONAL CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F4F7FC;
    color: #172B4D;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101E38, #173F6B);
}

[data-testid="stSidebar"] * {
    color: white;
}

.brand {
    font-size: 29px;
    font-weight: 800;
    color: white;
    padding-top: 12px;
}

.brand span {
    color: #64C5FF;
}

.sidebar-caption {
    font-size: 10px;
    letter-spacing: 2px;
    color: #B5D2EF;
    margin-bottom: 25px;
}

.profile-box {
    background: rgba(255,255,255,0.10);
    border-radius: 13px;
    padding: 15px;
    margin: 15px 0;
}

/* HERO */

.hero {
    background: linear-gradient(115deg, #101E38, #185A96, #287FC0);
    padding: 36px;
    border-radius: 20px;
    margin-bottom: 28px;
    box-shadow: 0 12px 30px rgba(16,50,90,0.10);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 34px;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: #E0EEFF !important;
    font-size: 15px;
    margin-top: 12px;
}

/* CARDS */

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E4EAF2;
    padding: 23px;
    border-radius: 16px;
    box-shadow: 0 5px 20px rgba(20,50,90,0.04);
    min-height: 115px;
}

.metric-label {
    color: #718096;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.metric-value {
    color: #173657;
    font-size: 29px;
    font-weight: 800;
    margin-top: 12px;
}

.section-title {
    color: #173657 !important;
    font-size: 23px;
    font-weight: 800;
    margin: 22px 0 15px;
}

/* AUTH PANEL */

.auth-panel {
    background: linear-gradient(145deg, #101E38, #1765A5);
    padding: 40px 35px;
    border-radius: 22px;
    min-height: 530px;
    box-shadow: 0 15px 40px rgba(15,45,80,0.13);
}

.auth-panel h1 {
    color: white !important;
    font-size: 40px;
    font-weight: 800;
}

.auth-panel h2 {
    color: white !important;
    font-size: 27px;
    line-height: 1.5;
}

.auth-panel p {
    color: #D5EAFF !important;
    line-height: 1.9;
}

.auth-tag {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    color: #C7E8FF;
    padding: 9px 15px;
    border-radius: 30px;
    font-size: 11px;
    margin-top: 20px;
}

.auth-card {
    background: white;
    border: 1px solid #E5EAF2;
    border-radius: 22px;
    padding: 28px;
    box-shadow: 0 10px 35px rgba(20,50,90,0.06);
}

/* INPUTS AND LABELS */


/* ALL INPUT FIELDS - HIGH CONTRAST */

[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-baseweb="input"] input {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
    -webkit-text-fill-color: #172B4D !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* NUMBER INPUT CONTAINER */

[data-testid="stNumberInput"] [data-baseweb="input"] {
    background-color: #FFFFFF !important;
}

/* NUMBER INPUT PLUS AND MINUS BUTTONS */

[data-testid="stNumberInput"] button {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
    border-color: #CBD5E1 !important;
}

/* SELECT BOX */

[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
    border: 1px solid #CBD5E1 !important;
}

[data-baseweb="select"] input {
    color: #172B4D !important;
}

/* FIELD LABELS */

[data-testid="stWidgetLabel"] p {
    color: #263B53 !important;
    font-weight: 600 !important;
}

/* HELP TEXT */

[data-testid="stCaptionContainer"] p {
    color: #64748B !important;
}

/* INPUT FOCUS */

[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border: 2px solid #2878C8 !important;
    box-shadow: 0 0 0 2px rgba(40,120,200,0.10) !important;
}


/* BUTTONS */

div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(100deg, #14569B, #287DCC);
    color: white !important;
    border: none;
    border-radius: 10px;
    min-height: 46px;
    font-weight: 700;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background: #103E70;
    color: white !important;
}

/* WHITE CONTAINERS */

[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border-radius: 15px;
    border-color: #E4EAF2;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #8492A5;
    font-size: 12px;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 8. REUSABLE COMPONENTS
# =========================================================

def page_header(title, subtitle):

    st.html(f"""
    <div class="hero">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """)


def metric_card(label, value):

    st.html(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """)


def footer():

    st.markdown("""
    <div class="footer">
        LoanSense AI | Intelligent Loan Prediction Platform<br>
        Developed using Python, Machine Learning and Streamlit
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 9. LOGIN AND REGISTRATION
# =========================================================

def authentication_page():

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.05, 1], gap="large")

    with left:

        st.html("""
        <div class="auth-panel">

            <div style="font-size:55px;">🏦</div>

            <div class="auth-tag">
                ✦ INTELLIGENT FINANCIAL TECHNOLOGY
            </div>

            <h1>
                Loan<span style="color:#65C5FF;">Sense</span> AI
            </h1>

            <h2>
                Smarter Loan Decisions.<br>
                Powered by Machine Learning.
            </h2>

            <p>
                Welcome to an intelligent loan prediction platform.
                Explore data-driven predictions through a simple
                and interactive experience.
            </p>

            <br>

            <p>
                ✦ Intelligent loan prediction<br>
                ✦ Personalized dashboard<br>
                ✦ Interactive analytics<br>
                ✦ Secure account authentication
            </p>

        </div>
        """)

    with right:

        with st.container(border=True):

            st.markdown("### Welcome to LoanSense 👋")
            st.caption("Sign in or create an account to continue.")

            login_tab, register_tab = st.tabs([
                "🔐 Login",
                "✨ Create Account"
            ])

            # LOGIN FORM

            with login_tab:

                st.markdown("#### Welcome Back!")
                st.caption("Enter your registered credentials.")

                with st.form("login_form"):

                    email = st.text_input(
                        "Email Address",
                        placeholder="name@example.com"
                    )

                    password = st.text_input(
                        "Password",
                        type="password",
                        placeholder="Enter your password"
                    )

                    login_submit = st.form_submit_button(
                        "LOGIN TO LOANSENSE →",
                        use_container_width=True
                    )

                if login_submit:

                    if not email.strip() or not password:

                        st.error("Please enter your email and password.")

                    else:

                        user = authenticate_user(
                            email,
                            password
                        )

                        if user:

                            st.session_state.authenticated = True
                            st.session_state.user = user
                            st.session_state.last_prediction = None

                            st.rerun()

                        else:

                            st.error("Invalid email or password.")

            # REGISTRATION FORM

            with register_tab:

                st.markdown("#### Create Your Account")
                st.caption("Register to access your personal dashboard.")

                with st.form("register_form"):

                    name = st.text_input(
                        "Full Name",
                        placeholder="Enter your full name"
                    )

                    new_email = st.text_input(
                        "Email Address",
                        placeholder="name@example.com"
                    )

                    new_password = st.text_input(
                        "Create Password",
                        type="password",
                        placeholder="Minimum 8 characters"
                    )

                    confirm_password = st.text_input(
                        "Confirm Password",
                        type="password",
                        placeholder="Re-enter password"
                    )

                    register_submit = st.form_submit_button(
                        "CREATE MY ACCOUNT →",
                        use_container_width=True
                    )

                if register_submit:

                    valid_email = re.fullmatch(
                        r"[^@\s]+@[^@\s]+\.[^@\s]+",
                        new_email.strip()
                    )

                    if not name.strip():

                        st.error("Enter your full name.")

                    elif not valid_email:

                        st.error("Enter a valid email address.")

                    elif len(new_password) < 8:

                        st.error("Password must have at least 8 characters.")

                    elif new_password != confirm_password:

                        st.error("Passwords do not match.")

                    else:

                        success, message = register_user(
                            name.strip(),
                            new_email,
                            new_password
                        )

                        if success:
                            st.success(message + " Please log in.")

                        else:
                            st.error(message)


    footer()


# =========================================================
# 10. MAIN APPLICATION AND SIDEBAR
# =========================================================

def main_application():

    user = st.session_state.user
    user_id = user["id"]

    with st.sidebar:

        st.markdown("""
        <div class="brand">
            Loan<span>Sense</span> AI 🏦
        </div>
        <div class="sidebar-caption">
            INTELLIGENT LOAN ANALYTICS
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div class="profile-box">',
            unsafe_allow_html=True
        )

        st.markdown(f"### 👋 {user['name']}")
        st.caption(user["email"])

        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        page = st.radio(
            "MAIN MENU",
            [
                "🏠 Dashboard",
                "📝 Loan Prediction",
                "📊 Prediction History",
                "📈 Analytics",
                "🤖 About Model",
                "👤 My Profile"
            ]
        )

        st.divider()

        if model is not None:
            st.success("● ML Model Online")
        else:
            st.error("● ML Model Offline")

        if st.button("🚪 Logout", use_container_width=True):

            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.last_prediction = None

            st.rerun()

        st.caption("LoanSense AI | Version 4.0")

    history = get_user_history(user_id)

    # =====================================================
    # DASHBOARD
    # =====================================================

    if page == "🏠 Dashboard":

        page_header(
            f"Welcome, {user['name']} 👋",
            "Your personal intelligent loan prediction workspace."
        )

        total = len(history)

        approved = (
            int((history["Result"] == "Approved").sum())
            if total else 0
        )

        rejected = total - approved

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card("TOTAL PREDICTIONS", total)

        with c2:
            metric_card("PREDICTED APPROVALS", approved)

        with c3:
            metric_card("PREDICTED REJECTIONS", rejected)

        st.markdown(
            '<div class="section-title">Explore Your Workspace</div>',
            unsafe_allow_html=True
        )

        a, b = st.columns(2)

        with a:

            with st.container(border=True):

                st.markdown("### 📝 Loan Prediction")

                st.write(
                    "Enter applicant information and receive "
                    "an ML-based loan prediction."
                )

        with b:

            with st.container(border=True):

                st.markdown("### 📊 Prediction Analytics")

                st.write(
                    "Explore your previous predictions and "
                    "view your personal statistics."
                )

        st.markdown("### Recent Activity")

        if not history.empty:

            st.dataframe(
                history.head(5),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Your prediction history will appear here "
                "after your first prediction."
            )

        if model is None:

            st.warning(
                "Model not found. Run python -m src.train first."
            )

    # =====================================================
    # LOAN PREDICTION
    # =====================================================

    elif page == "📝 Loan Prediction":

        page_header(
            "Loan Application",
            "Enter applicant information to generate a machine learning prediction."
        )

        if model is None:

            st.error(
                "Trained model not found. Run python -m src.train."
            )

        else:

            st.info(
                "Please provide accurate information. Your data "
                "is used to generate an educational model prediction."
            )

            with st.form("loan_form"):

                st.markdown("### 👤 Personal Information")

                c1, c2 = st.columns(2)

                with c1:

                    gender = st.selectbox(
                        "Gender",
                        ["Male", "Female"],
                        help="Select the applicant's gender as recorded in the dataset."
                    )

                    married = st.selectbox(
                        "Marital Status",
                        ["Yes", "No"],
                        help="Select Yes if the applicant is married."
                    )

                    dependents = st.selectbox(
                        "Number of Dependents",
                        ["0", "1", "2", "3+"],
                        help="People financially dependent on the applicant."
                    )

                    education = st.selectbox(
                        "Education",
                        ["Graduate", "Not Graduate"],
                        help="Select the applicant's education category."
                    )

                with c2:

                    self_employed = st.selectbox(
                        "Self Employed",
                        ["No", "Yes"],
                        help="Yes for a self-employed applicant or business owner."
                    )

                    property_area = st.selectbox(
                        "Property Area",
                        ["Urban", "Semiurban", "Rural"],
                        help="Select the location category of the property."
                    )

                    credit_history = st.selectbox(
                        "Credit History",
                        [1.0, 0.0],
                        format_func=lambda x:
                        "Positive (1)" if x == 1.0
                        else "Negative (0)",
                        help="Use the applicant's actual credit history. Do not guess."
                    )

                    loan_term = st.selectbox(
                        "Loan Term (Months)",
                        [360, 300, 240, 180, 120, 60],
                        help="The requested repayment duration in months."
                    )

                st.markdown("---")

                st.markdown("### 💰 Financial Information")

                c3, c4 = st.columns(2)

                with c3:

                    applicant_income_rupees = st.number_input(
                        "Applicant Monthly Income (₹)",
                        min_value=0,
                        value=50000,
                        step=1000,
                        help="Enter the applicant's monthly income in rupees."
                    )

                    st.caption(
                        "Enter monthly income before deductions."
                    )

                    coapplicant_income_rupees = st.number_input(
                        "Co-applicant Monthly Income (₹)",
                        min_value=0,
                        value=0,
                        step=1000,
                        help="Enter the joint applicant's monthly income. Use 0 if there is no co-applicant."
                    )

                with c4:

                    loan_amount_rupees = st.number_input(
                        "Requested Loan Amount (₹)",
                        min_value=1000,
                        value=150000,
                        step=10000,
                        help="Enter the total requested loan amount in rupees."
                    )

                    st.caption(
                        "The application automatically converts "
                        "this into thousands of rupees for the model."
                    )

                st.markdown("---")

                st.caption(
                    "Check your credit report before selecting credit history. "
                    "If you are unsure, do not guess."
                )

                submitted = st.form_submit_button(
                    "🚀 GENERATE LOAN PREDICTION",
                    use_container_width=True
                )

            if submitted:

                if applicant_income_rupees + coapplicant_income_rupees <= 0:

                    st.error("Please enter a valid total monthly income.")

                else:

                    # Convert rupees into thousands for LoanAmount.
                    # The original dataset stores LoanAmount in thousands.

                    loan_amount_model = loan_amount_rupees / 1000

                    applicant_income_model = applicant_income_rupees
                    coapplicant_income_model = coapplicant_income_rupees

                    applicant = pd.DataFrame([{

                        "Gender": gender,
                        "Married": married,
                        "Dependents": dependents,
                        "Education": education,
                        "Self_Employed": self_employed,
                        "ApplicantIncome": applicant_income_model,
                        "CoapplicantIncome": coapplicant_income_model,
                        "LoanAmount": loan_amount_model,
                        "Loan_Amount_Term": loan_term,
                        "Credit_History": credit_history,
                        "Property_Area": property_area

                    }])

                    try:

                        prediction = int(
                            model.predict(applicant)[0]
                        )

                        classes = list(model.classes_)
                        approval_index = classes.index(1)

                        probability = float(
                            model.predict_proba(applicant)[0][approval_index]
                        )

                        status = (
                            "Approved" if prediction == 1
                            else "Rejected"
                        )

                        save_prediction(
                            user_id,
                            applicant_income_model,
                            coapplicant_income_model,
                            loan_amount_model,
                            loan_term,
                            credit_history,
                            status,
                            probability * 100
                        )

                        st.session_state.last_prediction = {
                            "status": status,
                            "probability": probability
                        }

                        st.markdown("---")

                        st.markdown(
                            '<div class="section-title">Your Prediction Result</div>',
                            unsafe_allow_html=True
                        )

                        if prediction == 1:

                            st.success(
                                "✅ LOAN APPROVED – MODEL PREDICTION"
                            )

                        else:

                            st.error(
                                "❌ LOAN REJECTED – MODEL PREDICTION"
                            )

                        r1, r2 = st.columns(2)

                        with r1:

                            st.metric(
                                "Approval Probability",
                                f"{probability * 100:.2f}%"
                            )

                        with r2:

                            st.metric(
                                "Rejection Probability",
                                f"{(1-probability) * 100:.2f}%"
                            )

                        st.progress(probability)

                        st.info(
                            "Your prediction has been saved to your account history."
                        )

                        st.caption(
                            "This is an educational machine learning estimate, "
                            "not an actual bank approval or guarantee."
                        )

                    except Exception as error:

                        st.error(
                            f"Prediction failed: {error}"
                        )

                        st.warning(
                            "Check that your model was trained using the same "
                            "features and preprocessing as this application."
                        )

    # =====================================================
    # PREDICTION HISTORY
    # =====================================================

    elif page == "📊 Prediction History":

        page_header(
            "Prediction History",
            "View your previously saved loan prediction records."
        )

        if not history.empty:

            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True
            )

            csv_data = history.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇ Download My Prediction History",
                csv_data,
                "my_loan_history.csv",
                "text/csv",
                use_container_width=True
            )

            if st.button("🗑 Clear My Prediction History"):

                delete_user_history(user_id)

                st.success("Your history has been cleared.")

                st.rerun()

        else:

            st.info(
                "No prediction records found. Make your first prediction "
                "to see your history here."
            )

    # =====================================================
    # ANALYTICS
    # =====================================================

    elif page == "📈 Analytics":

        page_header(
            "Prediction Analytics",
            "Interactive visual insights from your prediction records."
        )

        if history.empty:

            st.info(
                "Generate predictions to unlock your personal analytics dashboard."
            )

        else:

            approved = int(
                (history["Result"] == "Approved").sum()
            )

            rejected = int(
                (history["Result"] == "Rejected").sum()
            )

            total = len(history)

            approval_rate = approved / total * 100

            c1, c2, c3 = st.columns(3)

            with c1:
                metric_card("TOTAL APPLICATIONS", total)

            with c2:
                metric_card("PREDICTED APPROVAL RATE", f"{approval_rate:.1f}%")

            with c3:
                metric_card("PREDICTED REJECTIONS", rejected)

            st.markdown(
                '<div class="section-title">Loan Approval Distribution</div>',
                unsafe_allow_html=True
            )

            chart_data = pd.DataFrame({
                "Loan Status": ["Approved", "Rejected"],
                "Applications": [approved, rejected]
            })

            fig1 = px.bar(
                chart_data,
                x="Loan Status",
                y="Applications",
                color="Loan Status",
                text="Applications",
                color_discrete_map={
                    "Approved": "#16A36A",
                    "Rejected": "#E45B65"
                },
                template="plotly_white"
            )

            fig1.update_layout(
                height=380,
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(color="#263B53"),
                showlegend=False,
                margin=dict(l=25, r=25, t=35, b=25),
                yaxis=dict(
                    title="Number of Applications",
                    rangemode="tozero"
                ),
                xaxis=dict(title="")
            )

            fig1.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                fig1,
                use_container_width=True
            )

            st.markdown(
                '<div class="section-title">Approval Probability Trend</div>',
                unsafe_allow_html=True
            )

            trend = history.copy()

            trend["Approval Probability (%)"] = pd.to_numeric(
                trend["Approval Probability (%)"]
            )

            trend = trend.sort_values("Prediction ID")

            trend["Application"] = range(
                1, len(trend) + 1
            )

            fig2 = px.line(
                trend,
                x="Application",
                y="Approval Probability (%)",
                markers=True,
                template="plotly_white",
                hover_data=["Date", "Result"]
            )

            fig2.update_traces(
                line=dict(
                    color="#2878C8",
                    width=3
                ),
                marker=dict(
                    size=9,
                    color="#2878C8"
                )
            )

            fig2.update_layout(
                height=380,
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(color="#263B53"),
                margin=dict(l=25, r=25, t=35, b=25),
                yaxis=dict(
                    title="Approval Probability (%)",
                    range=[0, 100]
                ),
                xaxis=dict(
                    title="Application Number",
                    dtick=1
                )
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

            st.caption(
                "These charts summarize your own model predictions, "
                "not actual bank approval statistics."
            )

    # =====================================================
    # ABOUT MODEL
    # =====================================================

    elif page == "🤖 About Model":

        page_header(
            "About LoanSense AI",
            "Technology and machine learning behind our application."
        )

        st.markdown("### 🌳 Random Forest Classifier")

        st.write(
            "Random Forest is an ensemble classification algorithm "
            "that combines multiple decision trees to generate a prediction."
        )

        st.markdown("### Technology Stack")

        tech = [
            ("Python", "Core programming"),
            ("Pandas", "Data analysis"),
            ("NumPy", "Numerical computing"),
            ("Scikit-learn", "Machine learning"),
            ("Random Forest", "Classification"),
            ("Streamlit", "Web application"),
            ("SQLite", "Database"),
            ("Plotly", "Interactive charts")
        ]

        for name, purpose in tech:

            st.markdown(f"**{name}** — {purpose}")

        st.markdown("### Model Features")

        st.write("""
        - Gender
        - Marital Status
        - Dependents
        - Education
        - Self Employment
        - Applicant Income
        - Co-applicant Income
        - Loan Amount
        - Loan Term
        - Credit History
        - Property Area
        """)

        st.info(
            "LoanSense is an academic machine learning prototype "
            "using historical data. It must not be used as a real "
            "bank lending or credit decision system."
        )

    # =====================================================
    # PROFILE
    # =====================================================

    elif page == "👤 My Profile":

        page_header(
            "My Profile",
            "Your personal LoanSense account information."
        )

        with st.container(border=True):

            st.markdown("### 👤 Account Details")

            st.write("**Full Name:**", user["name"])
            st.write("**Email Address:**", user["email"])
            st.write("**Account ID:**", user["id"])

            st.write(
                "**Total Predictions:**",
                len(history)
            )

            st.success("Your account is currently active.")

    footer()


# =========================================================
# 11. APPLICATION ENTRY POINT
# =========================================================

if st.session_state.authenticated:

    main_application()

else:

    authentication_page()

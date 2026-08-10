import hashlib
import io
import secrets
import string
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from database import (
    COLLECTIONS,
    append_dataframe_unique,
    clear_supplier_data,
    connect_to_mongodb,
    dataframe_from_collection,
    ensure_indexes,
    log_activity,
    replace_collection_from_dataframe,
)


st.set_page_config(
    page_title="Supplier Recommendation and Risk Analysis",
    page_icon="SR",
    layout="wide",
)


def inject_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --app-blue: #24489d;
            --app-blue-light: #5f83f2;
            --app-bg: #f6f8fc;
            --app-card: #ffffff;
            --app-border: #d9e0ec;
            --app-text: #0f172a;
            --app-muted: #64748b;
            --app-green: #16a34a;
            --app-red: #dc2626;
            --app-amber: #f59e0b;
            --app-purple: #8b5cf6;
            --app-cyan: #06b6d4;
        }

        .stApp {
            background: var(--app-bg);
            color: var(--app-text);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #24489d 0%, #1d3d89 100%);
            border-right: 0;
        }

        section[data-testid="stSidebar"] * {
            color: #ffffff !important;
        }

        section[data-testid="stSidebar"] h1 {
            font-size: 1.65rem !important;
            font-weight: 800 !important;
            margin-bottom: 1.2rem !important;
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0.35rem 0.15rem 1.2rem 0.15rem;
            margin-bottom: 0.4rem;
        }

        .brand-icon {
            width: 42px;
            height: 42px;
            display: grid;
            place-items: center;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.16);
            font-size: 1.45rem;
        }

        .brand-name {
            font-size: 1.45rem;
            font-weight: 850;
            line-height: 1.05;
        }

        .brand-name span {
            color: #8ee7ff !important;
        }

        .brand-subtitle {
            font-size: 0.76rem;
            opacity: 0.82;
            margin-top: 0.15rem;
        }

        .sidebar-role-badge {
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid rgba(255, 255, 255, 0.22);
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            margin: 0.75rem 0 0.75rem 0;
            font-weight: 750;
        }

        .sidebar-role-badge small {
            display: block;
            opacity: 0.78;
            font-weight: 600;
            margin-top: 0.1rem;
        }

        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            opacity: 0.82;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            border-radius: 8px;
            padding: 0.7rem 0.8rem;
            margin: 0.2rem 0;
            transition: background 0.2s ease, transform 0.2s ease;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label p {
            font-size: 1rem !important;
            font-weight: 650 !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.14);
            transform: translateX(2px);
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: #5f83f2;
            box-shadow: 0 10px 20px rgba(14, 34, 92, 0.25);
            font-weight: 800;
        }

        section[data-testid="stSidebar"] .stButton > button {
            background: rgba(255, 255, 255, 0.12) !important;
            border: 1px solid rgba(255, 255, 255, 0.22) !important;
            color: #ffffff !important;
            width: 100%;
        }

        .block-container {
            padding-top: 2rem;
            padding-left: 2.2rem;
            padding-right: 2.2rem;
            max-width: 1320px;
        }

        .page-hero {
            background: linear-gradient(135deg, #ffffff 0%, #edf4ff 62%, #e9fbff 100%);
            border: 1px solid var(--app-border);
            border-left: 6px solid var(--app-blue-light);
            border-radius: 8px;
            padding: 1.05rem 1.2rem;
            margin: 0 0 1.25rem 0;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        }

        .page-hero-top {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .page-hero-icon {
            width: 46px;
            height: 46px;
            display: grid;
            place-items: center;
            border-radius: 8px;
            color: #ffffff;
            background: linear-gradient(135deg, var(--app-blue), var(--app-blue-light));
            font-size: 1.35rem;
            box-shadow: 0 8px 18px rgba(36, 72, 157, 0.24);
        }

        .page-hero h1 {
            margin: 0 !important;
            padding: 0 !important;
            font-size: 2rem !important;
            line-height: 1.1 !important;
        }

        .page-hero p {
            margin: 0.25rem 0 0 0;
            color: var(--app-muted);
            font-size: 0.95rem;
        }

        .inline-page-title {
            background: linear-gradient(135deg, #ffffff 0%, #edf4ff 100%);
            border: 1px solid var(--app-border);
            border-left: 6px solid var(--app-blue-light);
            border-radius: 8px;
            padding: 0.95rem 1.1rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        }

        .inline-page-title h1 {
            margin: 0 !important;
            padding: 0 !important;
            font-size: 2rem !important;
        }

        .inline-page-title p {
            margin: 0.2rem 0 0 0;
            color: var(--app-muted);
        }

        .search-page-title {
            margin-bottom: 0.35rem;
        }

        .search-page-title h1 {
            margin: 0 !important;
            padding: 0 !important;
            font-size: 2rem !important;
            line-height: 1.12 !important;
        }

        .search-page-title p {
            margin: 0.25rem 0 0 0;
            color: var(--app-muted);
        }

        h1, h2, h3 {
            color: var(--app-text);
            letter-spacing: 0;
        }

        h1 {
            font-size: 2.1rem !important;
            font-weight: 800 !important;
        }

        h2, h3 {
            font-weight: 750 !important;
        }

        h2::before, h3::before {
            content: "";
            display: inline-block;
            width: 0.45rem;
            height: 0.9rem;
            background: var(--app-blue-light);
            border-radius: 99px;
            margin-right: 0.45rem;
            vertical-align: -0.08rem;
        }

        div[data-testid="stMetric"] {
            background: var(--app-card);
            border: 1px solid var(--app-border);
            border-top: 4px solid var(--app-blue-light);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="column"]:nth-of-type(2n) div[data-testid="stMetric"] {
            border-top-color: var(--app-green);
        }

        div[data-testid="column"]:nth-of-type(3n) div[data-testid="stMetric"] {
            border-top-color: var(--app-amber);
        }

        div[data-testid="column"]:nth-of-type(4n) div[data-testid="stMetric"] {
            border-top-color: var(--app-purple);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--app-muted) !important;
            font-size: 0.9rem;
            font-weight: 650;
        }

        div[data-testid="stMetricValue"] {
            color: var(--app-text);
            font-weight: 800;
        }

        div[data-testid="stMetricDelta"] {
            color: var(--app-green);
        }

        .stDataFrame,
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--app-border);
            border-radius: 8px;
            overflow: hidden;
            background: var(--app-card);
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stDataFrame"] div[role="columnheader"] {
            background: #eaf0ff !important;
            color: var(--app-text) !important;
            font-weight: 750 !important;
        }

        div[data-testid="stPlotlyChart"] {
            background: var(--app-card);
            border: 1px solid var(--app-border);
            border-radius: 8px;
            padding: 0.75rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stForm"],
        div[data-testid="stExpander"] {
            background: var(--app-card);
            border: 1px solid var(--app-border);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }

        .stTabs [data-baseweb="tab-list"] {
            background: #eaf0ff;
            border-radius: 8px;
            padding: 0.3rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 7px;
            font-weight: 700;
        }

        .stTabs [aria-selected="true"] {
            background: #ffffff;
            color: var(--app-blue) !important;
        }

        .stButton > button,
        .stDownloadButton > button,
        button[kind="secondary"] {
            border-radius: 8px !important;
            border: 1px solid #cbd5e1 !important;
            background: #ffffff !important;
            color: #0f172a !important;
            font-weight: 650 !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--app-blue-light) !important;
            color: var(--app-blue) !important;
        }

        .stButton > button[kind="primary"] {
            background: var(--app-red) !important;
            border-color: var(--app-red) !important;
            color: #ffffff !important;
        }

        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] {
            border-radius: 8px !important;
            background: #ffffff !important;
        }

        .stAlert {
            border-radius: 8px;
        }

        hr {
            border-color: var(--app-border);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


PAGE_DETAILS = {
    "Login": ("🚚", "Supplier Recommendation and Risk Analysis System", "MongoDB, Streamlit, analytics, recommendation, and user feedback."),
    "Home": ("🏠", "Home", "Quick overview of supplier performance, product trends, and rating activity."),
    "Upload Data": ("📤", "Upload Data", "Add supplier orders or product trend CSV data into the system."),
    "Clean Data": ("🧹", "Clean Data", "Prepare supplier and product trend data for analysis."),
    "View Data": ("📋", "View Data", "Inspect supplier metrics and product trend records."),
    "EDA & KPI Analysis": ("📈", "EDA & KPI Analysis", "Use charts to understand supplier performance, risk, ratings, and product trends."),
    "Supplier Category Trend & Prediction": ("📊", "Supplier Category Trend & Prediction", "Category-specific supplier analytics, product trends, prediction, and what-if analysis."),
    "Supplier Dashboard": ("📊", "Supplier Dashboard", "Your supplier performance, category demand, feedback, and summary."),
    "Supplier Trend": ("📈", "Trend", "Category-specific product trend, current demand, and up/down trend products."),
    "Future Prediction": ("🔮", "Future Prediction", "Future demand, supplier risk prediction, and what-if improvement analysis."),
    "User Rating": ("⭐", "User Rating", "Review user feedback submitted after supplier experience."),
    "Manage Accounts": ("👥", "Manage Accounts", "Manage user accounts, supplier requests, and supplier verification codes."),
    "Best Suppliers": ("🏠", "Best Suppliers", "Search suppliers quickly or view top supplier options."),
    "Find Supplier": ("🔎", "Find Supplier", "Enter requirements and receive ranked supplier recommendations."),
    "Rate Supplier": ("⭐", "Rate Supplier", "Submit feedback after selecting and using a supplier."),
    "My History": ("🕘", "My History", "View selected suppliers and ratings you submitted."),
}


def page_header(page_key):
    icon, title, subtitle = PAGE_DETAILS.get(page_key, ("📌", page_key, ""))
    st.markdown(
        f"""
        <div class="page-hero">
            <div class="page-hero-top">
                <div class="page-hero-icon">{icon}</div>
                <div>
                    <h1>{title}</h1>
                    <p>{subtitle}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def table_cell_style(value):
    text = str(value).strip().lower()
    if text == "high":
        return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"
    if text == "medium":
        return "background-color: #fef3c7; color: #92400e; font-weight: 700;"
    if text == "low":
        return "background-color: #dcfce7; color: #166534; font-weight: 700;"
    if text in {"improving", "uptrend", "delivered", "good service", "good recovery"}:
        return "background-color: #dcfce7; color: #166534; font-weight: 700;"
    if text in {"declining", "downtrend", "cancelled", "poor quality", "supply loss"}:
        return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"
    if text in {"stable", "recommended", "selected"}:
        return "background-color: #e0f2fe; color: #075985; font-weight: 700;"
    return ""


def numeric_cell_style(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number >= 80:
        return "background-color: #dcfce7; color: #166534; font-weight: 700;"
    if number >= 60:
        return "background-color: #e0f2fe; color: #075985; font-weight: 700;"
    if number >= 40:
        return "background-color: #fef3c7; color: #92400e; font-weight: 700;"
    return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"


def rating_cell_style(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number >= 4:
        return "background-color: #dcfce7; color: #166534; font-weight: 700;"
    if number >= 3:
        return "background-color: #fef3c7; color: #92400e; font-weight: 700;"
    return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"


def delay_cell_style(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number <= 1:
        return "background-color: #dcfce7; color: #166534; font-weight: 700;"
    if number <= 3:
        return "background-color: #fef3c7; color: #92400e; font-weight: 700;"
    return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"


def style_table(df):
    if df is None or df.empty:
        return df
    styled = df.style.set_properties(
        **{
            "background-color": "#ffffff",
            "color": "#0f172a",
            "border-color": "#e2e8f0",
        }
    )
    styled = styled.set_table_styles(
        [
            {"selector": "thead th", "props": [("background-color", "#eaf0ff"), ("color", "#0f172a"), ("font-weight", "750")]},
            {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f8fafc")]},
        ]
    )
    badge_columns = [
        "risk_level",
        "trend_level",
        "trend_status",
        "status",
        "event_type",
        "Level",
    ]
    score_columns = [
        "supplier_rank_score",
        "supplier_kpi_score",
        "final_score",
        "risk_prediction_score",
        "requirement_match_score",
        "trend_score",
        "quality_score",
        "Data Quality",
        "Value",
    ]
    rating_columns = [
        "final_rating",
        "user_rating",
        "risk_handling_rating",
        "quality_rating",
        "rating",
        "recent_rating",
    ]
    delay_columns = ["avg_delay", "max_delay", "delay_days", "delivery_duration"]
    for col in badge_columns:
        if col in df.columns:
            styled = styled.map(table_cell_style, subset=[col])
    for col in score_columns:
        if col in df.columns:
            styled = styled.map(numeric_cell_style, subset=[col])
    for col in rating_columns:
        if col in df.columns:
            styled = styled.map(rating_cell_style, subset=[col])
    for col in delay_columns:
        if col in df.columns:
            styled = styled.map(delay_cell_style, subset=[col])
    return styled


def ui_dataframe(df, width="stretch"):
    st.dataframe(style_table(df), width=width)


RAW_TO_CLEAN_COLUMNS = {
    "Order_ID": "order_id",
    "Buyer_ID": "buyer_id",
    "Supplier_ID": "supplier",
    "Product_Category": "product_category",
    "Quantity_Ordered": "quantity_ordered",
    "Order_Date": "order_date",
    "Dispatch_Date": "dispatch_date",
    "Delivery_Date": "delivery_date",
    "Shipping_Mode": "shipping_mode",
    "Order_Value_USD": "order_value_usd",
    "Delay_Days": "delay_days",
    "Disruption_Type": "disruption_type",
    "Disruption_Severity": "disruption_severity",
    "Historical_Disruption_Count": "historical_disruption_count",
    "Supplier_Reliability_Score": "reliability_score",
    "Organization_ID": "organization_id",
    "Supply_Risk_Flag": "supply_risk_flag",
}

SEVERITY_MAP = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
EVENT_TYPES = [
    "late delivery",
    "poor quality",
    "supply loss",
    "high cost",
    "good service",
    "good recovery",
]
DATA_EXPORT_KEYS = [
    "raw_orders",
    "cleaned_orders",
    "supplier_metrics",
    "supplier_ratings",
    "hot_suppliers",
    "recommendation_logs",
    "activity_logs",
    "supplier_verification_codes",
]
HOT_SUPPLIERS_COLLECTION = COLLECTIONS.get("hot_suppliers", "hot_suppliers")
AUTH_TOKEN_PARAM = "login_token"
PRODUCT_TRENDS_PATH = Path(__file__).parent / "data" / "product_trends.csv"

PRIORITY_WEIGHTS = {
    "Balanced": {"kpi": 0.30, "rating": 0.20, "risk": 0.25, "match": 0.25},
    "Low Cost": {"kpi": 0.20, "rating": 0.15, "risk": 0.20, "match": 0.45},
    "High Quality": {"kpi": 0.25, "rating": 0.35, "risk": 0.20, "match": 0.20},
    "Fast Delivery": {"kpi": 0.25, "rating": 0.15, "risk": 0.20, "match": 0.40},
    "Low Risk": {"kpi": 0.25, "rating": 0.15, "risk": 0.45, "match": 0.15},
}


def password_hash(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


@st.cache_resource
def get_database():
    db = connect_to_mongodb()
    if db is not None:
        ensure_indexes(db)
    return db


def ensure_default_users(db):
    users = db[COLLECTIONS["users"]]
    default_users = [
        {"username": "admin", "password": "admin123", "role": "admin", "supplier_id": None},
        {"username": "user", "password": "user123", "role": "user", "supplier_id": None},
        {"username": "supplier_s31", "password": "supplier123", "role": "supplier", "supplier_id": "S31"},
        {"username": "supplier_s12", "password": "supplier123", "role": "supplier", "supplier_id": "S12"},
        {"username": "supplier_s10", "password": "supplier123", "role": "supplier", "supplier_id": "S10"},
    ]
    for account in default_users:
        users.update_one(
            {"username": account["username"]},
            {
                "$set": {
                    "password_hash": password_hash(account["password"]),
                    "role": account["role"],
                    "supplier_id": account["supplier_id"],
                    "is_active": True,
                    "account_status": "approved",
                },
                "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
            },
            upsert=True,
        )


def session_user_from_doc(user_doc):
    return {
        "username": user_doc["username"],
        "role": user_doc["role"],
        "supplier_id": user_doc.get("supplier_id"),
    }


def create_login_session(db, user_doc):
    token = secrets.token_urlsafe(32)
    db[COLLECTIONS["auth_sessions"]].insert_one(
        {
            "token": token,
            "username": user_doc["username"],
            "created_at": datetime.now(timezone.utc),
        }
    )
    st.query_params[AUTH_TOKEN_PARAM] = token


def restore_login_session(db):
    token = st.query_params.get(AUTH_TOKEN_PARAM)
    if not token:
        return False
    session_doc = db[COLLECTIONS["auth_sessions"]].find_one({"token": token}, {"_id": 0})
    if not session_doc:
        return False
    user_doc = db[COLLECTIONS["users"]].find_one({"username": session_doc["username"]}, {"_id": 0})
    if not user_doc:
        return False
    if user_doc.get("role") == "supplier" and user_doc.get("account_status", "pending") != "approved":
        return False
    if not user_doc.get("is_active", False):
        return False
    st.session_state["user"] = session_user_from_doc(user_doc)
    return True


def clear_login_session(db):
    token = st.query_params.get(AUTH_TOKEN_PARAM)
    if token:
        db[COLLECTIONS["auth_sessions"]].delete_one({"token": token})
        del st.query_params[AUTH_TOKEN_PARAM]


def supplier_id_info(db, supplier_id, current_username=None):
    supplier_id = supplier_id.strip().upper() if supplier_id else ""
    clean_df = dataframe_from_collection(db, COLLECTIONS["cleaned_orders"], {"supplier": supplier_id})
    raw_df = dataframe_from_collection(db, COLLECTIONS["raw_orders"], {"Supplier_ID": supplier_id})
    source_df = clean_df if not clean_df.empty else raw_df
    categories = []
    if not clean_df.empty and "product_category" in clean_df.columns:
        categories = sorted(clean_df["product_category"].dropna().astype(str).unique())
    elif not raw_df.empty and "Product_Category" in raw_df.columns:
        categories = sorted(raw_df["Product_Category"].dropna().astype(str).unique())
    supplier_exists = not source_df.empty
    order_count = len(source_df)
    claimed_query = {
        "role": "supplier",
        "supplier_id": supplier_id,
        "account_status": "approved",
    }
    if current_username:
        claimed_query["username"] = {"$ne": current_username}
    claimed_account = db[COLLECTIONS["users"]].find_one(claimed_query, {"_id": 0, "username": 1})
    already_claimed = claimed_account is not None
    return {
        "supplier_id": supplier_id,
        "supplier_id_exists": "Yes" if supplier_exists else "No",
        "order_count": order_count,
        "categories": ", ".join(categories) if categories else "None",
        "category_match": "Yes" if categories else "No",
        "already_claimed": "Yes" if already_claimed else "No",
        "claimed_by": claimed_account.get("username", "None") if claimed_account else "None",
    }


def generate_verification_code(supplier_id):
    alphabet = string.ascii_uppercase + string.digits
    suffix = "".join(secrets.choice(alphabet) for _ in range(6))
    return f"{supplier_id.upper()}-{suffix}"


def verification_collection(db):
    return db[COLLECTIONS["supplier_verification_codes"]]


def get_supplier_verification_code(db, supplier_id):
    supplier_id = supplier_id.strip().upper()
    return verification_collection(db).find_one({"supplier_id": supplier_id}, {"_id": 0})


def save_supplier_verification_code(db, supplier_id, actor):
    supplier_id = supplier_id.strip().upper()
    code = generate_verification_code(supplier_id)
    verification_collection(db).update_one(
        {"supplier_id": supplier_id},
        {
            "$set": {
                "supplier_id": supplier_id,
                "verification_code": code,
                "is_used": False,
                "updated_at": datetime.now(timezone.utc),
                "updated_by": actor,
            },
            "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
        },
        upsert=True,
    )
    log_activity(db, "supplier_verification_code_generated", actor, {"supplier_id": supplier_id})
    return code


def validate_supplier_verification_code(db, supplier_id, verification_code):
    record = get_supplier_verification_code(db, supplier_id)
    if not record:
        return False, "No verification code found for this supplier ID. Ask admin to generate one."
    if record.get("is_used"):
        return False, "This verification code was already used."
    if str(record.get("verification_code", "")).strip() != verification_code.strip():
        return False, "Invalid supplier verification code."
    return True, "Verification code is valid."


def create_account(db, username, password, confirm_password, role, supplier_id=None, verification_code=None):
    username = username.strip()
    supplier_id = supplier_id.strip().upper() if supplier_id else None
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if password != confirm_password:
        return False, "Passwords do not match."
    if role not in {"user", "supplier"}:
        return False, "Only user and supplier signup are allowed."
    if role == "supplier" and not supplier_id:
        return False, "Supplier ID is required for supplier signup."
    if role == "supplier":
        valid_code, message = validate_supplier_verification_code(db, supplier_id, verification_code or "")
        if not valid_code:
            return False, message
        info = supplier_id_info(db, supplier_id)
        if info["supplier_id_exists"] != "Yes":
            return False, "Supplier ID was not found in uploaded supplier data."
        if info["order_count"] <= 0:
            return False, "Supplier ID has no order history."
        if info["category_match"] != "Yes":
            return False, "Supplier ID has no product category match."
        if info["already_claimed"] == "Yes":
            return False, "This supplier ID is already claimed by another approved supplier account."
    users = db[COLLECTIONS["users"]]
    if users.find_one({"username": username}):
        return False, "Username already exists."
    is_supplier = role == "supplier"
    users.insert_one(
        {
            "username": username,
            "password_hash": password_hash(password),
            "role": role,
            "supplier_id": supplier_id,
            "is_active": not is_supplier,
            "account_status": "pending" if is_supplier else "approved",
            "created_at": datetime.now(timezone.utc),
        }
    )
    if is_supplier:
        verification_collection(db).update_one(
            {"supplier_id": supplier_id},
            {"$set": {"is_used": True, "used_by": username, "used_at": datetime.now(timezone.utc)}},
        )
    log_activity(db, "account_created", username, {"role": role, "supplier_id": supplier_id})
    if is_supplier:
        return True, "Supplier request created. Please wait for admin approval."
    return True, "Account created. You can login now."


def load_collection(db, key):
    return dataframe_from_collection(db, COLLECTIONS[key])


def normalize_reliability(series):
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    return np.where(values > 1, values / 100, values)


def clean_orders(raw_df):
    missing = [col for col in RAW_TO_CLEAN_COLUMNS if col not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = raw_df[list(RAW_TO_CLEAN_COLUMNS)].rename(columns=RAW_TO_CLEAN_COLUMNS).copy()
    for col in ["order_date", "dispatch_date", "delivery_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_columns = [
        "quantity_ordered",
        "order_value_usd",
        "delay_days",
        "historical_disruption_count",
        "reliability_score",
        "supply_risk_flag",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["supplier"] = df["supplier"].astype(str).str.strip().str.upper()
    df["product_category"] = df["product_category"].astype(str).str.strip().str.title()
    df["shipping_mode"] = df["shipping_mode"].astype(str).str.strip().str.title()
    df["disruption_type"] = df["disruption_type"].fillna("None").astype(str).str.strip().str.title()
    df["disruption_severity"] = df["disruption_severity"].fillna("None").astype(str).str.strip().str.title()

    df = df.drop_duplicates(subset=["order_id"]).dropna(subset=["order_id", "supplier", "product_category"])
    df["quantity_ordered"] = df["quantity_ordered"].fillna(0).clip(lower=0)
    df["order_value_usd"] = df["order_value_usd"].fillna(0).clip(lower=0)
    df["delay_days"] = df["delay_days"].fillna(0).clip(lower=0)
    df["historical_disruption_count"] = df["historical_disruption_count"].fillna(0).clip(lower=0)
    df["reliability_score"] = normalize_reliability(df["reliability_score"])
    df["supply_risk_flag"] = df["supply_risk_flag"].fillna(0).clip(0, 1).astype(int)
    df["delivery_duration"] = (df["delivery_date"] - df["order_date"]).dt.days.fillna(df["delay_days"]).clip(lower=0)
    df["unit_price"] = np.where(df["quantity_ordered"] > 0, df["order_value_usd"] / df["quantity_ordered"], 0)
    df["severity_score"] = df["disruption_severity"].str.upper().map(SEVERITY_MAP).fillna(0)
    df["has_disruption"] = (df["disruption_type"].str.upper() != "NONE").astype(int)
    df["on_time_flag"] = (df["delay_days"] <= 0).astype(int)
    df["quality_score"] = ((df["reliability_score"] * 100) * 0.75) + ((100 - df["severity_score"] * 25) * 0.25)
    df["quality_rating"] = (df["quality_score"] / 20).clip(1, 5).round(2)

    for col in ["order_date", "dispatch_date", "delivery_date"]:
        df[col] = df[col].dt.strftime("%Y-%m-%d")
    return df


def data_quality_summary(raw_df):
    if raw_df.empty:
        return {
            "score": 0,
            "missing_values": 0,
            "duplicate_orders": 0,
            "invalid_dates": 0,
            "total_rows": 0,
        }
    missing_values = int(raw_df.isna().sum().sum())
    duplicate_orders = int(raw_df.duplicated(subset=["Order_ID"]).sum()) if "Order_ID" in raw_df else 0
    invalid_dates = 0
    for col in ["Order_Date", "Dispatch_Date", "Delivery_Date"]:
        if col in raw_df:
            invalid_dates += int(pd.to_datetime(raw_df[col], errors="coerce").isna().sum())
    total_cells = max(raw_df.shape[0] * raw_df.shape[1], 1)
    penalty = (missing_values / total_cells * 45) + (duplicate_orders / max(len(raw_df), 1) * 35) + (invalid_dates / max(len(raw_df), 1) * 20)
    return {
        "score": round(max(0, 100 - penalty), 1),
        "missing_values": missing_values,
        "duplicate_orders": duplicate_orders,
        "invalid_dates": invalid_dates,
        "total_rows": len(raw_df),
    }


def product_trend_quality_summary(raw_df):
    if raw_df.empty:
        return {
            "score": 0,
            "missing_values": 0,
            "duplicate_rows": 0,
            "invalid_months": 0,
            "total_rows": 0,
        }
    missing_values = int(raw_df.isna().sum().sum())
    duplicate_rows = int(raw_df.duplicated().sum())
    invalid_months = 0
    if "month" in raw_df:
        invalid_months = int(pd.to_datetime(raw_df["month"], errors="coerce").isna().sum())
    total_cells = max(raw_df.shape[0] * raw_df.shape[1], 1)
    penalty = (missing_values / total_cells * 45) + (duplicate_rows / max(len(raw_df), 1) * 35) + (invalid_months / max(len(raw_df), 1) * 20)
    return {
        "score": round(max(0, 100 - penalty), 1),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "invalid_months": invalid_months,
        "total_rows": len(raw_df),
    }


def clean_product_trends(raw_df):
    missing = [col for col in PRODUCT_TREND_REQUIRED_COLUMNS if col not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    columns = PRODUCT_TREND_REQUIRED_COLUMNS + [col for col in PRODUCT_TREND_OPTIONAL_COLUMNS if col in raw_df.columns]
    df = raw_df[columns].copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["product_category"] = df["product_category"].astype(str).str.strip().str.title()
    df["product_name"] = df["product_name"].astype(str).str.strip().str.title()

    for col in ["search_volume", "sales_count", "growth_rate", "trend_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "trend_level" not in df.columns:
        df["trend_level"] = np.select(
            [df["trend_score"] >= 75, df["trend_score"] >= 45],
            ["High", "Medium"],
            default="Low",
        )
    else:
        df["trend_level"] = df["trend_level"].astype(str).str.strip().str.title()
        df.loc[~df["trend_level"].isin(["High", "Medium", "Low"]), "trend_level"] = np.select(
            [df["trend_score"] >= 75, df["trend_score"] >= 45],
            ["High", "Medium"],
            default="Low",
        )
    if "data_source" not in df.columns:
        df["data_source"] = "Unknown"
    else:
        df["data_source"] = df["data_source"].fillna("Unknown").astype(str).str.strip()

    df = df.dropna(subset=["month", "product_category", "product_name"])
    df = df.drop_duplicates(subset=["month", "product_category", "product_name"])
    df["search_volume"] = df["search_volume"].clip(lower=0).round(0).astype(int)
    df["sales_count"] = df["sales_count"].clip(lower=0).round(0).astype(int)
    df["trend_score"] = df["trend_score"].clip(0, 100).round(2)
    df["growth_rate"] = df["growth_rate"].round(2)
    df["month"] = df["month"].dt.strftime("%Y-%m")
    return df.sort_values(["product_category", "product_name", "month"]).reset_index(drop=True)


def risk_level(score):
    if score < 35:
        return "Low"
    if score < 65:
        return "Medium"
    return "High"


def calculate_rating_aggregates(ratings_df):
    if ratings_df.empty:
        return pd.DataFrame(columns=["supplier", "product_category", "user_rating", "rating_count", "bad_feedback_count", "recent_rating"])
    df = ratings_df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    agg = (
        df.groupby(["supplier", "product_category"], as_index=False)
        .agg(
            user_rating=("rating", "mean"),
            rating_count=("rating", "count"),
            bad_feedback_count=("rating", lambda x: int((x <= 2).sum())),
            recent_rating=("rating", lambda x: round(x.tail(5).mean(), 2)),
        )
    )
    agg["user_rating"] = agg["user_rating"].round(2)
    return agg


def calculate_supplier_metrics(clean_df, ratings_df=None):
    if clean_df.empty:
        return clean_df

    df = clean_df.copy()
    numeric_cols = [
        "quantity_ordered",
        "order_value_usd",
        "delay_days",
        "historical_disruption_count",
        "reliability_score",
        "supply_risk_flag",
        "delivery_duration",
        "unit_price",
        "severity_score",
        "has_disruption",
        "on_time_flag",
        "quality_score",
        "quality_rating",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    grouped = (
        df.groupby(["supplier", "product_category"], as_index=False)
        .agg(
            total_orders=("order_id", "count"),
            total_quantity=("quantity_ordered", "sum"),
            total_order_value=("order_value_usd", "sum"),
            avg_unit_price=("unit_price", "mean"),
            avg_delay=("delay_days", "mean"),
            max_delay=("delay_days", "max"),
            reliability=("reliability_score", "mean"),
            on_time_delivery_rate=("on_time_flag", "mean"),
            disruption_count=("has_disruption", "sum"),
            disruption_frequency=("has_disruption", "mean"),
            avg_severity=("severity_score", "mean"),
            historical_disruption_count=("historical_disruption_count", "mean"),
            supply_risk_rate=("supply_risk_flag", "mean"),
            quality_rating=("quality_rating", "mean"),
        )
    )

    rating_agg = calculate_rating_aggregates(ratings_df if ratings_df is not None else pd.DataFrame())
    grouped = grouped.merge(rating_agg, on=["supplier", "product_category"], how="left")
    grouped["user_rating"] = grouped["user_rating"].fillna(grouped["quality_rating"]).clip(1, 5)
    grouped["recent_rating"] = grouped["recent_rating"].fillna(grouped["user_rating"]).clip(1, 5)
    grouped["rating_count"] = grouped["rating_count"].fillna(0).astype(int)
    grouped["bad_feedback_count"] = grouped["bad_feedback_count"].fillna(0).astype(int)

    delay_max = max(grouped["avg_delay"].max(), 1)
    history_max = max(grouped["historical_disruption_count"].max(), 1)
    bad_feedback_max = max(grouped["bad_feedback_count"].max(), 1)
    grouped["risk_score"] = (
        ((1 - grouped["reliability"]) * 25)
        + ((grouped["avg_delay"] / delay_max) * 20)
        + (grouped["disruption_frequency"] * 15)
        + (grouped["supply_risk_rate"] * 15)
        + ((grouped["avg_severity"] / 3) * 10)
        + ((5 - grouped["user_rating"]) / 4 * 10)
        + (grouped["bad_feedback_count"] / bad_feedback_max * 5)
    ).clip(0, 100)
    grouped["risk_level"] = grouped["risk_score"].apply(risk_level)
    grouped["risk_handling_rating"] = (5 - grouped["risk_score"] / 25).clip(1, 5)
    grouped["supplier_kpi_score"] = (
        grouped["reliability"] * 35
        + grouped["on_time_delivery_rate"] * 20
        + (grouped["quality_rating"] / 5 * 20)
        + ((100 - grouped["risk_score"]) / 100 * 25)
    ).clip(0, 100)
    grouped["final_rating"] = ((grouped["risk_handling_rating"] * 0.55) + (grouped["user_rating"] * 0.45)).clip(1, 5)
    grouped["trend_status"] = np.select(
        [
            grouped["recent_rating"] >= grouped["user_rating"] + 0.25,
            grouped["recent_rating"] <= grouped["user_rating"] - 0.25,
        ],
        ["Improving", "Declining"],
        default="Stable",
    )
    grouped["supplier_rank_score"] = (
        grouped["supplier_kpi_score"] * 0.55
        + grouped["user_rating"] / 5 * 20
        + (100 - grouped["risk_score"]) * 0.25
    ).clip(0, 100)

    for col in grouped.select_dtypes(include=[np.number]).columns:
        grouped[col] = grouped[col].round(2)
    return grouped.sort_values("supplier_rank_score", ascending=False)


def refresh_metrics(db):
    clean_df = load_collection(db, "cleaned_orders")
    ratings_df = load_collection(db, "supplier_ratings")
    metrics_df = calculate_supplier_metrics(clean_df, ratings_df)
    replace_collection_from_dataframe(db, COLLECTIONS["supplier_metrics"], metrics_df)
    return metrics_df


REQUIRED_METRIC_COLUMNS = {
    "supplier",
    "product_category",
    "supplier_kpi_score",
    "user_rating",
    "risk_score",
    "risk_level",
    "final_rating",
    "supplier_rank_score",
    "trend_status",
}


def load_supplier_metrics(db):
    metrics_df = load_collection(db, "supplier_metrics")
    if not metrics_df.empty and REQUIRED_METRIC_COLUMNS.issubset(metrics_df.columns):
        return metrics_df

    clean_df = load_collection(db, "cleaned_orders")
    if clean_df.empty:
        return pd.DataFrame()
    return refresh_metrics(db)


def safe_metric_table(metrics_df, columns):
    existing = [column for column in columns if column in metrics_df.columns]
    if not existing:
        return pd.DataFrame()
    return metrics_df[existing]


PRODUCT_TREND_REQUIRED_COLUMNS = [
    "month",
    "product_category",
    "product_name",
    "search_volume",
    "sales_count",
    "growth_rate",
    "trend_score",
]
PRODUCT_TREND_OPTIONAL_COLUMNS = ["trend_level", "data_source"]


def validate_product_trend_csv(trend_df):
    missing = set(PRODUCT_TREND_REQUIRED_COLUMNS) - set(trend_df.columns)
    if missing:
        return False, f"Missing required columns: {', '.join(sorted(missing))}"
    for col in ["search_volume", "sales_count", "growth_rate", "trend_score"]:
        trend_df[col] = pd.to_numeric(trend_df[col], errors="coerce")
    invalid_numeric = trend_df[["search_volume", "sales_count", "growth_rate", "trend_score"]].isna().sum().sum()
    if invalid_numeric:
        return False, "Trend CSV has invalid numeric values."
    return True, "Trend CSV is valid."


def explain_supplier(row, budget=None, deadline=None):
    reasons = []
    if row.get("reliability", 0) >= 0.8:
        reasons.append("high reliability")
    if row.get("avg_delay", 99) <= 2:
        reasons.append("low average delay")
    if row.get("risk_level") == "Low":
        reasons.append("low predicted supply risk")
    if row.get("user_rating", 0) >= 4:
        reasons.append("strong user rating")
    if budget and row.get("avg_unit_price", 0) <= budget:
        reasons.append("fits the budget")
    if deadline and row.get("avg_delay", 0) <= deadline:
        reasons.append("fits the deadline")
    if row.get("trend_status") == "Declining":
        reasons.append("but recent feedback is declining")
    return "Recommended because it has " + ", ".join(reasons or ["acceptable overall performance"]) + "."


RECOMMENDATION_REQUIRED_COLUMNS = {
    "supplier",
    "product_category",
    "supplier_kpi_score",
    "user_rating",
    "risk_score",
    "quality_rating",
    "avg_delay",
    "avg_unit_price",
    "total_quantity",
}


def validate_recommendation_input(metrics_df, priority):
    if metrics_df.empty:
        return False, "No supplier metrics available."
    missing_columns = RECOMMENDATION_REQUIRED_COLUMNS - set(metrics_df.columns)
    if missing_columns:
        return False, f"Missing recommendation columns: {', '.join(sorted(missing_columns))}"
    if priority not in PRIORITY_WEIGHTS:
        return False, "Unknown recommendation priority."
    return True, "Recommendation input is valid."


def filter_supplier_options(metrics_df, category, budget, min_quality, deadline):
    options = metrics_df[metrics_df["product_category"].str.lower() == category.lower()].copy()
    if options.empty:
        return options

    for col in ["avg_unit_price", "quality_rating", "avg_delay", "total_quantity"]:
        options[col] = pd.to_numeric(options[col], errors="coerce").fillna(0)
    if budget > 0:
        options = options[options["avg_unit_price"] <= budget]
    return options[(options["quality_rating"] >= min_quality) & (options["avg_delay"] <= deadline)]


def calculate_requirement_match(options, quantity, budget, deadline):
    scored = options.copy()
    scored["cost_match"] = np.where(budget > 0, ((budget - scored["avg_unit_price"]) / budget * 100).clip(0, 100), 70)
    scored["deadline_match"] = ((deadline - scored["avg_delay"]) / max(deadline, 1) * 100).clip(0, 100)
    scored["quality_match"] = (scored["quality_rating"] / 5 * 100).clip(0, 100)
    scored["quantity_match"] = np.where(scored["total_quantity"] >= quantity, 100, scored["total_quantity"] / max(quantity, 1) * 100)
    scored["requirement_match_score"] = (
        scored["cost_match"] * 0.30
        + scored["deadline_match"] * 0.30
        + scored["quality_match"] * 0.25
        + scored["quantity_match"] * 0.15
    ).round(2)
    return scored


def calculate_recommendation_scores(options, priority):
    scored = options.copy()
    scored["user_rating_score"] = (scored["user_rating"] / 5 * 100).round(2)
    scored["risk_prediction_score"] = (100 - scored["risk_score"]).round(2)

    weights = PRIORITY_WEIGHTS[priority]
    scored["final_score"] = (
        scored["supplier_kpi_score"] * weights["kpi"]
        + scored["user_rating_score"] * weights["rating"]
        + scored["risk_prediction_score"] * weights["risk"]
        + scored["requirement_match_score"] * weights["match"]
    ).round(2)
    return scored


def rank_recommendations(options, top_n):
    return options.sort_values("final_score", ascending=False).head(top_n)


def recommend_suppliers(metrics_df, category, quantity, budget, min_quality, deadline, priority, top_n):
    is_valid, _ = validate_recommendation_input(metrics_df, priority)
    if not is_valid:
        return pd.DataFrame()

    options = filter_supplier_options(metrics_df, category, budget, min_quality, deadline)
    if options.empty:
        return options

    options = calculate_requirement_match(options, quantity, budget, deadline)
    options = calculate_recommendation_scores(options, priority)
    options["explanation"] = options.apply(lambda row: explain_supplier(row, budget, deadline), axis=1)
    return rank_recommendations(options, top_n)


def build_backup_zip(db):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for key in DATA_EXPORT_KEYS:
            df = load_collection(db, key)
            archive.writestr(f"{key}.csv", df.to_csv(index=False))
    buffer.seek(0)
    return buffer.getvalue()


def build_product_trend_backup_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        if PRODUCT_TRENDS_PATH.exists():
            archive.write(PRODUCT_TRENDS_PATH, arcname="product_trends.csv")
        else:
            archive.writestr("product_trends.csv", "")
    buffer.seek(0)
    return buffer.getvalue()


def hot_supplier_keys(db, username):
    hot_df = dataframe_from_collection(db, HOT_SUPPLIERS_COLLECTION, {"username": username})
    if hot_df.empty or not {"supplier", "product_category"}.issubset(hot_df.columns):
        return set()
    return set(zip(hot_df["supplier"].astype(str), hot_df["product_category"].astype(str)))


def save_hot_supplier(db, username, row):
    doc = {
        "username": username,
        "supplier": row["supplier"],
        "product_category": row["product_category"],
        "final_score": row.get("final_score", row.get("supplier_rank_score")),
        "final_rating": row.get("final_rating"),
        "risk_level": row.get("risk_level"),
        "risk_score": row.get("risk_score"),
        "avg_delay": row.get("avg_delay"),
        "avg_unit_price": row.get("avg_unit_price"),
        "explanation": row.get("explanation"),
        "created_at": datetime.now(timezone.utc),
    }
    db[HOT_SUPPLIERS_COLLECTION].update_one(
        {"username": username, "supplier": row["supplier"], "product_category": row["product_category"]},
        {"$set": doc},
        upsert=True,
    )


def save_selected_supplier(db, username, row, source):
    selection = {
        "username": username,
        "supplier": row["supplier"],
        "product_category": row["product_category"],
        "final_score": row.get("final_score", row.get("supplier_rank_score")),
        "risk_level": row.get("risk_level"),
        "status": "selected",
        "source": source,
        "created_at": datetime.now(timezone.utc),
    }
    db[COLLECTIONS["recommendation_logs"]].insert_one(selection)
    log_activity(db, f"supplier_selected_from_{source}", username, selection)
    st.session_state["selected_supplier"] = selection
    return selection


def sync_supplier_saved_scores(db, supplier, product_category, metrics_df):
    if metrics_df.empty:
        return
    match = metrics_df[(metrics_df["supplier"] == supplier) & (metrics_df["product_category"] == product_category)]
    if match.empty:
        return
    row = match.iloc[0]
    updated_values = {
        "final_score": row.get("supplier_rank_score"),
        "final_rating": row.get("final_rating"),
        "risk_level": row.get("risk_level"),
        "risk_score": row.get("risk_score"),
        "avg_delay": row.get("avg_delay"),
        "avg_unit_price": row.get("avg_unit_price"),
    }
    db[HOT_SUPPLIERS_COLLECTION].update_many(
        {"supplier": supplier, "product_category": product_category},
        {"$set": updated_values},
    )
    db[COLLECTIONS["recommendation_logs"]].update_many(
        {"supplier": supplier, "product_category": product_category, "status": "selected"},
        {"$set": {"final_score": row.get("supplier_rank_score"), "risk_level": row.get("risk_level")}},
    )


def remove_hot_supplier(db, username, supplier, product_category):
    db[HOT_SUPPLIERS_COLLECTION].delete_one(
        {"username": username, "supplier": supplier, "product_category": product_category}
    )


def page_login(db):
    page_header("Login")
    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
        if submitted:
            user = db[COLLECTIONS["users"]].find_one(
                {"username": username.strip(), "password_hash": password_hash(password)}
            )
            if user:
                if user.get("role") == "supplier" and user.get("account_status", "pending") != "approved":
                    st.warning("Your supplier account is waiting for admin approval.")
                    return
                if not user.get("is_active", False):
                    st.error("This account is inactive. Please contact admin.")
                    return
                st.session_state["user"] = session_user_from_doc(user)
                create_login_session(db, user)
                st.rerun()
            st.error("Invalid username or password.")

    with signup_tab:
        with st.form("signup_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            role = st.selectbox("Account Type", ["user", "supplier"])
            supplier_id = ""
            verification_code = ""
            if role == "supplier":
                supplier_id = st.text_input("Supplier ID", placeholder="Example: S31")
                verification_code = st.text_input("Supplier Verification Code", placeholder="Example: S10-ABC123")
                st.caption("Supplier accounts are created as pending until admin approval.")
            signup_submitted = st.form_submit_button("Create Account")
        if signup_submitted:
            success, message = create_account(db, new_username, new_password, confirm_password, role, supplier_id, verification_code)
            if success:
                st.success(message)
            else:
                st.error(message)
    st.info("Demo accounts: admin/admin123, user/user123, supplier_s31/supplier123, supplier_s12/supplier123, supplier_s10/supplier123")


def alert_rows(metrics_df):
    if metrics_df.empty:
        return []
    alerts = []
    for _, row in metrics_df.iterrows():
        label = f"{row['supplier']} - {row['product_category']}"
        if row["user_rating"] < 3:
            alerts.append({"Alert": f"{label} rating dropped below 3.0", "Level": "High"})
        if row["risk_level"] == "High":
            alerts.append({"Alert": f"{label} predicted risk is High", "Level": "High"})
        if row["trend_status"] == "Declining":
            alerts.append({"Alert": f"{label} feedback trend is declining", "Level": "Medium"})
    return alerts[:10]


def page_admin_dashboard(db):
    page_header("Home")
    raw_df = load_collection(db, "raw_orders")
    clean_df = load_collection(db, "cleaned_orders")
    metrics_df = load_supplier_metrics(db)
    ratings_df = load_collection(db, "supplier_ratings")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Orders", len(raw_df))
    c2.metric("Suppliers", clean_df["supplier"].nunique() if not clean_df.empty else 0)
    c3.metric("Categories", clean_df["product_category"].nunique() if not clean_df.empty else 0)
    c4.metric("High Risk", int((metrics_df["risk_level"] == "High").sum()) if not metrics_df.empty else 0)
    c5.metric("Avg Rating", round(metrics_df["final_rating"].mean(), 2) if not metrics_df.empty else 0)

    quality = data_quality_summary(raw_df)
    st.subheader("Clean Data")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Raw Rows", len(raw_df))
    sc2.metric("Cleaned Rows", len(clean_df))
    sc3.metric("Data Quality", f"{quality['score']}/100")
    sc4.metric("Missing Values", quality["missing_values"])
    if len(raw_df) > 0:
        st.caption(f"Duplicates: {quality['duplicate_orders']} | Invalid dates: {quality['invalid_dates']}")

    alerts = alert_rows(metrics_df)
    if alerts:
        st.subheader("Admin Alerts")
        ui_dataframe(pd.DataFrame(alerts), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Supplier Summary")
        if not metrics_df.empty:
            ui_dataframe(
                safe_metric_table(
                    metrics_df,
                    ["supplier", "product_category", "total_orders", "final_rating", "user_rating", "risk_level", "avg_delay", "trend_status"],
                ).head(12),
                width="stretch",
            )
        else:
            st.info("No supplier metrics yet. Run cleaning first.")

    with col2:
        st.subheader("Product Trend Summary")
        trends_df = load_product_trends()
        if not trends_df.empty:
            latest_month = trends_df["month"].max()
            latest_trends = trends_df[trends_df["month"] == latest_month].copy()
            ui_dataframe(
                safe_metric_table(
                    latest_trends.sort_values("trend_score", ascending=False),
                    ["product_category", "product_name", "trend_level", "growth_rate", "trend_score", "data_source"],
                ).head(12),
                width="stretch",
            )
        else:
            st.info("No product trend data yet. Upload product trend data first.")

    if st.button("Open Data Page"):
        st.session_state["admin_nav_target"] = "View Data"
        st.rerun()

    st.subheader("Rating Activity")
    r1, r2 = st.columns(2)
    r1.metric("User Ratings", len(ratings_df))
    bad_feedback = 0
    if not ratings_df.empty and "rating" in ratings_df.columns:
        bad_feedback = int((pd.to_numeric(ratings_df["rating"], errors="coerce") <= 2).sum())
    r2.metric("Bad Feedback", bad_feedback)


def page_upload(db):
    page_header("Upload Data")
    mode_col1, mode_col2 = st.columns(2)
    if mode_col1.button("Supplier", use_container_width=True):
        st.session_state["upload_mode"] = "Supplier"
    if mode_col2.button("Product Trend", use_container_width=True):
        st.session_state["upload_mode"] = "Product Trend"
    upload_mode = st.session_state.get("upload_mode", "Supplier")

    if upload_mode == "Supplier":
        st.subheader("Supplier")
        st.caption("Uploads append new rows. Existing order IDs are skipped.")
        uploaded = st.file_uploader("Upload supplier order CSV", type=["csv"])
        if uploaded:
            try:
                raw_df = pd.read_csv(uploaded, encoding="ISO-8859-1")
                ui_dataframe(raw_df.head(20), width="stretch")
                if st.button("Append CSV to MongoDB"):
                    inserted, skipped = append_dataframe_unique(db, COLLECTIONS["raw_orders"], raw_df, "Order_ID")
                    log_activity(db, "raw_data_appended", st.session_state["user"]["username"], {"inserted": inserted, "skipped_duplicates": skipped})
                    st.success(f"Added {inserted} new rows. Skipped {skipped} duplicate or invalid rows.")
            except Exception as exc:
                st.error(f"Could not read/upload this CSV: {exc}")

        st.divider()
        st.subheader("Load CSV From Local Path")
        st.caption("Use this if the browser file uploader disconnects. It reads the CSV directly from your computer path.")
        default_test_path = r"C:\Users\harle\Documents\Codex\2026-08-05\ana\outputs\supplier_cleaning_upload_test_10_rows.csv"
        local_csv_path = st.text_input("CSV file path", value=default_test_path)
        if st.button("Append"):
            try:
                raw_df = pd.read_csv(local_csv_path, encoding="ISO-8859-1")
                inserted, skipped = append_dataframe_unique(db, COLLECTIONS["raw_orders"], raw_df, "Order_ID")
                log_activity(
                    db,
                    "local_csv_appended",
                    st.session_state["user"]["username"],
                    {"path": local_csv_path, "inserted": inserted, "skipped_duplicates": skipped},
                )
                st.success(f"Added {inserted} new rows. Skipped {skipped} duplicate or invalid rows.")
                ui_dataframe(raw_df.head(20), width="stretch")
            except Exception as exc:
                st.error(f"Could not load this local CSV path: {exc}")
        return

    st.subheader("Product Trend")
    st.caption("Updates the Supplier Dashboard product trend analysis. Source can be Google Trends CSV summarized into this format.")

    template_df = pd.DataFrame(
        [
            {
                "month": "2026-08",
                "product_category": "Machinery",
                "product_name": "Industrial Robot",
                "search_volume": 61000,
                "sales_count": 6300,
                "growth_rate": 18,
                "trend_score": 88,
                "trend_level": "High",
                "data_source": "Google Trends",
            }
        ]
    )
    st.download_button(
        "Download Product Trend CSV Template",
        template_df.to_csv(index=False),
        "product_trends_template.csv",
        "text/csv",
    )

    uploaded_trend = st.file_uploader("Upload product trend CSV", type=["csv"], key="product_trend_uploader")
    if uploaded_trend:
        try:
            trend_df = pd.read_csv(uploaded_trend)
            valid, message = validate_product_trend_csv(trend_df)
            if not valid:
                st.error(message)
                return
            st.success(message)
            st.subheader("Product Trend Preview")
            ui_dataframe(trend_df.head(30), width="stretch")
            if st.button("Save Product Trend Dataset"):
                PRODUCT_TRENDS_PATH.parent.mkdir(parents=True, exist_ok=True)
                trend_df.to_csv(PRODUCT_TRENDS_PATH, index=False)
                log_activity(
                    db,
                    "product_trends_updated",
                    st.session_state["user"]["username"],
                    {"rows": len(trend_df), "categories": int(trend_df["product_category"].nunique())},
                )
                st.success(f"Saved {len(trend_df)} product trend rows. Supplier Dashboard now uses this dataset.")
        except Exception as exc:
            st.error(f"Could not read product trend CSV: {exc}")

    st.divider()
    st.subheader("Load Product Trend CSV From Local Path")
    st.caption("Use this if the browser uploader disconnects. This replaces the current product trend dataset.")
    default_trend_path = str(PRODUCT_TRENDS_PATH)
    trend_csv_path = st.text_input("Product trend CSV file path", value=default_trend_path)
    if st.button("Save Product Trend From Local Path"):
        try:
            trend_df = pd.read_csv(trend_csv_path)
            valid, message = validate_product_trend_csv(trend_df)
            if not valid:
                st.error(message)
                return
            PRODUCT_TRENDS_PATH.parent.mkdir(parents=True, exist_ok=True)
            trend_df.to_csv(PRODUCT_TRENDS_PATH, index=False)
            log_activity(
                db,
                "product_trends_updated_from_path",
                st.session_state["user"]["username"],
                {"path": trend_csv_path, "rows": len(trend_df), "categories": int(trend_df["product_category"].nunique())},
            )
            st.success(f"Saved {len(trend_df)} product trend rows from local path.")
            ui_dataframe(trend_df.head(30), width="stretch")
        except Exception as exc:
            st.error(f"Could not load product trend local CSV path: {exc}")
    return



def page_clean(db):
    page_header("Clean Data")

    mode_col1, mode_col2 = st.columns(2)
    if mode_col1.button("Supplier", use_container_width=True):
        st.session_state["clean_data_mode"] = "Supplier"
    if mode_col2.button("Product Trend", use_container_width=True):
        st.session_state["clean_data_mode"] = "Product Trend"
    clean_mode = st.session_state.get("clean_data_mode", "Supplier")

    if clean_mode == "Product Trend":
        st.subheader("Product Trend")
        if not PRODUCT_TRENDS_PATH.exists():
            st.warning("Upload product trend data first.")
            return

        raw_trend_df = pd.read_csv(PRODUCT_TRENDS_PATH)
        try:
            cleaned_trend_preview = clean_product_trends(raw_trend_df)
        except Exception as exc:
            cleaned_trend_preview = pd.DataFrame()
            st.error(f"Product trend data cannot be cleaned yet: {exc}")
        quality = product_trend_quality_summary(raw_trend_df)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Raw Rows", len(raw_trend_df))
        c2.metric("Cleaned Rows", len(cleaned_trend_preview))
        c3.metric("Data Quality Score", f"{quality['score']}/100")
        c4.metric("Missing Values", quality["missing_values"])
        c5.metric("Duplicate Rows", quality["duplicate_rows"])
        st.caption(f"Invalid months: {quality['invalid_months']}")

        st.subheader("Column Quality")
        ui_dataframe(
            pd.DataFrame(
                {
                    "column": raw_trend_df.columns,
                    "missing_values": raw_trend_df.isna().sum().values,
                    "missing_percent": (raw_trend_df.isna().mean().values * 100).round(2),
                }
            ),
            width="stretch",
        )

        if st.button("Run Product Trend Cleaning"):
            try:
                cleaned_trend_df = clean_product_trends(raw_trend_df)
                PRODUCT_TRENDS_PATH.parent.mkdir(parents=True, exist_ok=True)
                cleaned_trend_df.to_csv(PRODUCT_TRENDS_PATH, index=False)
                log_activity(
                    db,
                    "product_trend_cleaning_completed",
                    st.session_state["user"]["username"],
                    {"records": len(cleaned_trend_df), "quality_score": quality["score"]},
                )
                st.success(f"Cleaned {len(cleaned_trend_df)} product trend rows.")
                st.subheader("Cleaned Product Trend Table")
                st.caption("This cleaned product trend table is saved to the product trend CSV and used by the Supplier Dashboard.")
                ui_dataframe(cleaned_trend_df, width="stretch")
            except Exception as exc:
                st.error(str(exc))
        return

    st.subheader("Supplier")
    raw_df = load_collection(db, "raw_orders")
    if raw_df.empty:
        st.warning("Upload supplier data first.")
        return

    quality = data_quality_summary(raw_df)
    try:
        cleaned_preview = clean_orders(raw_df)
    except Exception as exc:
        cleaned_preview = pd.DataFrame()
        st.error(f"Supplier data cannot be cleaned yet: {exc}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Raw Rows", len(raw_df))
    c2.metric("Cleaned Rows", len(cleaned_preview))
    c3.metric("Data Quality Score", f"{quality['score']}/100")
    c4.metric("Missing Values", quality["missing_values"])
    c5.metric("Duplicate Orders", quality["duplicate_orders"])
    st.caption(f"Invalid dates: {quality['invalid_dates']}")

    st.subheader("Column Quality")
    ui_dataframe(
        pd.DataFrame(
            {
                "column": raw_df.columns,
                "missing_values": raw_df.isna().sum().values,
                "missing_percent": (raw_df.isna().mean().values * 100).round(2),
            }
        ),
        width="stretch",
    )
    if st.button("Run Supplier Cleaning and Refresh Metrics"):
        try:
            clean_df = clean_orders(raw_df)
            count = replace_collection_from_dataframe(db, COLLECTIONS["cleaned_orders"], clean_df)
            metrics_df = refresh_metrics(db)
            log_activity(db, "cleaning_completed", st.session_state["user"]["username"], {"records": count, "quality_score": quality["score"]})
            st.success(f"Cleaned {count} supplier order rows and refreshed {len(metrics_df)} supplier metric rows.")
            st.subheader("Cleaned Supplier Data Table")
            st.caption("This cleaned supplier order table is saved in MongoDB and used to calculate supplier metrics.")
            ui_dataframe(clean_df, width="stretch")
        except Exception as exc:
            st.error(str(exc))
    return


def page_view_data(db):
    page_header("View Data")
    mode_col1, mode_col2 = st.columns(2)
    if mode_col1.button("Supplier", use_container_width=True):
        st.session_state["view_data_mode"] = "Supplier"
    if mode_col2.button("Product Trend", use_container_width=True):
        st.session_state["view_data_mode"] = "Product Trend"
    view_mode = st.session_state.get("view_data_mode", "Supplier")

    if view_mode == "Product Trend":
        trends_df = load_product_trends()
        st.subheader("Product Trend")
        if trends_df.empty:
            st.info("No product trend dataset found. Use Upload Data > Product Trend first.")
            return
        c1, c2, c3 = st.columns(3)
        category = c1.selectbox("Category", ["All"] + sorted(trends_df["product_category"].dropna().unique()), key="view_product_category")
        product = c2.selectbox("Product", ["All"] + sorted(trends_df["product_name"].dropna().unique()), key="view_product_name")
        trend_level_options = ["All"] + sorted(trends_df["trend_level"].dropna().unique()) if "trend_level" in trends_df.columns else ["All"]
        trend_level = c3.selectbox("Trend Level", trend_level_options, key="view_product_trend_level")

        filtered = trends_df.copy()
        if category != "All":
            filtered = filtered[filtered["product_category"] == category]
        if product != "All":
            filtered = filtered[filtered["product_name"] == product]
        if trend_level != "All" and "trend_level" in filtered.columns:
            filtered = filtered[filtered["trend_level"] == trend_level]

        ui_dataframe(filtered, width="stretch")
        st.download_button("Export Product Trend CSV", filtered.to_csv(index=False), "product_trends_report.csv", "text/csv")

        st.divider()
        st.subheader("DELETE")
        st.warning("Use this only when you want to reset product trend data and upload a new product trend dataset.")
        st.download_button(
            "Download Backup ZIP",
            build_product_trend_backup_zip(),
            file_name=f"product_trend_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
        )
        product_backup_saved = st.checkbox("I have saved the backup", key="product_trend_backup_saved")
        product_confirm_delete = st.text_input("Type DELETE to confirm reset", key="product_trend_confirm_delete")
        if st.button(
            "Delete All Product Trend Data",
            type="primary",
            disabled=not (product_backup_saved and product_confirm_delete == "DELETE"),
        ):
            if PRODUCT_TRENDS_PATH.exists():
                PRODUCT_TRENDS_PATH.unlink()
            log_activity(db, "product_trend_data_deleted", st.session_state["user"]["username"], {"backup_confirmed": True})
            st.success("Product trend data was deleted.")
            st.rerun()
        return

    metrics_df = load_supplier_metrics(db)

    if not metrics_df.empty:
        c1, c2, c3, c4 = st.columns(4)
        category = c1.selectbox("Category", ["All"] + sorted(metrics_df["product_category"].dropna().unique()))
        supplier = c2.selectbox("Supplier", ["All"] + sorted(metrics_df["supplier"].dropna().unique()))
        risk = c3.selectbox("Risk", ["All", "Low", "Medium", "High"])
        min_rating = c4.slider("Minimum Rating", 1.0, 5.0, 1.0, 0.1)
        filtered = metrics_df[metrics_df["final_rating"] >= min_rating].copy()
        if category != "All":
            filtered = filtered[filtered["product_category"] == category]
        if supplier != "All":
            filtered = filtered[filtered["supplier"] == supplier]
        if risk != "All":
            filtered = filtered[filtered["risk_level"] == risk]
        st.subheader("All Supplier Metrics")
        ui_dataframe(filtered, width="stretch")
        st.download_button("Export Supplier Metrics CSV", filtered.to_csv(index=False), "supplier_metrics_report.csv", "text/csv")
    else:
        st.info("No supplier metrics found. Upload and clean data first.")

    st.divider()
    st.subheader("DELETE")
    st.warning("Use this only when you want to reset all supplier data and upload a new dataset.")
    st.download_button(
        "Download Backup ZIP",
        build_backup_zip(db),
        file_name=f"supplier_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
    )
    backup_saved = st.checkbox("I have saved the backup")
    confirm_delete = st.text_input("Type DELETE to confirm reset")
    if st.button("Delete All Supplier Data", type="primary", disabled=not (backup_saved and confirm_delete == "DELETE")):
        clear_supplier_data(db)
        log_activity(db, "supplier_data_deleted", st.session_state["user"]["username"], {"backup_confirmed": True})
        st.success("Supplier data, cleaned data, metrics, user ratings, and recommendation history were deleted.")
        st.rerun()


def page_eda(db):
    page_header("EDA & KPI Analysis")
    mode_col1, mode_col2 = st.columns(2)
    if mode_col1.button("Supplier", use_container_width=True):
        st.session_state["eda_mode"] = "Supplier"
    if mode_col2.button("Product Trend", use_container_width=True):
        st.session_state["eda_mode"] = "Product Trend"
    eda_mode = st.session_state.get("eda_mode", "Supplier")

    if eda_mode == "Product Trend":
        trends_df = load_product_trends()
        st.subheader("Product Trend")
        if trends_df.empty:
            st.info("No product trend dataset found. Use Upload Data > Product Trend first.")
            return

        latest_month = trends_df["month"].max()
        latest_df = trends_df[trends_df["month"] == latest_month].copy()
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Trend Rows", len(trends_df))
        p2.metric("Categories", trends_df["product_category"].nunique())
        p3.metric("Products", trends_df["product_name"].nunique())
        p4.metric("Latest Month", latest_month)

        chart1, chart2 = st.columns(2)
        with chart1:
            st.plotly_chart(
                px.bar(
                    latest_df.sort_values("trend_score", ascending=False),
                    x="product_name",
                    y="trend_score",
                    color="product_category",
                    title="Latest Product Trend Score",
                ),
                width="stretch",
            )
        with chart2:
            if "trend_level" in trends_df.columns:
                trend_counts = latest_df.groupby(["product_category", "trend_level"], as_index=False).size()
                st.plotly_chart(
                    px.bar(
                        trend_counts,
                        x="product_category",
                        y="size",
                        color="trend_level",
                        title="High / Medium / Low Trend Count by Category",
                    ),
                    width="stretch",
                )

        chart3, chart4 = st.columns(2)
        with chart3:
            st.plotly_chart(
                px.line(
                    trends_df.sort_values("month"),
                    x="month",
                    y="trend_score",
                    color="product_name",
                    facet_col="product_category",
                    facet_col_wrap=2,
                    title="Product Trend Score Over Time",
                ),
                width="stretch",
            )
        with chart4:
            avg_growth = latest_df.groupby("product_category", as_index=False)["growth_rate"].mean()
            st.plotly_chart(
                px.bar(avg_growth, x="product_category", y="growth_rate", title="Average Latest Growth Rate by Category"),
                width="stretch",
            )

        return

    clean_df = load_collection(db, "cleaned_orders")
    metrics_df = load_supplier_metrics(db)
    ratings_df = load_collection(db, "supplier_ratings")
    if clean_df.empty or metrics_df.empty:
        st.warning("Clean data first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.bar(metrics_df.nlargest(10, "supplier_rank_score"), x="supplier", y="supplier_rank_score", color="product_category", title="Top 10 Suppliers by Score"), width="stretch")
    with col2:
        st.plotly_chart(px.bar(metrics_df.nlargest(10, "risk_score"), x="supplier", y="risk_score", color="risk_level", title="Highest Risk Suppliers"), width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        delay = clean_df.groupby("product_category", as_index=False)["delay_days"].mean()
        st.plotly_chart(px.bar(delay, x="product_category", y="delay_days", title="Average Delay by Category"), width="stretch")
    with col4:
        risk_counts = metrics_df.groupby(["product_category", "risk_level"], as_index=False).size()
        st.plotly_chart(px.bar(risk_counts, x="product_category", y="size", color="risk_level", title="Risk Count by Category"), width="stretch")

    if not ratings_df.empty:
        col5, col6 = st.columns(2)
        with col5:
            rating_avg = ratings_df.groupby("supplier", as_index=False)["rating"].mean().sort_values("rating", ascending=False)
            st.plotly_chart(px.bar(rating_avg.head(10), x="supplier", y="rating", title="Average User Rating by Supplier"), width="stretch")
        with col6:
            events = ratings_df.groupby("event_type", as_index=False).size().sort_values("size", ascending=False)
            st.plotly_chart(px.bar(events, x="event_type", y="size", title="Feedback Event Summary"), width="stretch")


def monthly_category_trend(clean_df, category):
    df = clean_df[clean_df["product_category"] == category].copy()
    if df.empty:
        return pd.DataFrame()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])
    for col in ["quantity_ordered", "delay_days", "supply_risk_flag", "has_disruption"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["month"] = df["order_date"].dt.to_period("M").astype(str)
    return (
        df.groupby("month", as_index=False)
        .agg(
            total_orders=("order_id", "count"),
            total_quantity=("quantity_ordered", "sum"),
            avg_delay=("delay_days", "mean"),
            risk_count=("supply_risk_flag", "sum"),
            disruption_count=("has_disruption", "sum"),
        )
        .round(2)
    )


def predict_next_demand(trend_df):
    if trend_df.empty:
        return 0, "Stable", "Not enough category order data for prediction."
    latest = float(trend_df["total_quantity"].iloc[-1])
    if len(trend_df) == 1:
        return round(latest, 2), "Stable", "Only one month of data is available, so the prediction uses the latest demand."
    previous = float(trend_df["total_quantity"].iloc[-2])
    avg_change = trend_df["total_quantity"].diff().dropna().mean()
    predicted = float(max(0, latest + avg_change))
    growth = ((latest - previous) / previous * 100) if previous else 0
    direction = "Increasing" if growth > 5 else "Decreasing" if growth < -5 else "Stable"
    reason = f"Latest month demand changed by {round(growth, 2)}% compared with the previous month."
    return round(predicted, 2), direction, reason


def predict_supplier_future_risk(metric_row, trend_df):
    current_risk = float(metric_row.get("risk_score", 0))
    delay_change = 0
    disruption_change = 0
    if len(trend_df) >= 2:
        delay_change = float(trend_df["avg_delay"].iloc[-1] - trend_df["avg_delay"].iloc[-2])
        disruption_change = float(trend_df["disruption_count"].iloc[-1] - trend_df["disruption_count"].iloc[-2])
    predicted = current_risk + max(delay_change, 0) * 4 + max(disruption_change, 0) * 2
    if metric_row.get("trend_status") == "Declining":
        predicted += 8
    elif metric_row.get("trend_status") == "Improving":
        predicted -= 5
    predicted = round(float(np.clip(predicted, 0, 100)), 2)
    reasons = []
    if delay_change > 0:
        reasons.append("category average delay increased")
    if disruption_change > 0:
        reasons.append("category disruptions increased")
    if metric_row.get("trend_status") == "Declining":
        reasons.append("recent user rating trend is declining")
    if not reasons:
        reasons.append("recent category trend is stable")
    return predicted, risk_level(predicted), ", ".join(reasons)


def apply_what_if(metric_row, simulated_delay, simulated_rating, simulated_disruption, simulated_reliability):
    current_score = float(metric_row.get("supplier_rank_score", 0))
    current_delay = float(metric_row.get("avg_delay", 0))
    current_disruption = float(metric_row.get("disruption_frequency", 0))
    current_reliability = float(metric_row.get("reliability", 0))
    current_rating = float(metric_row.get("user_rating", 0))

    simulated_risk = (
        ((1 - simulated_reliability) * 35)
        + (min(simulated_delay, 10) / 10 * 25)
        + (simulated_disruption * 25)
        + ((5 - simulated_rating) / 4 * 15)
    )
    current_comparable_risk = (
        ((1 - current_reliability) * 35)
        + (min(current_delay, 10) / 10 * 25)
        + (current_disruption * 25)
        + ((5 - current_rating) / 4 * 15)
    )
    risk_after = round(float(np.clip(simulated_risk, 0, 100)), 2)
    current_comparable_risk = round(float(np.clip(current_comparable_risk, 0, 100)), 2)
    score_after = current_score + (current_comparable_risk - risk_after) * 0.55 + (simulated_rating - current_rating) * 2
    score_after = round(float(np.clip(score_after, 0, 100)), 2)
    scenario_details = {
        "score_change": round(score_after - current_score, 2),
        "risk_change": round(risk_after - current_comparable_risk, 2),
        "current_comparable_risk": current_comparable_risk,
        "simulated_delay": round(simulated_delay, 2),
        "simulated_disruption_frequency": round(simulated_disruption, 2),
        "simulated_reliability": round(simulated_reliability, 2),
        "simulated_rating": round(simulated_rating, 2),
    }
    return score_after, risk_after, risk_level(risk_after), scenario_details


def load_product_trends():
    if not PRODUCT_TRENDS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(PRODUCT_TRENDS_PATH)
    for col in ["search_volume", "sales_count", "growth_rate", "trend_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["month"] = df["month"].astype(str)
    df["product_category"] = df["product_category"].astype(str).str.title()
    df["product_name"] = df["product_name"].astype(str)
    return df


def category_product_trends(category):
    trends_df = load_product_trends()
    if trends_df.empty:
        return pd.DataFrame()
    return trends_df[trends_df["product_category"] == category].copy()


def current_trending_products(product_df):
    if product_df.empty:
        return pd.DataFrame()
    latest_month = product_df["month"].max()
    latest = product_df[product_df["month"] == latest_month].copy()
    return latest.sort_values(["trend_score", "growth_rate", "sales_count"], ascending=False)


def future_trending_products(product_df):
    if product_df.empty:
        return pd.DataFrame()
    rows = []
    for product_name, group in product_df.sort_values("month").groupby("product_name"):
        latest = group.iloc[-1]
        avg_change = group["trend_score"].diff().dropna().tail(3).mean()
        if pd.isna(avg_change):
            avg_change = 0
        predicted_score = float(np.clip(latest["trend_score"] + avg_change, 0, 100))
        trend_direction = "Uptrend" if avg_change > 1 else "Downtrend" if avg_change < -1 else "Stable"
        rows.append(
            {
                "product_name": product_name,
                "product_category": latest["product_category"],
                "latest_month": latest["month"],
                "current_trend_score": round(float(latest["trend_score"]), 2),
                "recent_growth_rate": round(float(latest["growth_rate"]), 2),
                "current_sales_count": int(latest["sales_count"]),
                "predicted_next_trend_score": round(predicted_score, 2),
                "trend_direction": trend_direction,
                "trend_score_change": round(float(avg_change), 2),
                "prediction_reason": "Trend score is rising recently" if trend_direction == "Uptrend" else "Trend score is falling recently" if trend_direction == "Downtrend" else "Trend score is stable",
            }
        )
    return pd.DataFrame(rows).sort_values("predicted_next_trend_score", ascending=False)


def get_supplier_page_context(db):
    user = st.session_state["user"]
    supplier_id = user.get("supplier_id") or user["username"].upper()
    clean_df = load_collection(db, "cleaned_orders")
    metrics_df = load_supplier_metrics(db)
    ratings_df = load_collection(db, "supplier_ratings")
    if clean_df.empty or metrics_df.empty:
        st.warning("Admin must upload and clean data first.")
        return None

    supplier_metrics = metrics_df[metrics_df["supplier"] == supplier_id].copy()
    if supplier_metrics.empty:
        st.warning(f"No supplier metrics found for {supplier_id}.")
        return None

    category_options = sorted(supplier_metrics["product_category"].dropna().unique())
    category = st.selectbox("Your Product Category", category_options)
    metric_row = supplier_metrics[supplier_metrics["product_category"] == category].iloc[0]
    category_df = clean_df[clean_df["product_category"] == category].copy()
    category_metrics = metrics_df[metrics_df["product_category"] == category].copy()
    trend_df = monthly_category_trend(clean_df, category)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Supplier", supplier_id)
    c2.metric("Category", category)
    c3.metric("Current Risk", metric_row["risk_level"])
    c4.metric("Score", f"{metric_row['supplier_rank_score']}/100")
    c5.metric("Rating", f"{metric_row['final_rating']}/5")

    return {
        "supplier_id": supplier_id,
        "category": category,
        "clean_df": clean_df,
        "metrics_df": metrics_df,
        "ratings_df": ratings_df,
        "metric_row": metric_row,
        "category_df": category_df,
        "category_metrics": category_metrics,
        "trend_df": trend_df,
    }


def render_category_demand_charts(category, trend_df):
    st.subheader("Product Demand Chart")
    if trend_df.empty:
        st.info("No monthly trend data is available for this category.")
        return
    t1, t2 = st.columns(2)
    with t1:
        st.plotly_chart(px.line(trend_df, x="month", y="total_quantity", markers=True, title=f"{category} Monthly Quantity Demand"), width="stretch")
    with t2:
        st.plotly_chart(px.bar(trend_df, x="month", y="total_orders", title=f"{category} Monthly Order Count"), width="stretch")


def render_supplier_benchmark(supplier_id, category, metric_row, category_metrics):
    st.subheader(f"{supplier_id} vs {category} Average")
    benchmark = pd.DataFrame(
        {
            "Metric": ["Average Delay", "Final Rating", "Risk Score", "Reliability"],
            supplier_id: [
                metric_row["avg_delay"],
                metric_row["final_rating"],
                metric_row["risk_score"],
                metric_row["reliability"],
            ],
            f"{category} Average": [
                category_metrics["avg_delay"].mean(),
                category_metrics["final_rating"].mean(),
                category_metrics["risk_score"].mean(),
                category_metrics["reliability"].mean(),
            ],
        }
    ).round(2)
    ui_dataframe(benchmark, width="stretch")
    st.plotly_chart(px.bar(benchmark, x="Metric", y=[supplier_id, f"{category} Average"], barmode="group", title=f"{supplier_id} vs {category} Average"), width="stretch")


def render_supplier_feedback(supplier_id, category, ratings_df):
    st.subheader("Supplier Feedback")
    supplier_ratings = ratings_df[(ratings_df["supplier"] == supplier_id) & (ratings_df["product_category"] == category)] if not ratings_df.empty else pd.DataFrame()
    if supplier_ratings.empty:
        st.info("No user feedback for this supplier/category yet.")
        return supplier_ratings, 0, 0
    ui_dataframe(
        safe_metric_table(
            supplier_ratings.sort_values("created_at", ascending=False),
            ["created_at", "username", "rating", "event_type", "comment"],
        ),
        width="stretch",
    )
    avg_feedback = round(pd.to_numeric(supplier_ratings["rating"], errors="coerce").mean(), 2)
    return supplier_ratings, len(supplier_ratings), avg_feedback


def page_supplier_dashboard(db):
    page_header("Supplier Dashboard")
    context = get_supplier_page_context(db)
    if context is None:
        return

    supplier_id = context["supplier_id"]
    category = context["category"]
    metric_row = context["metric_row"]
    trend_df = context["trend_df"]
    category_metrics = context["category_metrics"]
    ratings_df = context["ratings_df"]

    render_category_demand_charts(category, trend_df)
    render_supplier_benchmark(supplier_id, category, metric_row, category_metrics)
    _, feedback_count, avg_feedback = render_supplier_feedback(supplier_id, category, ratings_df)

    st.subheader("Supplier Dashboard Summary")
    predicted_demand, demand_direction, _ = predict_next_demand(trend_df)
    predicted_risk, predicted_risk_level, _ = predict_supplier_future_risk(metric_row, trend_df)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Predicted Demand", predicted_demand, demand_direction)
    s2.metric("Future Risk", predicted_risk_level, f"{predicted_risk}/100")
    s3.metric("Avg Feedback", f"{avg_feedback}/5", f"{feedback_count} ratings")
    s4.metric("Current Score", f"{metric_row['supplier_rank_score']}/100")


def page_supplier_trend(db):
    page_header("Supplier Trend")
    context = get_supplier_page_context(db)
    if context is None:
        return

    category = context["category"]
    product_df = category_product_trends(category)
    current_products = current_trending_products(product_df)
    future_products = future_trending_products(product_df)

    st.subheader("Specific Product Trend Analysis")
    if product_df.empty:
        st.info("No product trend dataset found for this category.")
        return

    latest_month = product_df["month"].max()
    current_top = current_products.iloc[0]
    future_top = future_products.sort_values("predicted_next_trend_score", ascending=False).iloc[0]
    downtrend_products = future_products.sort_values("trend_score_change", ascending=True)
    future_down = downtrend_products.iloc[0]
    pt1, pt2, pt3, pt4 = st.columns(4)
    pt1.metric("Latest Trend Month", latest_month)
    pt2.metric("Trending Now", current_top["product_name"], f"{current_top['trend_score']}/100")
    pt3.metric("Predicted Future Trend", future_top["product_name"], f"{future_top['predicted_next_trend_score']}/100")
    pt4.metric("Predicted Downtrend", future_down["product_name"], f"{future_down['trend_score_change']} score change")

    product_chart = product_df.sort_values("month")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.plotly_chart(
            px.line(product_chart, x="month", y="trend_score", color="product_name", markers=True, title=f"{category} Product Trend Score Over Time"),
            width="stretch",
        )
    with pc2:
        st.plotly_chart(
            px.bar(current_products, x="product_name", y="sales_count", color="growth_rate", title=f"{category} Current Product Demand"),
            width="stretch",
        )

    st.subheader("Current Trending Products")
    ui_dataframe(safe_metric_table(current_products, ["product_name", "search_volume", "sales_count", "growth_rate", "trend_score"]), width="stretch")
    st.subheader("Future Product Trend Prediction")
    ui_dataframe(
        safe_metric_table(
            future_products,
            ["product_name", "current_trend_score", "recent_growth_rate", "current_sales_count", "predicted_next_trend_score", "trend_direction", "trend_score_change", "prediction_reason"],
        ),
        width="stretch",
    )
    st.subheader("Predicted Downtrend Products")
    ui_dataframe(
        safe_metric_table(
            downtrend_products,
            ["product_name", "current_trend_score", "recent_growth_rate", "predicted_next_trend_score", "trend_direction", "trend_score_change", "prediction_reason"],
        ),
        width="stretch",
    )


def page_supplier_future_prediction(db):
    page_header("Future Prediction")
    context = get_supplier_page_context(db)
    if context is None:
        return

    category = context["category"]
    metric_row = context["metric_row"]
    trend_df = context["trend_df"]

    st.subheader("Category Risk Trend")
    if not trend_df.empty:
        r1, r2 = st.columns(2)
        with r1:
            st.plotly_chart(px.line(trend_df, x="month", y="avg_delay", markers=True, title=f"{category} Average Delay Trend"), width="stretch")
        with r2:
            st.plotly_chart(px.bar(trend_df, x="month", y="risk_count", title=f"{category} Risk Flag Count"), width="stretch")
    else:
        st.info("No monthly trend data is available for this category.")

    st.subheader("Future Prediction")
    predicted_demand, demand_direction, demand_reason = predict_next_demand(trend_df)
    predicted_risk, predicted_risk_level, risk_reason = predict_supplier_future_risk(metric_row, trend_df)
    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted Next Demand", predicted_demand)
    p2.metric("Demand Trend", demand_direction)
    p3.metric("Predicted Future Risk", predicted_risk_level, f"{predicted_risk}/100")
    st.write(f"Demand reason: {demand_reason}")
    st.write(f"Risk reason: {risk_reason}.")
    st.info(
        "Risk level is based on the supplier risk score. Low means 0-34, Medium means 35-64, and High means 65-100. "
        "The score uses reliability, average delay, disruption frequency, supply risk flags, disruption severity, user rating, and bad feedback."
    )

    st.subheader("What-If Supplier Improvement")
    w1, w2 = st.columns(2)
    current_delay = float(metric_row.get("avg_delay", 0))
    current_rating = float(metric_row.get("user_rating", 0))
    current_disruption = float(metric_row.get("disruption_frequency", 0))
    current_reliability = float(metric_row.get("reliability", 0))
    simulated_delay = w1.slider("What if average delay becomes days", 0.0, 10.0, min(10.0, current_delay), 0.5)
    simulated_disruption = w2.slider("What if disruption frequency becomes", 0.0, 1.0, min(1.0, current_disruption), 0.05)
    simulated_rating = w1.slider("What if user rating becomes", 1.0, 5.0, min(5.0, max(1.0, current_rating)), 0.1)
    simulated_reliability = w2.slider("What if reliability becomes", 0.0, 1.0, min(1.0, max(0.0, current_reliability)), 0.05)
    score_after, risk_after, risk_after_level, scenario_details = apply_what_if(
        metric_row,
        simulated_delay,
        simulated_rating,
        simulated_disruption,
        simulated_reliability,
    )
    score_delta = scenario_details["score_change"]
    risk_delta = scenario_details["risk_change"]
    result_score_label = "What-If Score"
    result_risk_label = "What-If Risk"
    st.caption(
        f"Current values: delay {round(current_delay, 2)} days, disruption {round(current_disruption, 2)}, "
        f"reliability {round(current_reliability, 2)}, rating {round(current_rating, 2)}. "
        f"What-if values: delay {scenario_details['simulated_delay']} days, disruption {scenario_details['simulated_disruption_frequency']}, "
        f"reliability {scenario_details['simulated_reliability']}, rating {scenario_details['simulated_rating']}."
    )
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Current Score", f"{metric_row['supplier_rank_score']}/100")
    a2.metric(result_score_label, f"{score_after}/100", score_delta)
    a3.metric("Current Risk", f"{risk_level(scenario_details['current_comparable_risk'])} ({scenario_details['current_comparable_risk']}/100)")
    a4.metric(result_risk_label, f"{risk_after_level} ({risk_after}/100)", risk_delta)
    st.info(
        "What-if risk is recalculated from the slider values. More delay, more disruption, lower reliability, or lower rating increases risk. "
        "Less delay, less disruption, higher reliability, or higher rating reduces risk."
    )
    st.plotly_chart(
        px.bar(
            pd.DataFrame(
                {
                    "Metric": ["Current Score", result_score_label, "Current Risk", result_risk_label],
                    "Value": [metric_row["supplier_rank_score"], score_after, scenario_details["current_comparable_risk"], risk_after],
                }
            ),
            x="Metric",
            y="Value",
            title="Current vs What-If Result",
        ),
        width="stretch",
    )

    st.plotly_chart(
        px.bar(
            pd.DataFrame(
                {
                    "Metric": ["Score", "Score", "Risk", "Risk"],
                    "Scenario": ["Current", "What-If", "Current", "What-If"],
                    "Value": [metric_row["supplier_rank_score"], score_after, scenario_details["current_comparable_risk"], risk_after],
                }
            ),
            x="Metric",
            y="Value",
            color="Scenario",
            barmode="group",
            title="Summary: Current vs What-If",
        ),
        width="stretch",
    )


def page_ratings_feedback(db):
    page_header("User Rating")
    ratings_df = load_collection(db, "supplier_ratings")
    st.subheader("User Rating")
    if ratings_df.empty:
        st.info("No user ratings submitted yet.")
    else:
        activity = ratings_df.sort_values("created_at", ascending=False).copy()
        ui_dataframe(
            safe_metric_table(
                activity,
                ["created_at", "username", "supplier", "product_category", "rating", "event_type", "comment"],
            ),
            width="stretch",
        )
        st.download_button("Export Ratings CSV", ratings_df.to_csv(index=False), "supplier_ratings_report.csv", "text/csv")


def user_account_table(db):
    users = dataframe_from_collection(db, COLLECTIONS["users"], {"role": "user"})
    if users.empty:
        return users
    logs = load_collection(db, "recommendation_logs")
    ratings = load_collection(db, "supplier_ratings")
    selected_counts = pd.DataFrame(columns=["username", "selected_supplier_count"])
    rating_counts = pd.DataFrame(columns=["username", "ratings_given_count"])
    if not logs.empty and {"username", "status"}.issubset(logs.columns):
        selected_counts = (
            logs[logs["status"] == "selected"]
            .groupby("username", as_index=False)
            .size()
            .rename(columns={"size": "selected_supplier_count"})
        )
    if not ratings.empty and "username" in ratings.columns:
        rating_counts = ratings.groupby("username", as_index=False).size().rename(columns={"size": "ratings_given_count"})
    users = users.merge(selected_counts, on="username", how="left").merge(rating_counts, on="username", how="left")
    users["selected_supplier_count"] = users["selected_supplier_count"].fillna(0).astype(int)
    users["ratings_given_count"] = users["ratings_given_count"].fillna(0).astype(int)
    return safe_metric_table(users, ["username", "is_active", "created_at", "selected_supplier_count", "ratings_given_count"])


def supplier_account_table(db):
    suppliers = dataframe_from_collection(db, COLLECTIONS["users"], {"role": "supplier"})
    if suppliers.empty:
        return suppliers
    ratings = load_collection(db, "supplier_ratings")
    rows = []
    for _, account in suppliers.iterrows():
        supplier_id = str(account.get("supplier_id", "")).upper()
        info = supplier_id_info(db, supplier_id, current_username=account.get("username"))
        supplier_ratings = ratings[ratings["supplier"] == supplier_id] if not ratings.empty and "supplier" in ratings.columns else pd.DataFrame()
        avg_rating = 0
        if not supplier_ratings.empty and "rating" in supplier_ratings.columns:
            avg_rating = round(pd.to_numeric(supplier_ratings["rating"], errors="coerce").mean(), 2)
        rows.append(
            {
                "username": account.get("username"),
                "supplier_id": supplier_id,
                "is_active": account.get("is_active", False),
                "account_status": account.get("account_status", "pending"),
                "supplier_id_exists": info["supplier_id_exists"],
                "order_count": info["order_count"],
                "categories": info["categories"],
                "already_claimed": info["already_claimed"],
                "created_at": account.get("created_at"),
                "feedback_count": len(supplier_ratings),
                "avg_rating": avg_rating,
            }
        )
    return pd.DataFrame(rows)


def update_account_username(db, old_username, new_username, actor):
    new_username = new_username.strip()
    if len(new_username) < 3:
        return False, "Username must be at least 3 characters."
    if db[COLLECTIONS["users"]].find_one({"username": new_username}):
        return False, "Username already exists."
    db[COLLECTIONS["users"]].update_one({"username": old_username}, {"$set": {"username": new_username}})
    log_activity(db, "account_username_updated", actor, {"old_username": old_username, "new_username": new_username})
    return True, "Username updated."


def update_account_password(db, username, new_password, actor):
    if len(new_password) < 6:
        return False, "Password must be at least 6 characters."
    db[COLLECTIONS["users"]].update_one({"username": username}, {"$set": {"password_hash": password_hash(new_password)}})
    log_activity(db, "account_password_reset", actor, {"username": username})
    return True, "Password reset."


def page_manage_users(db):
    page_header("Manage Accounts")
    actor = st.session_state["user"]["username"]
    mode_col1, mode_col2 = st.columns(2)
    if mode_col1.button("User Accounts", use_container_width=True):
        st.session_state["manage_account_mode"] = "User Accounts"
    if mode_col2.button("Supplier Accounts", use_container_width=True):
        st.session_state["manage_account_mode"] = "Supplier Accounts"
    mode = st.session_state.get("manage_account_mode", "User Accounts")

    if mode == "Supplier Accounts":
        supplier_df = supplier_account_table(db)
        pending_df = supplier_df[supplier_df["account_status"] == "pending"].copy() if not supplier_df.empty else pd.DataFrame()
        managed_supplier_df = supplier_df[supplier_df["account_status"] != "pending"].copy() if not supplier_df.empty else pd.DataFrame()
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Supplier Accounts", len(supplier_df))
        s2.metric("Pending Requests", len(pending_df))
        s3.metric("Approved", int((supplier_df["account_status"] == "approved").sum()) if not supplier_df.empty else 0)
        s4.metric("Active", int((supplier_df["is_active"] == True).sum()) if not supplier_df.empty else 0)

        st.subheader("Supplier Accounts")
        if supplier_df.empty:
            st.info("No supplier accounts yet.")
        else:
            ui_dataframe(supplier_df, width="stretch")

        if not managed_supplier_df.empty:
            st.subheader("Manage Supplier Account")
            selected_supplier_user = st.selectbox("Select supplier account", managed_supplier_df["username"].tolist(), key="selected_supplier_account")
            selected_doc = db[COLLECTIONS["users"]].find_one({"username": selected_supplier_user}, {"_id": 0})
            current_supplier_id = selected_doc.get("supplier_id", "") if selected_doc else ""
            info = supplier_id_info(db, current_supplier_id, current_username=selected_supplier_user)

            u1, u2 = st.columns(2)
            new_supplier_username = u1.text_input("Update username", value=selected_supplier_user, key="supplier_new_username")
            if u1.button("Update Supplier Username"):
                success, message = update_account_username(db, selected_supplier_user, new_supplier_username, actor)
                st.success(message) if success else st.error(message)
                if success:
                    st.rerun()
            new_supplier_password = u2.text_input("New password", type="password", key="supplier_new_password")
            if u2.button("Reset Supplier Password"):
                success, message = update_account_password(db, selected_supplier_user, new_supplier_password, actor)
                st.success(message) if success else st.error(message)

            a1, a2 = st.columns(2)
            if a1.button("Activate"):
                db[COLLECTIONS["users"]].update_one({"username": selected_supplier_user}, {"$set": {"is_active": True}})
                log_activity(db, "supplier_account_activated", actor, {"username": selected_supplier_user})
                st.success("Supplier account activated.")
                st.rerun()
            if a2.button("Deactivate"):
                db[COLLECTIONS["users"]].update_one({"username": selected_supplier_user}, {"$set": {"is_active": False}})
                log_activity(db, "supplier_account_deactivated", actor, {"username": selected_supplier_user})
                st.success("Supplier account deactivated.")
                st.rerun()

            st.subheader("Supplier Activity")
            ratings = load_collection(db, "supplier_ratings")
            supplier_ratings = ratings[ratings["supplier"] == current_supplier_id] if not ratings.empty and "supplier" in ratings.columns else pd.DataFrame()
            if supplier_ratings.empty:
                st.info("No supplier feedback activity.")
            else:
                ui_dataframe(safe_metric_table(supplier_ratings.sort_values("created_at", ascending=False), ["created_at", "username", "supplier", "product_category", "rating", "event_type", "comment"]), width="stretch")
        else:
            st.info("No approved or rejected supplier accounts to manage yet.")

        st.subheader("Create Supplier Account")
        with st.form("admin_create_supplier_account"):
            supplier_username = st.text_input("Supplier username")
            supplier_password = st.text_input("Supplier password", type="password")
            supplier_id = st.text_input("Supplier ID", placeholder="Example: S10").strip().upper()
            create_supplier_submitted = st.form_submit_button("Create Approved Supplier")
        if create_supplier_submitted:
            info = supplier_id_info(db, supplier_id)
            if db[COLLECTIONS["users"]].find_one({"username": supplier_username.strip()}):
                st.error("Username already exists.")
            elif len(supplier_password) < 6:
                st.error("Password must be at least 6 characters.")
            elif info["supplier_id_exists"] != "Yes" or info["order_count"] <= 0 or info["category_match"] != "Yes":
                st.error("Supplier ID verification failed.")
            elif info["already_claimed"] == "Yes":
                st.error("This supplier ID is already claimed by another approved supplier account.")
            else:
                db[COLLECTIONS["users"]].insert_one(
                    {
                        "username": supplier_username.strip(),
                        "password_hash": password_hash(supplier_password),
                        "role": "supplier",
                        "supplier_id": supplier_id,
                        "is_active": True,
                        "account_status": "approved",
                        "created_at": datetime.now(timezone.utc),
                    }
                )
                log_activity(db, "supplier_account_created_by_admin", actor, {"username": supplier_username.strip(), "supplier_id": supplier_id})
                st.success("Supplier account created and approved.")
                st.rerun()

        if not pending_df.empty:
            st.subheader("Approval")
            pending_supplier_user = st.selectbox("Select pending supplier request", pending_df["username"].tolist(), key="pending_supplier_account")
            pending_doc = db[COLLECTIONS["users"]].find_one({"username": pending_supplier_user}, {"_id": 0})
            pending_supplier_id = pending_doc.get("supplier_id", "") if pending_doc else ""
            pending_info = supplier_id_info(db, pending_supplier_id, current_username=pending_supplier_user)
            selected_status = pending_doc.get("account_status", "pending") if pending_doc else "pending"
            selected_active = "Active" if pending_doc and pending_doc.get("is_active", False) else "Inactive"
            st.info(
                f"Pending account: {pending_supplier_user} | Supplier ID: {pending_supplier_id} | "
                f"Status: {selected_status} | Account: {selected_active}"
            )
            ap1, ap2, ap3, ap4, ap5 = st.columns(5)
            ap1.metric("Supplier ID Exists", pending_info["supplier_id_exists"])
            ap2.metric("Order History", pending_info["order_count"])
            ap3.metric("Category Match", pending_info["category_match"])
            ap4.metric("Already Claimed", pending_info["already_claimed"])
            ap5.metric("Claimed By", pending_info["claimed_by"])
            st.caption(f"Categories: {pending_info['categories']}")
            a1, a2 = st.columns(2)
            if a1.button("Approve"):
                if pending_info["supplier_id_exists"] == "Yes" and pending_info["order_count"] > 0 and pending_info["category_match"] == "Yes" and pending_info["already_claimed"] == "No":
                    db[COLLECTIONS["users"]].update_one({"username": pending_supplier_user}, {"$set": {"account_status": "approved", "is_active": True}})
                    log_activity(db, "supplier_account_approved", actor, {"username": pending_supplier_user, "supplier_id": pending_supplier_id})
                    st.success("Supplier account approved.")
                    st.rerun()
                else:
                    st.error(f"Cannot approve. This supplier ID is already claimed by {pending_info['claimed_by']}.")
            if a2.button("Reject"):
                db[COLLECTIONS["users"]].update_one({"username": pending_supplier_user}, {"$set": {"account_status": "rejected", "is_active": False}})
                log_activity(db, "supplier_account_rejected", actor, {"username": pending_supplier_user, "supplier_id": pending_supplier_id})
                st.success("Supplier account rejected.")
                st.rerun()
            if pending_info["already_claimed"] == "Yes":
                st.warning(f"{pending_supplier_id} is already approved for {pending_info['claimed_by']}. Use replace only if this pending account is the real supplier.")
                if st.button("Approve and Replace Existing Claim"):
                    db[COLLECTIONS["users"]].update_many(
                        {
                            "role": "supplier",
                            "supplier_id": pending_supplier_id,
                            "account_status": "approved",
                            "username": {"$ne": pending_supplier_user},
                        },
                        {"$set": {"account_status": "replaced", "is_active": False}},
                    )
                    db[COLLECTIONS["users"]].update_one(
                        {"username": pending_supplier_user},
                        {"$set": {"account_status": "approved", "is_active": True}},
                    )
                    log_activity(
                        db,
                        "supplier_account_approved_replacing_claim",
                        actor,
                        {"username": pending_supplier_user, "supplier_id": pending_supplier_id, "previous_claim": pending_info["claimed_by"]},
                    )
                    st.success("Supplier account approved and previous claim was deactivated.")
                    st.rerun()
        else:
            st.subheader("Approval")
            st.info("No pending supplier requests to approve.")

        st.subheader("Supplier Verification Codes")
        vc1, vc2 = st.columns([2, 1])
        supplier_id_for_code = vc1.text_input("Supplier ID for code", placeholder="Example: S10").strip().upper()
        if vc2.button("Generate Code", use_container_width=True):
            info = supplier_id_info(db, supplier_id_for_code)
            if not supplier_id_for_code:
                st.error("Enter supplier ID first.")
            elif info["supplier_id_exists"] != "Yes":
                st.error("Supplier ID not found in uploaded supplier data.")
            else:
                code = save_supplier_verification_code(db, supplier_id_for_code, actor)
                st.success(f"Verification code for {supplier_id_for_code}: {code}")

        codes_df = dataframe_from_collection(db, COLLECTIONS["supplier_verification_codes"])
        if not codes_df.empty:
            ui_dataframe(safe_metric_table(codes_df, ["supplier_id", "verification_code", "is_used", "used_by", "created_at", "updated_at"]), width="stretch")

        st.subheader("Pending Supplier Requests")
        if pending_df.empty:
            st.info("No pending supplier requests.")
        else:
            ui_dataframe(
                safe_metric_table(
                    pending_df,
                    ["username", "supplier_id", "supplier_id_exists", "order_count", "categories", "already_claimed", "account_status", "created_at"],
                ),
                width="stretch",
            )
        return

    users_df = user_account_table(db)
    u1, u2, u3 = st.columns(3)
    u1.metric("User Accounts", len(users_df))
    u2.metric("Active Users", int((users_df["is_active"] == True).sum()) if not users_df.empty else 0)
    u3.metric("Ratings Given", int(users_df["ratings_given_count"].sum()) if not users_df.empty else 0)

    st.subheader("User Accounts")
    if users_df.empty:
        st.info("No user accounts yet.")
    else:
        ui_dataframe(users_df, width="stretch")

    if not users_df.empty:
        st.subheader("Manage User Accounts")
        selected = st.selectbox("Select user account", users_df["username"].tolist(), key="selected_user_account")
        c1, c2 = st.columns(2)
        new_username = c1.text_input("Update username", value=selected, key="new_user_username")
        if c1.button("Update Username"):
            success, message = update_account_username(db, selected, new_username, actor)
            st.success(message) if success else st.error(message)
            if success:
                st.rerun()
        new_password = c2.text_input("New password", type="password", key="new_user_password")
        if c2.button("Reset Password"):
            success, message = update_account_password(db, selected, new_password, actor)
            st.success(message) if success else st.error(message)

        a1, a2 = st.columns(2)
        if a1.button("Activate User"):
            db[COLLECTIONS["users"]].update_one({"username": selected}, {"$set": {"is_active": True}})
            log_activity(db, "user_account_activated", actor, {"username": selected})
            st.success("User account activated.")
            st.rerun()
        if a2.button("Deactivate User"):
            db[COLLECTIONS["users"]].update_one({"username": selected}, {"$set": {"is_active": False}})
            log_activity(db, "user_account_deactivated", actor, {"username": selected})
            st.success("User account deactivated.")
            st.rerun()

        st.subheader("User Activity")
        logs = dataframe_from_collection(db, COLLECTIONS["recommendation_logs"], {"username": selected})
        ratings = dataframe_from_collection(db, COLLECTIONS["supplier_ratings"], {"username": selected})
        act1, act2 = st.columns(2)
        with act1:
            st.caption("Selected Suppliers")
            selected_logs = logs[logs["status"] == "selected"] if not logs.empty and "status" in logs.columns else pd.DataFrame()
            ui_dataframe(safe_metric_table(selected_logs.sort_values("created_at", ascending=False) if not selected_logs.empty else selected_logs, ["supplier", "product_category", "final_score", "risk_level", "created_at"]), width="stretch")
        with act2:
            st.caption("Ratings Given")
            ui_dataframe(safe_metric_table(ratings.sort_values("created_at", ascending=False) if not ratings.empty else ratings, ["supplier", "product_category", "rating", "event_type", "comment", "created_at"]), width="stretch")

    st.subheader("Create New Account")
    with st.form("admin_create_user_account"):
        username = st.text_input("New username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Create User")
    if submitted:
        success, message = create_account(db, username, password, password, "user")
        st.success(message) if success else st.error(message)
        if success:
            st.rerun()


def page_user_home(db):
    metrics_df = load_supplier_metrics(db)
    if metrics_df.empty:
        st.warning("No supplier data available yet.")
        return
    if "supplier_rank_score" not in metrics_df.columns:
        st.warning("Supplier metrics need to be refreshed by admin from Clean Data.")
        return
    st.markdown('<div style="height: 1.25rem;"></div>', unsafe_allow_html=True)
    title_col, search_col = st.columns([1.55, 1])
    search_text = search_col.text_input("Search", placeholder="Example: S10, Machinery, Food").strip()
    if not search_text:
        title_col.markdown(
            """
            <div class="inline-page-title">
                <h1>🏠 Best Suppliers</h1>
                <p>Quickly view top suppliers or search by supplier ID and category.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        title_col.markdown(
            """
            <div class="inline-page-title">
                <h1>🔎 Search Results</h1>
                <p>Select or favourite matching suppliers from your search.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    result_columns = ["supplier", "product_category", "final_rating", "risk_level", "supplier_rank_score", "avg_delay", "avg_unit_price", "trend_status"]
    username = st.session_state["user"]["username"]

    if search_text:
        query = search_text.lower()
        supplier_match = metrics_df["supplier"].astype(str).str.lower() == query
        category_match = metrics_df["product_category"].astype(str).str.lower() == query
        if supplier_match.any():
            search_results = metrics_df[supplier_match].copy().sort_values("product_category")
        elif category_match.any():
            search_results = metrics_df[category_match].copy().sort_values("supplier_rank_score", ascending=False)
        else:
            search_results = metrics_df[
                metrics_df["supplier"].astype(str).str.lower().str.contains(query, na=False)
                | metrics_df["product_category"].astype(str).str.lower().str.contains(query, na=False)
            ].copy().sort_values("supplier_rank_score", ascending=False)

        if search_results.empty:
            st.info("No matching supplier or category found.")
        else:
            st.subheader("Select Supplier")
            st.caption("Choose one supplier from the search results below.")
            current_hot = hot_supplier_keys(db, username)
            h1, h2, h3, h4, h5, h6, h7, h8, h9 = st.columns([1, 1.4, 1, 1, 1, 1, 1, 0.8, 0.5])
            h1.caption("Supplier")
            h2.caption("Category")
            h3.caption("Rating")
            h4.caption("Risk")
            h5.caption("Score")
            h6.caption("Delay")
            h7.caption("Price")
            h8.caption("Select")
            h9.caption("Fav")
            for _, row in search_results.iterrows():
                c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns([1, 1.4, 1, 1, 1, 1, 1, 0.8, 0.5])
                c1.write(str(row["supplier"]))
                c2.write(str(row["product_category"]))
                c3.write(f"{row.get('final_rating', 0)}/5")
                c4.write(str(row.get("risk_level", "")))
                c5.write(f"{row.get('supplier_rank_score', 0)}/100")
                c6.write(f"{row.get('avg_delay', 0)} days")
                c7.write(row.get("avg_unit_price", 0))
                if c8.button("Select", key=f"home_select_{row['supplier']}_{row['product_category']}"):
                    save_selected_supplier(db, username, row, "home_search")
                    st.success(f"Selected {row['supplier']} for {row['product_category']}.")
                is_favourite = (str(row["supplier"]), str(row["product_category"])) in current_hot
                star_label = "★" if is_favourite else "☆"
                star_help = "Remove from favourite supplier" if is_favourite else "Add to favourite supplier"
                if c9.button(star_label, key=f"home_fav_{row['supplier']}_{row['product_category']}", help=star_help):
                    if is_favourite:
                        remove_hot_supplier(db, username, row["supplier"], row["product_category"])
                        log_activity(db, "favourite_supplier_removed", username, {"supplier": row["supplier"], "category": row["product_category"], "source": "home_search"})
                        st.success(f"{row['supplier']} removed from favourite supplier.")
                    else:
                        save_hot_supplier(db, username, row)
                        log_activity(db, "favourite_supplier_saved", username, {"supplier": row["supplier"], "category": row["product_category"], "source": "home_search"})
                        st.success(f"{row['supplier']} saved as a favourite supplier.")
                    st.rerun()
        return
    else:
        best = metrics_df.sort_values("supplier_rank_score", ascending=False).groupby("product_category", as_index=False).first()
        ui_dataframe(
            safe_metric_table(best, ["product_category", "supplier", "final_rating", "risk_level", "supplier_rank_score", "trend_status"]),
            width="stretch",
        )

    st.subheader("Favourite Supplier")
    hot_df = dataframe_from_collection(db, HOT_SUPPLIERS_COLLECTION, {"username": st.session_state["user"]["username"]})
    if hot_df.empty:
        st.info("No favourite supplier yet. Use the star button in Find Supplier to save one.")
    else:
        if "created_at" in hot_df.columns:
            hot_df = hot_df.sort_values("created_at", ascending=False)
        fh1, fh2, fh3, fh4, fh5, fh6, fh7, fh8 = st.columns([1, 1.4, 1, 1, 1, 1, 0.9, 0.5])
        fh1.caption("Supplier")
        fh2.caption("Category")
        fh3.caption("Score")
        fh4.caption("Rating")
        fh5.caption("Risk")
        fh6.caption("Price")
        fh7.caption("Select")
        fh8.caption("Fav")
        for _, row in hot_df.iterrows():
            fc1, fc2, fc3, fc4, fc5, fc6, fc7, fc8 = st.columns([1, 1.4, 1, 1, 1, 1, 0.9, 0.5])
            fc1.write(str(row.get("supplier")))
            fc2.write(str(row.get("product_category")))
            fc3.write(f"{row.get('final_score', 0)}/100")
            fc4.write(f"{row.get('final_rating', 0)}/5")
            fc5.write(str(row.get("risk_level", "")))
            fc6.write(row.get("avg_unit_price", 0))
            if fc7.button("Select", key=f"home_reselect_{row.get('supplier')}_{row.get('product_category')}"):
                save_selected_supplier(db, username, row, "favourite_reselect")
                st.success(f"Selected {row.get('supplier')} for {row.get('product_category')}.")
            if fc8.button("★", key=f"home_remove_fav_{row.get('supplier')}_{row.get('product_category')}", help="Remove from favourite supplier"):
                remove_hot_supplier(db, username, row.get("supplier"), row.get("product_category"))
                log_activity(db, "favourite_supplier_removed", username, {"supplier": row.get("supplier"), "category": row.get("product_category"), "source": "home"})
                st.success(f"{row.get('supplier')} removed from favourite supplier.")
                st.rerun()


def page_find_supplier(db):
    page_header("Find Supplier")
    metrics_df = load_supplier_metrics(db)
    if metrics_df.empty:
        st.warning("Admin must clean data first.")
        return
    categories = sorted(metrics_df["product_category"].dropna().unique())
    with st.form("recommendation_form"):
        c1, c2 = st.columns(2)
        category = c1.selectbox("Product Category", categories)
        quantity = c2.number_input("Quantity", min_value=1, value=1000, step=100)
        budget = c1.number_input("Budget per Unit USD", min_value=0.0, value=100.0, step=5.0)
        min_quality = c2.slider("Quality Rating (Supplier)", 1.0, 5.0, 4.0, 0.1)
        deadline = c1.number_input("Deadline Days", min_value=1, value=14, step=1)
        priority = c2.selectbox("Priority", list(PRIORITY_WEIGHTS.keys()))
        top_n = st.slider("Number of Suppliers", 3, 10, 3)
        submitted = st.form_submit_button("Recommend Suppliers")

    if submitted:
        results = recommend_suppliers(metrics_df, category, quantity, budget, min_quality, deadline, priority, top_n)
        log_doc = {
            "username": st.session_state["user"]["username"],
            "category": category,
            "quantity": quantity,
            "budget": budget,
            "min_quality": min_quality,
            "deadline": deadline,
            "priority": priority,
            "result_count": len(results),
            "selected_supplier": None,
            "status": "recommended",
            "created_at": datetime.now(timezone.utc),
        }
        inserted = db[COLLECTIONS["recommendation_logs"]].insert_one(log_doc)
        st.session_state["last_recommendation_id"] = str(inserted.inserted_id)
        st.session_state["last_results"] = results.to_dict("records")
        if results.empty:
            st.error("No supplier satisfies all requirements.")
            category_options = metrics_df[metrics_df["product_category"].str.lower() == category.lower()].copy()
            if not category_options.empty:
                st.info("Closest available suppliers in this category are shown below. Try increasing budget/deadline or lowering quality.")
                ui_dataframe(
                    safe_metric_table(
                        category_options.sort_values("supplier_rank_score", ascending=False),
                        ["supplier", "product_category", "avg_unit_price", "avg_delay", "quality_rating", "final_rating", "risk_level", "supplier_rank_score"],
                    ),
                    width="stretch",
                )
            return
        st.success(f"Found {len(results)} recommended suppliers.")

    results = pd.DataFrame(st.session_state.get("last_results", []))
    if not results.empty:
        st.subheader("Recommendation Results")
        username = st.session_state["user"]["username"]
        current_hot = hot_supplier_keys(db, username)
        for _, row in results.iterrows():
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Supplier", row["supplier"])
                c2.metric("Final Score", f"{row['final_score']}/100")
                c3.metric("Risk", f"{row['risk_level']} ({row['risk_score']}/100)")
                c4.metric("Rating", f"{row['final_rating']}/5")
                st.write(row["explanation"])
                st.caption(
                    f"Details: category {row['product_category']} | average delay {row['avg_delay']} days | "
                    f"unit price {row['avg_unit_price']} | user rating {row['user_rating']}/5 | trend {row['trend_status']}"
                )
                select_col, hot_col = st.columns([4, 1])
                if select_col.button(f"Select {row['supplier']} for {row['product_category']}", key=f"select_{row['supplier']}_{row['product_category']}"):
                    selection = {
                        "username": st.session_state["user"]["username"],
                        "supplier": row["supplier"],
                        "product_category": row["product_category"],
                        "final_score": row["final_score"],
                        "risk_level": row["risk_level"],
                        "status": "selected",
                        "created_at": datetime.now(timezone.utc),
                    }
                    db[COLLECTIONS["recommendation_logs"]].insert_one(selection)
                    log_activity(db, "supplier_selected", st.session_state["user"]["username"], selection)
                    st.session_state["selected_supplier"] = selection
                    st.success(f"Selected {row['supplier']}. You can rate it after the experience.")
                is_favourite = (str(row["supplier"]), str(row["product_category"])) in current_hot
                star_label = "★" if is_favourite else "☆"
                star_help = "Remove from favourite supplier" if is_favourite else "Add to favourite supplier"
                if hot_col.button(star_label, key=f"hot_{row['supplier']}_{row['product_category']}", help=star_help):
                    if is_favourite:
                        remove_hot_supplier(db, username, row["supplier"], row["product_category"])
                        log_activity(db, "favourite_supplier_removed", username, {"supplier": row["supplier"], "category": row["product_category"]})
                        st.success(f"{row['supplier']} removed from favourite supplier.")
                    else:
                        save_hot_supplier(db, username, row)
                        log_activity(db, "favourite_supplier_saved", username, {"supplier": row["supplier"], "category": row["product_category"]})
                        st.success(f"{row['supplier']} saved as a favourite supplier.")
                    st.rerun()

        st.subheader("Supplier Comparison")
        ui_dataframe(
            safe_metric_table(results, ["supplier", "product_category", "final_score", "final_rating", "user_rating", "risk_level", "avg_delay", "avg_unit_price", "trend_status"]),
            width="stretch",
        )


def page_rate_supplier(db):
    page_header("Rate Supplier")
    selected = st.session_state.get("selected_supplier")
    history = load_collection(db, "recommendation_logs")
    user = st.session_state["user"]["username"]
    required_history_cols = {"username", "status", "supplier", "product_category"}
    if not history.empty and required_history_cols.issubset(history.columns):
        selected_rows = history[(history["username"] == user) & (history["status"] == "selected")]
    else:
        selected_rows = pd.DataFrame()
    options = []
    if selected:
        options.append(f"{selected['supplier']} | {selected['product_category']}")
    if not selected_rows.empty:
        options.extend((selected_rows["supplier"] + " | " + selected_rows["product_category"]).dropna().unique().tolist())
    options = sorted(set(options))
    if not options:
        st.warning("Select a supplier from recommendations before rating.")
        return

    with st.form("rating_form"):
        choice = st.selectbox("Selected supplier", options)
        supplier, category = [part.strip() for part in choice.split("|", 1)]
        rating = st.slider("Rating", 1, 5, 4)
        event_type = st.selectbox("Event Type", EVENT_TYPES)
        comment = st.text_area("Comment")
        submitted = st.form_submit_button("Submit Rating")
    if submitted:
        doc = {
            "username": user,
            "supplier": supplier,
            "product_category": category,
            "rating": rating,
            "event_type": event_type,
            "comment": comment,
            "created_at": datetime.now(timezone.utc),
        }
        db[COLLECTIONS["supplier_ratings"]].insert_one(doc)
        log_activity(db, "supplier_rated", user, {"supplier": supplier, "category": category, "rating": rating, "event_type": event_type})
        metrics_df = refresh_metrics(db)
        sync_supplier_saved_scores(db, supplier, category, metrics_df)
        st.success("Rating saved. Supplier score and future recommendations now use this feedback.")


def page_user_history(db):
    page_header("My History")
    user = st.session_state["user"]["username"]
    logs = dataframe_from_collection(db, COLLECTIONS["recommendation_logs"], {"username": user})
    ratings = dataframe_from_collection(db, COLLECTIONS["supplier_ratings"], {"username": user})
    favourites = dataframe_from_collection(db, HOT_SUPPLIERS_COLLECTION, {"username": user})
    metrics_df = load_supplier_metrics(db)

    def add_supplier_details(history_df, use_rank_as_final_score=False):
        if history_df.empty or metrics_df.empty:
            return history_df
        metric_cols = [
            "supplier",
            "product_category",
            "supplier_rank_score",
            "final_rating",
            "risk_level",
            "risk_score",
            "avg_delay",
            "avg_unit_price",
        ]
        available_metric_cols = [col for col in metric_cols if col in metrics_df.columns]
        details = metrics_df[available_metric_cols].copy()
        enriched = history_df.merge(details, on=["supplier", "product_category"], how="left", suffixes=("", "_metric"))
        for col in ["final_rating", "risk_level", "risk_score", "avg_delay", "avg_unit_price"]:
            metric_col = f"{col}_metric"
            if metric_col in enriched.columns:
                if col in enriched.columns:
                    enriched[col] = enriched[col].combine_first(enriched[metric_col])
                else:
                    enriched[col] = enriched[metric_col]
                enriched = enriched.drop(columns=[metric_col])
        if use_rank_as_final_score and "supplier_rank_score" in enriched.columns:
            enriched["final_score"] = enriched["supplier_rank_score"]
        return enriched.drop(columns=["supplier_rank_score"], errors="ignore")


    st.subheader("Selected Suppliers")
    if logs.empty or "status" not in logs.columns:
        st.info("No selected suppliers yet.")
    else:
        selected_logs = logs[logs["status"] == "selected"].copy()
        if selected_logs.empty:
            st.info("No selected suppliers yet.")
        else:
            selected_logs = add_supplier_details(selected_logs, use_rank_as_final_score=True)
            if "created_at" in selected_logs.columns:
                selected_logs = selected_logs.sort_values("created_at", ascending=False)
            ui_dataframe(
                safe_metric_table(
                    selected_logs,
                    ["supplier", "product_category", "final_score", "final_rating", "risk_level", "risk_score", "avg_delay", "avg_unit_price", "created_at"],
                ),
                width="stretch",
            )
    st.subheader("Ratings Given")
    if ratings.empty:
        st.info("No ratings given yet.")
    else:
        ratings = add_supplier_details(ratings, use_rank_as_final_score=True)
        if "created_at" in ratings.columns:
            ratings = ratings.sort_values("created_at", ascending=False)
        ui_dataframe(
            safe_metric_table(
                ratings,
                ["supplier", "product_category", "rating", "event_type", "comment", "created_at"],
            ),
            width="stretch",
        )


def render_app():
    inject_custom_css()
    db = get_database()
    if db is None:
        st.error("MongoDB is not connected. Start MongoDB or check MONGODB_URI in .env.")
        st.stop()
    ensure_default_users(db)

    if "user" not in st.session_state:
        if not restore_login_session(db):
            page_login(db)
            return

    user = st.session_state["user"]
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="brand-icon">🚚</div>
                <div>
                    <div class="brand-name">Supply<span>Logix</span></div>
                    <div class="brand-subtitle">Supplier Intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"{user['username']} ({user['role']})")
        if st.button("Logout"):
            clear_login_session(db)
            st.session_state.clear()
            st.rerun()
        if user["role"] == "admin":
            st.markdown('<div class="sidebar-role-badge">🛠️ Admin Pages<small>Data control and analysis</small></div>', unsafe_allow_html=True)
            pages = ["Dashboard", "Upload Data", "Clean Data", "View Data", "EDA & KPI", "User Rating", "Manage Accounts"]
            page_icons = {
                "Dashboard": "🏠 Dashboard",
                "Upload Data": "📤 Upload Data",
                "Clean Data": "🧹 Clean Data",
                "View Data": "📋 View Data",
                "EDA & KPI": "📈 EDA & KPI",
                "User Rating": "⭐ User Rating",
                "Manage Accounts": "👥 Manage Accounts",
            }
            if st.session_state.get("admin_nav_target") in pages:
                st.session_state["admin_page"] = st.session_state.pop("admin_nav_target")
            if st.session_state.get("admin_page") not in pages:
                st.session_state["admin_page"] = "Dashboard"
            st.radio("Admin Pages", pages, key="admin_page", format_func=lambda page_name: page_icons.get(page_name, page_name))
            page = st.session_state["admin_page"]
        elif user["role"] == "supplier":
            st.markdown('<div class="sidebar-role-badge">🚚 Supplier Pages<small>Performance and prediction</small></div>', unsafe_allow_html=True)
            supplier_pages = ["Supplier Dashboard", "Supplier Trend", "Future Prediction"]
            supplier_icons = {
                "Supplier Dashboard": "📊 Dashboard",
                "Supplier Trend": "📈 Trend",
                "Future Prediction": "🔮 Future Prediction",
            }
            page = st.radio("Supplier Pages", supplier_pages, format_func=lambda page_name: supplier_icons.get(page_name, page_name))
        else:
            st.markdown('<div class="sidebar-role-badge">👤 User Pages<small>Find, rate, and review</small></div>', unsafe_allow_html=True)
            user_pages = ["Home", "Find Supplier", "Rate Supplier", "My History"]
            user_icons = {
                "Home": "🏠 Home",
                "Find Supplier": "🔎 Find Supplier",
                "Rate Supplier": "⭐ Rate Supplier",
                "My History": "🕘 My History",
            }
            page = st.radio("User Pages", user_pages, format_func=lambda page_name: user_icons.get(page_name, page_name))

    if page == "Dashboard":
        page_admin_dashboard(db)
    elif page == "Upload Data":
        page_upload(db)
    elif page == "Clean Data":
        page_clean(db)
    elif page == "View Data":
        page_view_data(db)
    elif page == "EDA & KPI":
        page_eda(db)
    elif page == "User Rating":
        page_ratings_feedback(db)
    elif page == "Manage Accounts":
        page_manage_users(db)
    elif page == "Supplier Dashboard":
        page_supplier_dashboard(db)
    elif page == "Supplier Trend":
        page_supplier_trend(db)
    elif page == "Future Prediction":
        page_supplier_future_prediction(db)
    elif page == "Home":
        page_user_home(db)
    elif page == "Find Supplier":
        page_find_supplier(db)
    elif page == "Rate Supplier":
        page_rate_supplier(db)
    elif page == "My History":
        page_user_history(db)


if __name__ == "__main__":
    render_app()

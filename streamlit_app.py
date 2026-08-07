import hashlib
import io
import zipfile
from datetime import datetime, timezone

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
]
HOT_SUPPLIERS_COLLECTION = COLLECTIONS.get("hot_suppliers", "hot_suppliers")

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
        {"username": "admin", "password": "admin123", "role": "admin"},
        {"username": "user", "password": "user123", "role": "user"},
    ]
    for account in default_users:
        users.update_one(
            {"username": account["username"]},
            {
                "$set": {
                    "password_hash": password_hash(account["password"]),
                    "role": account["role"],
                    "is_active": True,
                },
                "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
            },
            upsert=True,
        )


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


def recommend_suppliers(metrics_df, category, quantity, budget, min_quality, deadline, priority, top_n):
    if metrics_df.empty:
        return metrics_df
    options = metrics_df[metrics_df["product_category"].str.lower() == category.lower()].copy()
    if options.empty:
        return options

    for col in ["avg_unit_price", "quality_rating", "avg_delay", "total_quantity"]:
        options[col] = pd.to_numeric(options[col], errors="coerce").fillna(0)
    if budget > 0:
        options = options[options["avg_unit_price"] <= budget]
    options = options[(options["quality_rating"] >= min_quality) & (options["avg_delay"] <= deadline)]
    if options.empty:
        return options

    options["cost_match"] = np.where(budget > 0, ((budget - options["avg_unit_price"]) / budget * 100).clip(0, 100), 70)
    options["deadline_match"] = ((deadline - options["avg_delay"]) / max(deadline, 1) * 100).clip(0, 100)
    options["quality_match"] = (options["quality_rating"] / 5 * 100).clip(0, 100)
    options["quantity_match"] = np.where(options["total_quantity"] >= quantity, 100, options["total_quantity"] / max(quantity, 1) * 100)
    options["requirement_match_score"] = (
        options["cost_match"] * 0.30
        + options["deadline_match"] * 0.30
        + options["quality_match"] * 0.25
        + options["quantity_match"] * 0.15
    ).round(2)
    options["user_rating_score"] = (options["user_rating"] / 5 * 100).round(2)
    options["risk_prediction_score"] = (100 - options["risk_score"]).round(2)

    weights = PRIORITY_WEIGHTS[priority]
    options["final_score"] = (
        options["supplier_kpi_score"] * weights["kpi"]
        + options["user_rating_score"] * weights["rating"]
        + options["risk_prediction_score"] * weights["risk"]
        + options["requirement_match_score"] * weights["match"]
    ).round(2)
    options["explanation"] = options.apply(lambda row: explain_supplier(row, budget, deadline), axis=1)
    return options.sort_values("final_score", ascending=False).head(top_n)


def build_backup_zip(db):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for key in DATA_EXPORT_KEYS:
            df = load_collection(db, key)
            archive.writestr(f"{key}.csv", df.to_csv(index=False))
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
        "final_score": row.get("final_score"),
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


def remove_hot_supplier(db, username, supplier, product_category):
    db[HOT_SUPPLIERS_COLLECTION].delete_one(
        {"username": username, "supplier": supplier, "product_category": product_category}
    )


def page_login(db):
    st.title("Supplier Recommendation and Risk Analysis System")
    st.caption("MongoDB + Streamlit + data analysis + user feedback")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        user = db[COLLECTIONS["users"]].find_one(
            {"username": username.strip(), "password_hash": password_hash(password), "is_active": True}
        )
        if user:
            st.session_state["user"] = {"username": user["username"], "role": user["role"]}
            st.rerun()
        st.error("Invalid login or inactive account.")
    st.info("Demo accounts: admin/admin123 and user/user123")


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
    st.header("Admin Dashboard / Home")
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
    st.subheader("Clean Data Summary")
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
        st.dataframe(pd.DataFrame(alerts), width="stretch")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Supplier Summary")
        if not metrics_df.empty:
            st.dataframe(
                safe_metric_table(
                    metrics_df,
                    ["supplier", "product_category", "total_orders", "final_rating", "user_rating", "risk_level", "avg_delay", "trend_status"],
                ).head(12),
                width="stretch",
            )
            if st.button("Open Data Page"):
                st.session_state["admin_page"] = "View Data"
                st.rerun()
        else:
            st.info("No supplier metrics yet. Run cleaning first.")

    with col2:
        st.subheader("Rating Activity")
        st.metric("User Ratings", len(ratings_df))
        bad_feedback = 0
        if not ratings_df.empty and "rating" in ratings_df.columns:
            bad_feedback = int((pd.to_numeric(ratings_df["rating"], errors="coerce") <= 2).sum())
        st.metric("Bad Feedback", bad_feedback)


def page_upload(db):
    st.header("Upload Data")
    st.caption("Uploads append new rows. Existing order IDs are skipped.")
    uploaded = st.file_uploader("Upload supplier order CSV", type=["csv"])
    if uploaded:
        try:
            raw_df = pd.read_csv(uploaded, encoding="ISO-8859-1")
            st.dataframe(raw_df.head(20), width="stretch")
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
    if st.button("Append Local CSV to MongoDB"):
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
            st.dataframe(raw_df.head(20), width="stretch")
        except Exception as exc:
            st.error(f"Could not load this local CSV path: {exc}")


def page_clean(db):
    st.header("Clean Data")
    raw_df = load_collection(db, "raw_orders")
    clean_df_existing = load_collection(db, "cleaned_orders")
    if raw_df.empty:
        st.warning("Upload data first.")
        return

    quality = data_quality_summary(raw_df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Raw Rows", len(raw_df))
    c2.metric("Cleaned Rows", len(clean_df_existing))
    c3.metric("Data Quality Score", f"{quality['score']}/100")
    c4.metric("Missing Values", quality["missing_values"])
    c5.metric("Duplicate Orders", quality["duplicate_orders"])
    st.caption(f"Invalid dates: {quality['invalid_dates']}")

    st.subheader("Column Quality")
    st.dataframe(
        pd.DataFrame(
            {
                "column": raw_df.columns,
                "missing_values": raw_df.isna().sum().values,
                "missing_percent": (raw_df.isna().mean().values * 100).round(2),
            }
        ),
        width="stretch",
    )
    if st.button("Run Cleaning and Refresh Metrics"):
        try:
            clean_df = clean_orders(raw_df)
            count = replace_collection_from_dataframe(db, COLLECTIONS["cleaned_orders"], clean_df)
            metrics_df = refresh_metrics(db)
            log_activity(db, "cleaning_completed", st.session_state["user"]["username"], {"records": count, "quality_score": quality["score"]})
            st.success(f"Cleaned {count} records and refreshed {len(metrics_df)} supplier metric rows.")
            st.dataframe(clean_df.head(20), width="stretch")
        except Exception as exc:
            st.error(str(exc))
    return


def page_view_data(db):
    st.header("View Data")
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
        st.dataframe(filtered, width="stretch")
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
    st.header("EDA & KPI Analysis")
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

    st.subheader("Supplier KPI Table")
    st.dataframe(metrics_df, width="stretch")


def page_ratings_feedback(db):
    st.header("Ratings & Feedback")
    ratings_df = load_collection(db, "supplier_ratings")
    metrics_df = load_supplier_metrics(db)
    if metrics_df.empty:
        st.warning("No supplier metrics found.")
        return
    st.subheader("User Rating Activity")
    if ratings_df.empty:
        st.info("No user ratings submitted yet.")
    else:
        activity = ratings_df.sort_values("created_at", ascending=False).copy()
        st.dataframe(
            safe_metric_table(
                activity,
                ["created_at", "username", "supplier", "product_category", "rating", "event_type", "comment"],
            ),
            width="stretch",
        )
        st.download_button("Export Ratings CSV", ratings_df.to_csv(index=False), "supplier_ratings_report.csv", "text/csv")

    st.subheader("Category-Based Supplier Rating")
    st.dataframe(
        safe_metric_table(
            metrics_df,
            ["supplier", "product_category", "user_rating", "final_rating", "risk_level", "trend_status", "bad_feedback_count", "rating_count"],
        ),
        width="stretch",
    )


def page_manage_users(db):
    st.header("Manage Users")
    users = load_collection(db, "users")
    st.dataframe(users.drop(columns=["password_hash"], errors="ignore"), width="stretch")
    with st.form("create_user"):
        username = st.text_input("New username")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["user", "admin"])
        submitted = st.form_submit_button("Create User")
    if submitted:
        try:
            db[COLLECTIONS["users"]].insert_one(
                {
                    "username": username.strip(),
                    "password_hash": password_hash(password),
                    "role": role,
                    "is_active": True,
                    "created_at": datetime.now(timezone.utc),
                }
            )
            log_activity(db, "user_created", st.session_state["user"]["username"], {"username": username, "role": role})
            st.success("User created.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not create user: {exc}")
    if not users.empty:
        selected = st.selectbox("Select account", users["username"].tolist())
        c1, c2 = st.columns(2)
        if c1.button("Activate"):
            db[COLLECTIONS["users"]].update_one({"username": selected}, {"$set": {"is_active": True}})
            st.success("Account activated.")
        if c2.button("Deactivate"):
            db[COLLECTIONS["users"]].update_one({"username": selected}, {"$set": {"is_active": False}})
            st.success("Account deactivated.")


def page_user_home(db):
    st.header("Best Suppliers by Category")
    metrics_df = load_supplier_metrics(db)
    if metrics_df.empty:
        st.warning("No supplier data available yet.")
        return
    if "supplier_rank_score" not in metrics_df.columns:
        st.warning("Supplier metrics need to be refreshed by admin from Clean Data.")
        return
    best = metrics_df.sort_values("supplier_rank_score", ascending=False).groupby("product_category", as_index=False).first()
    st.dataframe(
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
        st.dataframe(
            safe_metric_table(
                hot_df,
                ["supplier", "product_category", "final_score", "final_rating", "risk_level", "risk_score", "avg_delay", "avg_unit_price", "created_at"],
            ),
            width="stretch",
        )


def page_find_supplier(db):
    st.header("Find Supplier")
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
        min_quality = c2.slider("Quality Requirement", 1.0, 5.0, 4.0, 0.1)
        deadline = c1.number_input("Deadline Days", min_value=1, value=14, step=1)
        priority = c2.selectbox("Priority", list(PRIORITY_WEIGHTS.keys()))
        top_n = st.slider("Top Suppliers", 3, 10, 3)
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
                st.dataframe(
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
        st.dataframe(
            safe_metric_table(results, ["supplier", "product_category", "final_score", "final_rating", "user_rating", "risk_level", "avg_delay", "avg_unit_price", "trend_status"]),
            width="stretch",
        )


def page_rate_supplier(db):
    st.header("Rate Supplier After Experience")
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
        refresh_metrics(db)
        st.success("Rating saved. Future recommendations now use this feedback.")


def page_user_history(db):
    st.header("My History")
    user = st.session_state["user"]["username"]
    logs = dataframe_from_collection(db, COLLECTIONS["recommendation_logs"], {"username": user})
    ratings = dataframe_from_collection(db, COLLECTIONS["supplier_ratings"], {"username": user})
    favourites = dataframe_from_collection(db, HOT_SUPPLIERS_COLLECTION, {"username": user})

    st.subheader("Favourite Supplier")
    if favourites.empty:
        st.info("No favourite supplier saved yet.")
    else:
        if "created_at" in favourites.columns:
            favourites = favourites.sort_values("created_at", ascending=False)
        st.dataframe(
            safe_metric_table(
                favourites,
                ["supplier", "product_category", "final_score", "final_rating", "risk_level", "risk_score", "avg_delay", "avg_unit_price", "created_at"],
            ),
            width="stretch",
        )
        for _, row in favourites.iterrows():
            c1, c2 = st.columns([5, 1])
            c1.write(f"{row.get('supplier')} | {row.get('product_category')}")
            if c2.button("★", key=f"history_unstar_{row.get('supplier')}_{row.get('product_category')}", help="Undo favourite supplier"):
                remove_hot_supplier(db, user, row.get("supplier"), row.get("product_category"))
                log_activity(db, "favourite_supplier_removed", user, {"supplier": row.get("supplier"), "category": row.get("product_category")})
                st.success(f"{row.get('supplier')} removed from favourite supplier.")
                st.rerun()

    st.subheader("Selections and Recommendations")
    st.dataframe(logs.sort_values("created_at", ascending=False) if not logs.empty else logs, width="stretch")
    st.subheader("Ratings Given")
    st.dataframe(ratings.sort_values("created_at", ascending=False) if not ratings.empty else ratings, width="stretch")


def render_app():
    db = get_database()
    if db is None:
        st.error("MongoDB is not connected. Start MongoDB or check MONGODB_URI in .env.")
        st.stop()
    ensure_default_users(db)

    if "user" not in st.session_state:
        page_login(db)
        return

    user = st.session_state["user"]
    with st.sidebar:
        st.title("Navigation")
        st.caption(f"{user['username']} ({user['role']})")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()
        if user["role"] == "admin":
            pages = ["Dashboard", "Upload Data", "Clean Data", "View Data", "EDA & KPI", "Ratings & Feedback"]
            default_page = st.session_state.get("admin_page", "Dashboard")
            page = st.radio("Admin Pages", pages, index=pages.index(default_page) if default_page in pages else 0)
            st.session_state["admin_page"] = page
        else:
            page = st.radio("User Pages", ["Home", "Find Supplier", "Rate Supplier", "My History"])

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
    elif page == "Ratings & Feedback":
        page_ratings_feedback(db)
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

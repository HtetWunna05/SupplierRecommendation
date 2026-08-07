import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "supplier_db")

COLLECTIONS = {
    "raw_orders": "raw_orders",
    "cleaned_orders": "cleaned_orders",
    "supplier_metrics": "supplier_metrics",
    "supplier_ratings": "supplier_ratings",
    "hot_suppliers": "hot_suppliers",
    "recommendation_logs": "recommendation_logs",
    "activity_logs": "activity_logs",
    "users": "users",
}


def connect_to_mongodb():
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client[MONGODB_DB_NAME]
    except Exception as exc:
        print(f"MongoDB connection failed: {exc}")
        return None


def ensure_indexes(db):
    db[COLLECTIONS["users"]].create_index("username", unique=True)
    db[COLLECTIONS["raw_orders"]].create_index("Order_ID", unique=True, sparse=True)
    db[COLLECTIONS["cleaned_orders"]].create_index("order_id", unique=True, sparse=True)
    db[COLLECTIONS["cleaned_orders"]].create_index([("supplier", 1), ("product_category", 1)])
    db[COLLECTIONS["supplier_metrics"]].create_index([("supplier", 1), ("product_category", 1)], unique=True)
    db[COLLECTIONS["supplier_ratings"]].create_index([("supplier", 1), ("product_category", 1), ("username", 1)])
    db[COLLECTIONS["hot_suppliers"]].create_index([("username", 1), ("supplier", 1), ("product_category", 1)], unique=True)
    db[COLLECTIONS["recommendation_logs"]].create_index([("username", 1), ("created_at", -1)])
    db[COLLECTIONS["activity_logs"]].create_index("created_at")


def create_database():
    db = connect_to_mongodb()
    if db is not None:
        ensure_indexes(db)
        print("MongoDB link is active and ready.")
    return db


def dataframe_from_collection(db, collection_name, query=None):
    records = list(db[collection_name].find(query or {}, {"_id": 0}))
    return pd.DataFrame(records)


def _mongo_safe(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def records_from_dataframe(dataframe):
    if dataframe.empty:
        return []
    return [
        {key: _mongo_safe(value) for key, value in record.items()}
        for record in dataframe.to_dict("records")
    ]


def replace_collection_from_dataframe(db, collection_name, dataframe):
    db[collection_name].delete_many({})
    records = records_from_dataframe(dataframe)
    if records:
        db[collection_name].insert_many(records, ordered=False)
    return len(records)


def append_dataframe_unique(db, collection_name, dataframe, unique_field):
    records = records_from_dataframe(dataframe)
    inserted = 0
    skipped = 0
    for record in records:
        key = record.get(unique_field)
        if key in (None, ""):
            skipped += 1
            continue
        result = db[collection_name].update_one(
            {unique_field: key},
            {"$setOnInsert": record},
            upsert=True,
        )
        if result.upserted_id is not None:
            inserted += 1
        else:
            skipped += 1
    return inserted, skipped


def log_activity(db, action, actor="system", details=None):
    db[COLLECTIONS["activity_logs"]].insert_one(
        {
            "action": action,
            "actor": actor,
            "details": details or {},
            "created_at": datetime.now(timezone.utc),
        }
    )


def clear_supplier_data(db):
    for key in [
        "raw_orders",
        "cleaned_orders",
        "supplier_metrics",
        "supplier_ratings",
        "hot_suppliers",
        "recommendation_logs",
    ]:
        db[COLLECTIONS[key]].delete_many({})


if __name__ == "__main__":
    create_database()

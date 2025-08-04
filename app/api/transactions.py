from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from db import get_db, Transaction
from datetime import datetime
from collections import Counter

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/transactions")
def list_transactions(request: Request):
    return templates.TemplateResponse("transactions.html", {"request": request})

@router.get("/transactions/data")
def transactions_data(db: Session = Depends(get_db)):
    """
    Endpoint to retrieve aggregated transaction data per minute, grouped by recommendation and recommendation type.
    Returns:
        JSONResponse: A JSON object containing:
            - Approve: List of [minute, count] for approved transactions.
            - DenyModel: List of [minute, count] for denied transactions (model-based).
            - DenyRule: List of [minute, count] for denied transactions (rule-based).
            - Trend: Dictionary with trend indicators for each group:
                2 (rising a lot), 1 (rising), 0 (stable), -1 (falling), -2 (falling a lot).
            - Overall: Total count for each group.
    Query Parameters:
        None
    Dependencies:
        db (Session): SQLAlchemy database session, injected by FastAPI Depends.
    Notes:
        - Only transactions from 2019 onwards are considered.
        - The endpoint aggregates data per minute, removing seconds and microseconds.
        - Trend is calculated using a sliding window approach.
    """
    # The data is from 2019, so adjust the start_time to that year
    start_time = datetime(2019, 1, 1)
    txs = db.query(Transaction).filter(Transaction.transaction_date >= start_time).all()

    # Group by recommendation and recommendation_type
    buckets = {
        "Approve": [],
        "DenyModel": [],
        "DenyRule": []
    }
    for tx in txs:
        minute = tx.transaction_date.replace(second=0, microsecond=0)
        if tx.recommendation == "approve":
            buckets["Approve"].append(minute)
        elif tx.recommendation == "deny":
            if tx.recommendation_type == "model_based":
                buckets["DenyModel"].append(minute)
            else:
                buckets["DenyRule"].append(minute)

    def count_per_minute(times):
        c = Counter(times)
        return sorted([[t.strftime("%Y-%m-%d %H:%M"), c[t]] for t in c])

    def calc_trend(data, window=5):
        """Returns 2 (rising a lot), 1 (rising), 0 (stable), -1 (falling), -2 (falling a lot)"""
        if len(data) < window * 2:
            return 0
        prev = sum([x[1] for x in data[-window*2:-window]])
        curr = sum([x[1] for x in data[-window:]])
        if prev == 0 and curr == 0:
            return 0
        if prev == 0 and curr > 0:
            return 2
        delta = curr - prev
        if delta > prev * 0.5:
            return 2
        elif delta > 0:
            return 1
        elif delta < -prev * 0.5:
            return -2
        elif delta < 0:
            return -1
        else:
            return 0
        
    approve_data = count_per_minute(buckets["Approve"])
    deny_model_data = count_per_minute(buckets["DenyModel"])
    deny_rule_data = count_per_minute(buckets["DenyRule"])

    data = {
        "Approve": count_per_minute(buckets["Approve"]),
        "DenyModel": count_per_minute(buckets["DenyModel"]),
        "DenyRule": count_per_minute(buckets["DenyRule"]),
        "Trend": {
            "Approve": calc_trend(approve_data),
            "DenyModel": calc_trend(deny_model_data),
            "DenyRule": calc_trend(deny_rule_data)
        },
        "Overall": {
            "Approve": len(buckets["Approve"]),
            "DenyModel": len(buckets["DenyModel"]),
            "DenyRule": len(buckets["DenyRule"])
        }
    }
    return JSONResponse(data)
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session
from schemas.transaction import TransactionIn, TransactionOut
from core.model_utils import predict_single, save_transaction_to_db
from db import get_db, Transaction
import numpy as np

router = APIRouter()

@router.post("/predict", response_model=TransactionOut)
def predict_transaction(payload: TransactionIn, db: Session = Depends(get_db)):

    """
    Endpoint to predict the outcome of a transaction using a fraud detection model and a set of business rules.
    Receives transaction data, applies the prediction model and enforces the following rules:
    1. Denies the transaction if the user has had a chargeback in the past week.
    2. Denies the transaction if the user has made 3 or more transactions in the last 5 minutes.
    3. Denies the transaction if the sum of transaction amounts in the last 3 hours plus the current transaction exceeds 2 standard deviations above the user's 30-day mean.
    4. Uses the model's prediction to approve or deny the transaction if none of the above rules apply.
    All transactions are saved to the database with the model's fraud probability, predicted class and the final recommendation.
    Args:
        payload (TransactionIn): Input transaction data.
        db (Session, optional): SQLAlchemy database session dependency.
    Returns:
        TransactionOut: Transaction ID and recommendation ("approve" or "deny").
    """

    tx_data = payload.model_dump()

    # applies the prediction model to the transaction data
    predict = predict_single(tx_data, db)

    # RULE: Check if user has ever had a chargeback
    # I'm considering a week window because as we have a simulated database, the transactions already have the has_cbk field
    cbk_cutoff_date = tx_data["transaction_date"] if isinstance(tx_data["transaction_date"], datetime) else datetime.fromisoformat(tx_data["transaction_date"])
    cbk_cutoff_date -= timedelta(weeks=1)
    user_has_cbk = db.query(Transaction).filter(
        Transaction.user_id == tx_data["user_id"],
        Transaction.has_cbk == True,
        Transaction.transaction_date <= cbk_cutoff_date
    ).first()
    if user_has_cbk:
        recommendation = "deny"
        save_transaction_to_db(tx_data, predict['fraud_probability'], predict['predicted_class'], recommendation, "rule_based", db)
        return TransactionOut(transaction_id=payload.transaction_id, recommendation=recommendation)


    # RULE: Check for rapid succession of transactions
    now = tx_data["transaction_date"] if isinstance(tx_data["transaction_date"], datetime) else datetime.fromisoformat(tx_data["transaction_date"])
    window_start = now - timedelta(minutes=5)
    tx_count = db.query(Transaction).filter(
        Transaction.user_id == tx_data["user_id"],
        Transaction.transaction_date >= window_start,
        Transaction.transaction_date < now
    ).count()
    if tx_count >= 3:
        recommendation = "deny"
        save_transaction_to_db(tx_data, predict['fraud_probability'], predict['predicted_class'], recommendation, "rule_based", db)
        return TransactionOut(transaction_id=payload.transaction_id, recommendation=recommendation)


    # RULE: Check for 2 standard deviations above user mean
    user_id = tx_data["user_id"]
    tx_amount = tx_data["transaction_amount"]

    # Find user historical mean (last N days)
    history_start = now - timedelta(days=30)
    user_transactions = db.query(Transaction).filter(
        Transaction.user_id == user_id,
        Transaction.transaction_date < now,
        Transaction.transaction_date >= history_start
    ).all()

    if user_transactions:

        values = np.array([t.transaction_amount for t in user_transactions])
        user_mean = values.mean()
        user_std = values.std(ddof=0) if len(values) > 1 else 0
        std_multiplier = 2
        user_limit = user_mean + std_multiplier * user_std

        std_multiplier = 2
        user_limit = user_mean + std_multiplier * user_std

        # Sum of transaction amounts in the last N hours
        window_start = now - timedelta(hours=3)
        total_in_window = db.query(func.sum(Transaction.transaction_amount)).filter(
            Transaction.user_id == user_id,
            Transaction.transaction_date >= window_start,
            Transaction.transaction_date < now
        ).scalar() or 0

        if (total_in_window + tx_amount) > user_limit:
            recommendation = "deny"
            save_transaction_to_db(tx_data, predict['fraud_probability'], predict['predicted_class'], recommendation, "rule_based", db)
            return TransactionOut(transaction_id=payload.transaction_id, recommendation=recommendation)

    # RULE: Check if prediction model flagged the transaction as fraud
    print(predict['predicted_class'])
    if predict['predicted_class']:
        recommendation = "deny"
    else:
        recommendation = "approve"

    save_transaction_to_db(tx_data, predict['fraud_probability'], predict['predicted_class'], recommendation, "model_based", db)
    return TransactionOut(transaction_id=payload.transaction_id, recommendation=recommendation)
import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from db import Transaction

def get_user_history(user_id, current_date, db: Session):
    """
    Retrieves the transaction history for a specific user up to (but not including) a given date.

    Args:
        user_id (int or str): The unique identifier of the user whose transaction history is to be fetched.
        current_date (datetime): The cutoff date; only transactions before this date are included.
        db (Session): SQLAlchemy database session used to query the transactions.

    Returns:
        pandas.DataFrame: A DataFrame containing the user's transaction history

        If no transactions are found, returns an empty DataFrame.
    """
    # Fetch user's previous transaction history
    history = db.query(Transaction).filter(
        Transaction.user_id == user_id,
        Transaction.transaction_date < current_date
    ).order_by(Transaction.transaction_date.asc()).all()
    # Convert to DataFrame
    if not history:
        return pd.DataFrame()
    data = []
    for t in history:
        data.append({
            "transaction_id": t.transaction_id,
            "merchant_id": t.merchant_id,
            "user_id": t.user_id,
            "device_id": t.device_id,
            "card_number": t.card_number,
            "transaction_date": t.transaction_date,
            "transaction_amount": t.transaction_amount,
            "fraud_probability": t.fraud_probability,
            "predicted_class": t.predicted_class
        })
    hist_df = pd.DataFrame(data)
    hist_df['transaction_date'] = pd.to_datetime(hist_df['transaction_date'])
    return hist_df


def predict_single(transaction_dict, db: Session):
    """
    Predicts the probability of fraud for a single transaction using a pre-trained model and user transaction history.
    Args:
        transaction_dict (dict): Dictionary containing transaction data. Expected keys include:
            - 'user_id': Identifier of the user.
            - 'transaction_date': Date and time of the transaction (string or datetime).
            - 'transaction_amount': Amount of the transaction.
            - 'card_number': Card number used in the transaction.
            - 'merchant_id': Identifier of the merchant.
            - 'device_id': Identifier of the device used.
        db (Session): SQLAlchemy database session used to retrieve user transaction history.
    Returns:
        dict: Dictionary with the following keys:
            - 'fraud_probability' (float): Probability that the transaction is fraudulent (between 0 and 1).
            - 'predicted_class' (int): Predicted class (1 if fraudulent, 0 otherwise), using a threshold of 0.65.
    Notes:
        - The function loads the model, scaler and label encoders from the './models/' directory.
        - It computes several historical and behavioral features based on the user's transaction history.
        - If the user has no transaction history, default values are used for historical features.
        - The function expects the transaction_dict to have all necessary fields required by the model.
        - All categorical fields are label-encoded using pre-fitted encoders.
        - The function is designed for single transaction prediction and is not optimized for batch processing.
    """

    # Load model, scaler and label_encoders
    model = joblib.load('./models/fraud_xgb_model.pkl')
    scaler = joblib.load('./models/scaler.pkl')
    label_encoders = joblib.load('./models/label_encoders.pkl')
    
    # Extract main info
    user_id = transaction_dict['user_id']
    transaction_date = pd.to_datetime(transaction_dict['transaction_date'])

    # Fetch user's transaction history
    hist_df = get_user_history(user_id, transaction_date, db)

    # Create DataFrame for the current transaction
    df = pd.DataFrame([transaction_dict])
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['hour'] = df['transaction_date'].dt.hour
    df['transaction_amount'] = np.log1p(df['transaction_amount'].fillna(0))

    # ==== Historical features ====
    if not hist_df.empty:

        user_cbk_rate = hist_df['predicted_class'].expanding().mean().iloc[-1]
        prev_date = hist_df['transaction_date'].iloc[-1]
        time_since_last_tx = (transaction_date - prev_date).total_seconds() / 60
        tx_24h = hist_df[hist_df['transaction_date'] > (transaction_date - pd.Timedelta(hours=24))].shape[0]
        amounts_last5 = hist_df['transaction_amount'].tail(5)
        mean_amount_last5 = amounts_last5.mean()
        std_amount_last5 = amounts_last5.std() if amounts_last5.shape[0] > 1 else 0

        user_mean_amount = hist_df['transaction_amount'].expanding().mean().iloc[-1]
        user_median_amount = hist_df['transaction_amount'].expanding().median().iloc[-1]
        user_std_amount = hist_df['transaction_amount'].expanding().std().iloc[-1] or 1

        amount_vs_user_mean = df['transaction_amount'].iloc[0] / user_mean_amount if user_mean_amount != 0 else 1
        amount_vs_user_median = df['transaction_amount'].iloc[0] / user_median_amount if user_median_amount != 0 else 1
        amount_zscore_user = (df['transaction_amount'].iloc[0] - user_mean_amount) / user_std_amount if user_std_amount != 0 else 0

        # Rolling percentile (relative to history)
        amount_percentile_user = (hist_df['transaction_amount'] < df['transaction_amount'].iloc[0]).mean()

        is_new_device = int(df['device_id'].iloc[0] not in hist_df['device_id'].values)
        is_new_merchant = int(df['merchant_id'].iloc[0] not in hist_df['merchant_id'].values)
        is_new_card = int(df['card_number'].iloc[0] not in hist_df['card_number'].values)

        tx_last_1h = hist_df[hist_df['transaction_date'] > (transaction_date - pd.Timedelta(hours=1))].shape[0]
        tx_last_6h = hist_df[hist_df['transaction_date'] > (transaction_date - pd.Timedelta(hours=6))].shape[0]
        tx_last_7d = hist_df[hist_df['transaction_date'] > (transaction_date - pd.Timedelta(days=7))].shape[0]
    else:
        # Defaults for users with no history
        user_cbk_rate = 0
        time_since_last_tx = 99999
        tx_24h = 0
        mean_amount_last5 = df['transaction_amount'].iloc[0]
        std_amount_last5 = 0
        user_mean_amount = df['transaction_amount'].iloc[0]
        user_median_amount = df['transaction_amount'].iloc[0]
        user_std_amount = 1
        amount_vs_user_mean = 1
        amount_vs_user_median = 1
        amount_zscore_user = 0
        amount_percentile_user = 0.5
        is_new_device = 1
        is_new_merchant = 1
        is_new_card = 1
        tx_last_1h = 0
        tx_last_6h = 0
        tx_last_7d = 0

    # Fill features in DataFrame
    df['user_cbk_rate'] = user_cbk_rate
    df['time_since_last_tx'] = time_since_last_tx
    df['tx_last_24h'] = tx_24h
    df['mean_amount_last5'] = mean_amount_last5
    df['std_amount_last5'] = std_amount_last5
    df['user_mean_amount'] = user_mean_amount
    df['user_median_amount'] = user_median_amount
    df['user_std_amount'] = user_std_amount
    df['amount_vs_user_mean'] = amount_vs_user_mean
    df['amount_vs_user_median'] = amount_vs_user_median
    df['amount_zscore_user'] = amount_zscore_user
    df['amount_percentile_user'] = amount_percentile_user
    df['is_new_device'] = is_new_device
    df['is_new_merchant'] = is_new_merchant
    df['is_new_card'] = is_new_card
    df['tx_last_1h'] = tx_last_1h
    df['tx_last_6h'] = tx_last_6h
    df['tx_last_7d'] = tx_last_7d

    df['weekday'] = df['transaction_date'].dt.weekday

    # Label encoding
    for col in ['card_number', 'user_id', 'merchant_id', 'device_id']:
        le = label_encoders.get(col)
        if le is not None:
            try:
                df[col] = le.transform(df[col].astype(str))
            except:
                df[col] = 0
        else:
            df[col] = 0

    # Feature selection and scaling
    features = [
        'transaction_amount', 'hour', 'card_number', 'user_id', 'merchant_id', 'device_id',
        'time_since_last_tx', 'tx_last_24h', 'mean_amount_last5', 'std_amount_last5',
        'user_mean_amount', 'user_median_amount', 'user_std_amount',
        'amount_vs_user_mean', 'amount_vs_user_median', 'amount_zscore_user',
        'amount_percentile_user',
        'is_new_device', 'is_new_merchant', 'is_new_card',
        'tx_last_1h', 'tx_last_6h', 'tx_last_7d',
        'user_cbk_rate'
    ]
    num_cols = [
        'transaction_amount', 'hour', 'time_since_last_tx', 'tx_last_24h', 'mean_amount_last5', 'std_amount_last5',
        'user_mean_amount', 'user_median_amount', 'user_std_amount',
        'amount_vs_user_mean', 'amount_vs_user_median', 'amount_zscore_user',
        'amount_percentile_user',
        'tx_last_1h', 'tx_last_6h', 'tx_last_7d',
        'user_cbk_rate'
    ]
    df[num_cols] = scaler.transform(df[num_cols])

    # Prediction
    proba = model.predict_proba(df[features])[:, 1][0]
    predicted_class = int(proba >= 0.65)

    return {"fraud_probability": round(proba, 4), "predicted_class": predicted_class}


def save_transaction_to_db(transaction_dict, fraud_probability, predicted_class, recommendation, recommendation_type, db: Session):
    """
    Saves a transaction to the database with additional fraud analysis results.
    Args:
        transaction_dict (dict): Dictionary containing transaction details. Must include 'transaction_id' and 'transaction_date'.
        fraud_probability (float): Probability that the transaction is fraudulent.
        predicted_class (str or int): Predicted class label for the transaction (e.g., 'fraud' or 'legit').
        recommendation (str): Recommendation based on the fraud analysis (e.g., 'approve', 'review', 'decline').
        recommendation_type (str): Type or reason for the recommendation.
        db (Session): SQLAlchemy database session object.
    Returns:
        dict: A dictionary with a message and the transaction ID. If the transaction already exists, returns a message indicating so.
    """

    # Save transaction to the database
    tx_data = transaction_dict.copy()
    tx_data['fraud_probability'] = float(fraud_probability)
    tx_data['predicted_class'] = predicted_class
    tx_data['recommendation'] = recommendation
    tx_data['recommendation_type'] = recommendation_type
    tx_data['transaction_date'] = pd.to_datetime(tx_data['transaction_date'])

    # Check if a transaction with the same ID already exists
    existing_tx = db.query(Transaction).filter(Transaction.transaction_id == tx_data["transaction_id"]).first()
    if existing_tx:
        return {"message": "Transaction already exists", "transaction_id": tx_data["transaction_id"]}

    # Create Transaction object and add to the database
    new_tx = Transaction(**tx_data)
    db.add(new_tx)
    db.commit()
    db.refresh(new_tx)
    return {"message": "Transaction saved successfully", "transaction_id": new_tx.transaction_id}
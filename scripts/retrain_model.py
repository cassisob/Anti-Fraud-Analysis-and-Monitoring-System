"""
    Main function to retrain the XGBoost model pipeline.
    1. Loads the dataset.
    2. Preprocesses the data, including feature scaling and label encoding.
    3. Trains and evaluates an XGBoost model using a specified threshold.
    4. Prints evaluation metrics: confusion matrix, classification report, ROC AUC and Precision-Recall AUC.
    5. Saves the trained model, scaler and label encoders to disk.
    All paths and model parameters are assumed to be defined elsewhere in the script.
"""

import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import os

# =============================
# 0. LOAD DATA FROM SQLITE
# =============================

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'antifraud.db')) # Path to SQLite database with transaction data
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'fraud_xgb_model.pkl')) # Path to save the trained XGBoost model
SCALER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')) # Path to save the fitted scaler object
ENCODERS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'label_encoders.pkl')) # Path to save the fitted label encoders

def load_data():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df = pd.read_sql("SELECT * FROM transactions", engine)
    return df

# =============================
# 1. ADVANCED PREPROCESSING
# =============================

def preprocess_data(data):
    """
    Preprocesses transaction data for model training by generating time-based, user-based and statistical features.
    Steps performed:
    - Converts transaction dates to datetime and extracts hour of transaction.
    - Applies log transformation to transaction amounts and handles missing values.
    - Sorts data by user and transaction date.
    - Computes user-specific historical chargeback rate (user_cbk_rate).
    - Calculates time since last transaction for each user.
    - Counts transactions in the last 24 hours for each user.
    - Computes rolling mean and standard deviation of transaction amounts for the last 5 transactions per user.
    - Calculates expanding mean, median and standard deviation of transaction amounts per user.
    - Derives features comparing current transaction amount to user's historical statistics (mean, median, z-score, percentile).
    - Flags if the device, merchant, or card is new for the user.
    - Counts transactions in the last 1 hour, 6 hours and 7 days for each user.
    - Handles missing and infinite values.
    - Encodes categorical features (card_number, user_id, merchant_id, device_id) using LabelEncoder.
    - Scales numerical features using StandardScaler.
    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame containing transaction data. Must include columns:
        ['transaction_date', 'transaction_amount', 'user_id', 'has_cbk', 'device_id', 'merchant_id', 'card_number']
    Returns
    -------
    X : pandas.DataFrame
        Preprocessed feature matrix ready for model input.
    y : numpy.ndarray
        Target variable array (chargeback labels).
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler for numerical features.
    label_encoders : dict
        Dictionary of fitted LabelEncoders for categorical features.
    """
    data = data.copy()
    data['transaction_date'] = pd.to_datetime(data['transaction_date'])
    data['hour'] = data['transaction_date'].dt.hour
    data['transaction_amount'] = np.log1p(data['transaction_amount'].fillna(0))
    data = data.sort_values(['user_id', 'transaction_date'])

    data['user_cbk_rate'] = (
        data.groupby('user_id')['has_cbk']
        .transform(lambda x: x.shift().expanding().mean())
        .fillna(0)
    )

    data['prev_date'] = data.groupby('user_id')['transaction_date'].shift(1)
    data['time_since_last_tx'] = (data['transaction_date'] - data['prev_date']).dt.total_seconds() / 60
    data['time_since_last_tx'] = data['time_since_last_tx'].fillna(99999)

    data['tx_last_24h'] = 0
    for user_id, user_df in data.groupby('user_id'):
        idx = user_df.index
        user_df = user_df.set_index('transaction_date')
        tx_last_24h = user_df['has_cbk'].rolling('24h', closed='left').count()
        data.loc[idx, 'tx_last_24h'] = tx_last_24h.values

    data['mean_amount_last5'] = (
        data.groupby('user_id')['transaction_amount']
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    )
    data['std_amount_last5'] = (
        data.groupby('user_id')['transaction_amount']
        .transform(lambda x: x.rolling(window=5, min_periods=1).std().fillna(0))
    )

    data['user_mean_amount'] = data.groupby('user_id')['transaction_amount'].transform(lambda x: x.expanding().mean())
    data['user_median_amount'] = data.groupby('user_id')['transaction_amount'].transform(lambda x: x.expanding().median())
    data['user_std_amount'] = data.groupby('user_id')['transaction_amount'].transform(lambda x: x.expanding().std().replace(0, 1).fillna(1))
    data['amount_vs_user_mean'] = data['transaction_amount'] / data['user_mean_amount']
    data['amount_vs_user_median'] = data['transaction_amount'] / data['user_median_amount']
    data['amount_zscore_user'] = (data['transaction_amount'] - data['user_mean_amount']) / data['user_std_amount']

    def rolling_percentile(x):
        pctls = []
        for i in range(len(x)):
            if i == 0:
                pctls.append(0.5)
            else:
                pctls.append((x.iloc[:i] < x.iloc[i]).mean())
        return pd.Series(pctls, index=x.index)
    data['amount_percentile_user'] = data.groupby('user_id')['transaction_amount'].transform(rolling_percentile)

    data['is_new_device'] = (data.groupby('user_id')['device_id']
                                .transform(lambda x: ~x.duplicated()).astype(int))
    data['is_new_merchant'] = (data.groupby('user_id')['merchant_id']
                                .transform(lambda x: ~x.duplicated()).astype(int))
    data['is_new_card'] = (data.groupby('user_id')['card_number']
                                .transform(lambda x: ~x.duplicated()).astype(int))

    data['tx_last_1h'] = (data.groupby('user_id')
        .apply(lambda x: x.set_index('transaction_date')['transaction_amount']
            .rolling('1h', closed='left').count())
        .reset_index(level=0, drop=True))
    data['tx_last_6h'] = (data.groupby('user_id')
        .apply(lambda x: x.set_index('transaction_date')['transaction_amount']
            .rolling('6h', closed='left').count())
        .reset_index(level=0, drop=True))
    data['tx_last_7d'] = (data.groupby('user_id')
        .apply(lambda x: x.set_index('transaction_date')['transaction_amount']
            .rolling('7d', closed='left').count())
        .reset_index(level=0, drop=True))

    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)

    label_encoders = {}
    for col in ['card_number', 'user_id', 'merchant_id', 'device_id']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

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
    X = data[features].copy()

    scaler = StandardScaler()
    num_cols = [
        'transaction_amount', 'hour', 'time_since_last_tx', 'tx_last_24h', 'mean_amount_last5', 'std_amount_last5',
        'user_mean_amount', 'user_median_amount', 'user_std_amount',
        'amount_vs_user_mean', 'amount_vs_user_median', 'amount_zscore_user',
        'amount_percentile_user',
        'tx_last_1h', 'tx_last_6h', 'tx_last_7d',
        'user_cbk_rate'
    ]
    X[num_cols] = scaler.fit_transform(X[num_cols])

    y = data['has_cbk'].values

    return X, y, scaler, label_encoders

# =============================
# 2. XGBOOST TRAINING AND EVALUATION
# =============================

def train_xgb(X, y, threshold=0.65):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    xgb_model = xgb.XGBClassifier(
        n_estimators=700,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=1.0,
        scale_pos_weight=(sum(y_train == 0) / sum(y_train == 1)),
        gamma=1,
        min_child_weight=1,
        eval_metric='logloss',
        random_state=42,
        tree_method="hist"
    )

    xgb_model.fit(X_train, y_train)

    y_proba = xgb_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return xgb_model, y_test, y_pred, y_proba

# =============================
# 3. MAIN EXECUTION
# =============================

def main():

    print("Loading data...")
    df = load_data()
    print(f"Total records: {df.shape[0]}")

    print("Preprocessing...")
    X, y, scaler, label_encoders = preprocess_data(df)

    print("Training and evaluating XGBoost model...")
    xgb_model, y_test, y_pred, y_proba = train_xgb(
        X, y, threshold=0.65
    )
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(xgb_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoders, ENCODERS_PATH)

    print("Retraining completed and models saved!")

if __name__ == "__main__":
    main()
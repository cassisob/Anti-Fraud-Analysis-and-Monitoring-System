# Anti-Fraud Transaction Monitoring System

## 1. Introduction

This project is a technical assessment for a Data Analyst position at a private company.  
It implements a simple anti-fraud system for real-time transaction monitoring, combining rule-based and model-based approaches.  
The system provides an API for transaction evaluation and a dashboard for monitoring transaction trends and anomalies.

In addition to the application, the project includes a detailed data analysis documented in the `analysis/data_analyst_task.ipynb` notebook and a summary presentation in `analysis/presentation.pdf`.  
**Objectives of the analysis:**  
- Explore and understand the transaction dataset, identifying key patterns and risk factors.
- Engineer relevant features for fraud detection.
- Develop and evaluate a machine learning model to support the anti-fraud logic.
- Provide actionable insights and recommendations for improving fraud detection strategies.

## 2. Project Structure

```
analysis/
    data_analyst_task.ipynb  # Data analysis and model development
    presentation.pdf         # Summary presentation of analysis and results
app/
    api/
        predict.py           # API endpoint for transaction prediction
        transactions.py      # API endpoint for dashboard data and HTML
    core/
        config.py            # Configuration (paths, DB)
        model_utils.py       # Model loading and prediction utilities
    schemas/
        transaction.py       # Pydantic schemas for API
    static/
        css/
            transactions.css # Dashboard styles
        js/
            transactions.js  # Dashboard logic
    templates/
        transactions.html    # Dashboard HTML
    db.py                    # Database models and session
    main.py                  # FastAPI app entrypoint
    run.py                   # Script to run the server
data/                        # SQLite database (created at runtime) and CSV (simulated data)     
models/                      # Saved machine learning models (.pkl, etc.)
scripts/                     # Utility scripts (retrain_model.py, simulate.py, etc.)
```

## 3. Features

### 3.1 API Endpoints

**POST `/predict`**  
Receives transaction data and returns a recommendation to approve or deny the transaction.  
The decision is based on:<br>
    - Rule 1: Deny if the user had a chargeback in the last week.<br>
    - Rule 2: Deny if the user made 3 or more transactions in the last 5 minutes.<br>
    - Rule 3: Deny if the sum of transaction amounts in the last 3 hours plus the current transaction exceeds 2 standard deviations above the user's lastly mean.<br>
    - Rule 4: If none of the above, use the model's prediction.<br>

**GET `/transactions`**  
Returns the dashboard HTML for real-time monitoring.

**GET `/transactions/data`**  
Returns aggregated transaction data for the dashboard, including:<br>
    - Time series of approved and denied transactions (model/rule)<br>
    - Trend indicators for each status

### 3.2 Dashboard

- Real-time charts for transaction status and trends
- Responsive and modern UI

### 3.3 Data Analysis

- All data analysis, feature engineering and model training are documented in `analysis/data_analyst_task.ipynb`.
- A summary of the analysis and key findings is available in `analysis/presentation.pdf`.

## 4. How to Run

### 4.1 Requirements

- Python
- pip

### 4.2 Installation

1. Clone the repository:
     ```
     git clone <repo_url>
     cd <repo_folder>
     ```

2. Install dependencies:
     ```
     pip install -r requirements.txt
     ```

3. (Optional) Run the data analysis notebook for model retraining:
     ```
     cd analysis
     jupyter notebook data_analyst_task.ipynb
     ```

### 4.3 Running the Application

From the root folder, run:

```
uvicorn app.main:app --reload
```
or
```
python app/run.py
```

The API and dashboard will be available at `http://127.0.0.1:8000/`.

### 4.4 API Usage Example

**Request:**
```json
POST /predict
{
    "transaction_id": 2342357,
    "merchant_id": 29744,
    "user_id": 97051,
    "card_number": "434505******9116",
    "transaction_date": "2019-11-31T23:16:32.812632",
    "transaction_amount": 373,
    "device_id": 285475
}
```

**Response:**
```json
{
    "transaction_id": 2342357,
    "recommendation": "approve"
}
```

### 4.5 Model Retraining and Data Simulation

**`retrain_model.py`**  
This script is designed to work as a service that periodically retrains the machine learning model using the latest transaction data stored in the database.
- It can be scheduled (for example, via cron or a background job) to ensure the anti-fraud model stays up-to-date with new fraud patterns.
- The retrained model is saved and automatically used by the API for future predictions.

**How to run retraining manually:**
```
python app/retrain_model.py
```
Or configure a scheduler to run it periodically.

---

**`simulate.py`**  
This script simulates the real-time ingestion of transaction data into the system.
- It generates and sends synthetic transactions to the API/database, mimicking real-world transaction flows.
- Useful for testing, stress-testing and demonstrating the dashboard and anti-fraud logic in action.

**How to run simulation:**
```
python app/simulate.py
```

## 5. Anti-Fraud Logic

**Rule-based rejection:**  
- Too many transactions in a short period
- Transactions above a threshold for the user
- Previous chargebacks

**Model-based rejection:**  
- Machine learning model trained on historical data (see analysis notebook)

**All transactions and recommendations are stored in the database for audit and retraining.**

## 6. Notes

- The dashboard is accessible at `/transactions`.
- All static files are served with cache disabled for development convenience.
- The database is created automatically on first run.

### 6.1. Why SQLite?

SQLite was chosen for this project due to its simplicity, zero-configuration setup and suitability for small-scale applications and technical assessments.  

**Important:**  
For a real-world, production-grade anti-fraud system, a more robust and scalable database solution (such as PostgreSQL, MySQL, or a cloud-managed database) should be used to ensure data integrity, performance and security.

## 7. Further Improvements

- Add filter improvements in the dashboard route
- Improve model retraining pipeline and monitoring.

## 8. License

This project is for technical assessment and demonstration purposes
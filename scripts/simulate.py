import pandas as pd
import requests
import time

def simulate_post_requests(csv_path, api_url, num_requests=10, delay_seconds=0.2):
    """
    Simulates sending POST requests to a specified API endpoint using transaction data from a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing transaction data.
        api_url (str): The API endpoint URL to which POST requests will be sent.
        num_requests (int, optional): Number of requests to simulate. Defaults to 10.
        delay_seconds (float, optional): Delay in seconds between each request. Defaults to 0.2.

    The function reads transaction records from the CSV file, constructs a JSON payload for each transaction,
    and sends it as a POST request to the given API URL. It prints the transaction ID, response status code,
    the 'has_cbk' flag, and the result of each request. Handles both JSON and non-JSON responses gracefully.
    """
    
    df = pd.read_csv(csv_path)
    for _, row in df.head(num_requests).iterrows():
        payload = {
            "transaction_id": str(row["transaction_id"]),
            "transaction_date": row["transaction_date"],
            "transaction_amount": float(row["transaction_amount"]),
            "device_id": int(row["device_id"]) if not pd.isna(row["device_id"]) else 0,
            "card_number": str(row["card_number"]),
            "user_id": str(row["user_id"]),
            "merchant_id": str(row["merchant_id"]),
            "has_cbk": bool(row["has_cbk"])
        }
        response = requests.post(api_url, json=payload)
        try:
            result = response.json()
        except ValueError:
            result = response.text
        print(
            f"Transaction {row['transaction_id']} | Status: {response.status_code} | has_cbk = {row['has_cbk']} | Result: {result}"
        )
        time.sleep(delay_seconds)

# Example usage:
if __name__ == "__main__":
    simulate_post_requests(
        "./data/transactional-sample.csv",
        "http://127.0.0.1:8000/predict",
        num_requests=3200
    )
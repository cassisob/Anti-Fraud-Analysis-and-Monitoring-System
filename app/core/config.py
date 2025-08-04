import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # app

DB_PATH = os.path.join(BASE_DIR, "../data/antifraud.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"
STATIC_DIR = os.path.join(os.path.dirname(BASE_DIR), "app", "static")
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from core.config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Transaction(Base):
    __tablename__ = "transactions"
    transaction_id = Column(Integer, primary_key=True, index=True)
    merchant_id = Column(Integer)
    user_id = Column(Integer)
    device_id = Column(Integer)
    card_number = Column(String)
    transaction_date = Column(DateTime)
    transaction_amount = Column(Float)
    has_cbk = Column(Boolean)
    fraud_probability = Column(Float)
    predicted_class = Column(Boolean)
    recommendation = Column(String)
    recommendation_type = Column(String)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create the database if not
Base.metadata.create_all(bind=engine)
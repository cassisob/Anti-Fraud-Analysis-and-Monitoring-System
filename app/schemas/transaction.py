from pydantic import BaseModel

class TransactionIn(BaseModel):
    transaction_id: int
    merchant_id: int
    user_id: int
    card_number: str
    transaction_date: str
    transaction_amount: float
    device_id: int
    has_cbk: bool

class TransactionOut(BaseModel):
    transaction_id: int
    recommendation: str
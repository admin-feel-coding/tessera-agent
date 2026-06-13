from pydantic import BaseModel


class Transaction(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    currency: str = "USD"
    merchant_category: str = ""
    ip_address: str = ""
    device_id: str = ""
    card_bin: str = ""
    email: str = ""

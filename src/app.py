from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from datetime import datetime, timedelta
from src.rule_engine import compute_risk


app = FastAPI(
    title="Churn Risk Prediction Service",
    description="Rule-based churn risk prediction API using customer and ticket data",
    version="1.0"
)


# Load datasets once when the service starts
customers = pd.read_csv("data/processed/customers.csv")
tickets = pd.read_csv("data/processed/tickets.csv")

tickets["created_at"] = pd.to_datetime(tickets["created_at"])


class CustomerRequest(BaseModel):
    customer_id: str


@app.get("/")
def health_check():
    return {"status": "service running"}


def compute_features(customer_id):

    # Fetch customer row
    customer_row = customers[customers["customer_id"] == customer_id]

    if customer_row.empty:
        raise HTTPException(status_code=404, detail="Customer not found")

    contract_type = customer_row.iloc[0]["contract_type"]

    # Fetch ticket history
    customer_tickets = tickets[tickets["customer_id"] == customer_id]

    now = datetime.now()
    window_30 = now - timedelta(days=30)

    # tickets_last_30_days
    tickets_30 = customer_tickets[customer_tickets["created_at"] > window_30]
    tickets_last_30_days = len(tickets_30)

    # complaint_ticket
    complaint_ticket = int((customer_tickets["ticket_type"] == "complaint").any())

    # negative_ratio
    if len(customer_tickets) > 0:
        negative_ratio = (
            (customer_tickets["sentiment"] == "negative").sum()
            / len(customer_tickets)
        )
    else:
        negative_ratio = 0

    return {
        "contract_type": contract_type,
        "tickets_last_30_days": tickets_last_30_days,
        "complaint_ticket": complaint_ticket,
        "negative_ratio": negative_ratio
    }


@app.post("/predict-risk")
def predict_risk(request: CustomerRequest):

    features = compute_features(request.customer_id)

    risk = compute_risk(features)

    return {
        "customer_id": request.customer_id,
        "risk_category": risk
    }
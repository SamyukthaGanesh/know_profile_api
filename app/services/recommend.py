import pandas as pd
from typing import List

def tips_for_user(row: pd.Series) -> list[str]:
    tips = []
    total = row.get("total_spend_inr", 0.0)
    dti = row.get("dti_ratio", 0.0)
    credit = row.get("credit_score", 0)
    int_income = row.get("annual_interest_inr", 0)
    savings = row.get("savings_balance_inr", 0)
    mf = row.get("mf_value_inr", 0)
    demat = row.get("demat_value_inr", 0)

    # Shopping insight
    if row.get("Shopping", 0) > 0.25 * total:
        perc = round(100 * row.get("Shopping", 0) / total, 1)
        tips.append(f"🛍️ Your shopping spend is {perc}% of total — above the ideal 20%. "
                    "Reducing it can improve creditworthiness and savings rate.")

    # Debt-to-income reasoning
    if dti > 0.45:
        tips.append(f"💸 Your DTI is {dti:.2f}, which is high. "
                    "Try to keep total EMIs under 35% of income for better loan eligibility.")

    # Credit score reasoning
    if credit < 680:
        tips.append(f"📉 Credit score {credit} is below ideal (700+). "
                    "Pay bills on time and reduce credit utilization below 30%.")
    elif credit > 780:
        tips.append(f"🌟 Excellent credit score ({credit})! Maintain low utilization and timely payments.")

    # Interest optimization reasoning
    if int_income < 0.04 * (savings + mf + demat):
        tips.append("💰 You could earn higher returns by diversifying into mutual funds or fixed deposits.")

    # Asset diversification tip
    if mf + demat < 0.3 * (savings + mf + demat):
        tips.append("📊 Consider allocating at least 30% of long-term savings into MF or DEMAT for better yield.")

    if not tips:
        tips.append("✅ You’re on track — balanced spending, strong savings, and low debt!")
    return tips


def rewards_for_user(row: pd.Series) -> dict:
    # synthetic rewards/benefits demo
    points = int(1000 + (row.get("total_spend_inr",0)/1000)%5000)
    cashback = int(min(2000, row.get("Shopping",0)*0.02))
    unused = ["Fuel surcharge waiver", "Movie ticket BOGO", "Dining 10% off"]
    return {"points": points, "cashback_eligible_inr": cashback, "unused_benefits": unused}


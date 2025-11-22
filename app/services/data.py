import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict
import os
from pathlib import Path

class DataStore:
    def __init__(self):
        self.users: pd.DataFrame | None = None
        self.tx: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None

DATA = DataStore()

# Path to CSV files
DATA_DIR = Path(__file__).parent.parent.parent / "data"
USERS_CSV = DATA_DIR / "users.csv"
TX_CSV = DATA_DIR / "transactions.csv"

def load_from_csv() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data from CSV files if they exist."""
    if USERS_CSV.exists() and TX_CSV.exists():
        users = pd.read_csv(USERS_CSV)
        tx = pd.read_csv(TX_CSV)
        tx["date"] = pd.to_datetime(tx["date"])
        tx["month"] = tx["date"].dt.to_period("M").astype(str)
        DATA.users, DATA.tx = users, tx
        print(f"✅ Loaded {len(users)} users and {len(tx)} transactions from CSV")
        return users, tx
    return None, None

def save_to_csv(users: pd.DataFrame, tx: pd.DataFrame):
    """Save data to CSV files."""
    DATA_DIR.mkdir(exist_ok=True)
    users.to_csv(USERS_CSV, index=False)
    tx_save = tx.copy()
    tx_save["date"] = tx_save["date"].dt.strftime("%Y-%m-%d")
    tx_save.to_csv(TX_CSV, index=False)
    print(f"✅ Saved {len(users)} users and {len(tx)} transactions to CSV")

def generate_synthetic(seed: int = 42, n_users: int = 200) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Try loading from CSV first
    users, tx = load_from_csv()
    if users is not None and tx is not None:
        return users, tx
    
    # Generate new data if CSV doesn't exist
    print("📊 Generating synthetic data...")
    np.random.seed(seed)
    user_ids = [f"U{1000+i}" for i in range(n_users)]
    incomes = np.random.normal(900000, 250000, n_users).clip(250000, 2500000)
    credit_scores = np.random.normal(720, 70, n_users).clip(300, 900)
    savings_bal = np.random.normal(250000, 120000, n_users).clip(0, 1500000)
    cc_limits = np.random.normal(200000, 80000, n_users).clip(50000, 1000000)
    # --- Add after cc_limits ---
    fd_balance = np.random.normal(200000, 100000, n_users).clip(0, 800000)    # Fixed Deposits
    rd_balance = np.random.normal(100000, 40000, n_users).clip(0, 300000)     # Recurring Deposits
    mf_value   = np.random.normal(250000, 100000, n_users).clip(0, 1000000)   # Mutual Funds
    demat_value = np.random.normal(150000, 80000, n_users).clip(0, 700000)    # Stocks/DEMAT

    regions = np.random.choice(["North","South","East","West"], n_users, p=[0.25,0.35,0.2,0.2])
    users = pd.DataFrame({
        "user_id": user_ids,
        "annual_income_inr": incomes.round(0).astype(int),
        "credit_score": credit_scores.round(0).astype(int),
        "savings_balance_inr": savings_bal.round(0).astype(int),
        "cc_limit_inr": cc_limits.round(0).astype(int),
        "region": regions,
        "fd_balance_inr": fd_balance.round(0).astype(int),
        "rd_balance_inr": rd_balance.round(0).astype(int),
        "mf_value_inr":   mf_value.round(0).astype(int),
        "demat_value_inr": demat_value.round(0).astype(int),
    })

    categories = {
        "Groceries": ["BigBazaar","DMart","Nature's Basket","JioMart"],
        "Dining": ["Zomato","Swiggy","Cafe Coffee Day","Barista"],
        "Transport": ["Uber","Ola","Metro","Fuel Station"],
        "Shopping": ["Amazon","Flipkart","Myntra","Ajio"],
        "Utilities": ["Electricity Board","Water Dept","Gas Company","Mobile Prepaid"],
        "Health": ["Pharmacy","Clinic","Diagnostics","Insurance"],
        "Entertainment": ["BookMyShow","OTT","Music","Gaming"],
        "Rent": ["House Rent"],
        "Savings": ["Auto-Sweep"],
        "Investments": ["Mutual Fund","Fixed Deposit","Stocks"]
    }
    start_date, end_date = datetime(2025,5,1), datetime(2025,10,31)
    days = (end_date - start_date).days + 1

    rows = []
    for uid in user_ids:
        n_tx = np.random.randint(120, 260)
        for _ in range(n_tx):
            d = start_date + timedelta(days=int(np.random.randint(0, days)))
            cat = np.random.choice(list(categories.keys()), p=[0.12,0.11,0.08,0.14,0.12,0.09,0.08,0.06,0.05,0.15])
            merch = np.random.choice(categories[cat])
            base_map = {
                "Groceries": np.random.normal(2500, 1200),
                "Dining": np.random.normal(900, 500),
                "Transport": np.random.normal(400, 250),
                "Shopping": np.random.normal(3000, 2000),
                "Utilities": np.random.normal(1800, 1200),
                "Health": np.random.normal(2200, 1800),
                "Entertainment": np.random.normal(600, 400),
                "Rent": np.random.normal(20000, 2000),
                "Savings": np.random.normal(5000, 2500),
                "Investments": np.random.normal(8000, 6000),
            }
            base = base_map[cat]
            noise_scale = max(1.0, abs(base)*0.15)
            amt = max(50, abs(base + np.random.normal(0, noise_scale)))
            method = np.random.choice(["UPI","Card","NetBanking"], p=[0.6,0.3,0.1])
            rows.append([uid, d.date().isoformat(), cat, merch, round(float(amt),2), method])

    tx = pd.DataFrame(rows, columns=["user_id","date","category","merchant","amount_inr","method"])
    tx["date"] = pd.to_datetime(tx["date"])
    tx["month"] = tx["date"].dt.to_period("M").astype(str)

    DATA.users, DATA.tx = users, tx
    
    # Save to CSV
    save_to_csv(users, tx)
    
    return users, tx

def build_features() -> pd.DataFrame:
    users, tx = DATA.users, DATA.tx
    assert users is not None and tx is not None, "Call generate_synthetic() first."

    spend_by_cat = tx.pivot_table(index="user_id", columns="category",
                                  values="amount_inr", aggfunc="sum").fillna(0)
    spend_total = tx.groupby("user_id")["amount_inr"].sum().rename("total_spend_inr")
    recurring = (tx[tx["category"].isin(["Rent","Utilities"])]
                 .groupby(["user_id","category"])["month"]
                 .nunique().unstack().fillna(0).add_prefix("recurring_"))
    users = DATA.users  # already defined above in the function

    users["total_assets_inr"] = (
    users["savings_balance_inr"]
    + users["fd_balance_inr"]
    + users["rd_balance_inr"]
    + users["mf_value_inr"]
    + users["demat_value_inr"]
    )

    # Set user_id as index FIRST before calculating interest and DTI
    users_indexed = users.set_index("user_id")
    
    # Calculate annual interest with proper indexing
    interest = (
    users_indexed["savings_balance_inr"] * 0.035
    + users_indexed["fd_balance_inr"]     * 0.065
    + users_indexed["rd_balance_inr"]     * 0.055
    + users_indexed["mf_value_inr"]       * 0.10
    + users_indexed["demat_value_inr"]    * 0.12
    ).rename("annual_interest_inr").round(0).astype(int)

    cat_sums = tx.groupby(["user_id","category"])["amount_inr"].sum().unstack().fillna(0)
    monthly_income = users_indexed["annual_income_inr"] / 12.0
    
    # Calculate monthly debt more realistically
    # Rent is recurring monthly, Utilities are monthly, add credit card and other recurring payments
    # Transactions are over 6 months (May-Oct 2025), so divide by 6 to get monthly average
    months_data = 6.0
    monthly_rent = cat_sums.get("Rent", 0) / months_data
    monthly_utilities = cat_sums.get("Utilities", 0) / months_data
    monthly_health = cat_sums.get("Health", 0) / months_data  # Insurance etc
    
    # Assume credit card debt as a percentage of shopping/dining
    estimated_cc_debt = (cat_sums.get("Shopping", 0) + cat_sums.get("Dining", 0)) * 0.15 / months_data
    
    monthly_debt = monthly_rent + monthly_utilities + monthly_health + estimated_cc_debt
    
    # DTI = Monthly debt obligations / Monthly gross income
    dti = (monthly_debt / monthly_income.replace(0, 1)).clip(0, 1).rename("dti_ratio")
    
    # Join all the computed features
    features = users_indexed.join([spend_by_cat, spend_total, recurring, interest, dti]).fillna(0)
    
    # Ensure asset columns are present (they should be from users_indexed)
    # This is a safety check
    required_cols = ["annual_income_inr", "credit_score", "savings_balance_inr", "cc_limit_inr", 
                     "fd_balance_inr", "rd_balance_inr", "mf_value_inr", "demat_value_inr", 
                     "total_assets_inr", "region"]
    missing_cols = [col for col in required_cols if col not in features.columns]
    if missing_cols:
        for col in missing_cols:
            if col in users_indexed.columns:
                features[col] = users_indexed[col]
    
    DATA.features = features
    print(f"✅ Built features with {len(features)} users and {len(features.columns)} columns")
    return features

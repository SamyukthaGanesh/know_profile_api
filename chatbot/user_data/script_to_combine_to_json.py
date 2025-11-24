import pandas as pd
import json

# Load CSVs
users = pd.read_csv('ai_governance_chatbot/user_data/users.csv')
transactions = pd.read_csv('ai_governance_chatbot/user_data/transactions.csv')

# Prepare user_details as a list of dicts (each with user_id and details)
user_details = users.to_dict(orient='records')

# Prepare user_transactions as a list of dicts (each with user_id and transaction details)
user_transactions = transactions.to_dict(orient='records')

# Combine into the required schema
combined_data = {
    "user_details": user_details,
    "user_transactions": user_transactions
}

# Save to JSON
with open('ai_governance_chatbot/user_data/users_combined_data.json', 'w') as f:
    json.dump(combined_data, f, indent=2)

print("Combined JSON saved as users_with_transactions.json")
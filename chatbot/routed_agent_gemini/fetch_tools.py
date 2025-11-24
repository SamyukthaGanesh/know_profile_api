import os, sqlite3
from typing import Any, Dict

# Database path - can be overridden with environment variable
DB_PATH = os.getenv("USER_DATA_DB", "/Users/ssunehra/ghci/ai_governance_chatbot/user_data.db")

def get_db_connection() -> sqlite3.Connection:
    """Create a connection to the SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    # Enable dictionary row factory for easier JSON conversion
    conn.row_factory = sqlite3.Row
    return conn

def dict_factory(cursor, row):
    """Convert SQLite row objects to dictionaries"""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

def tool_get_customer(customer_id: str) -> Dict[str, Any]:
    """Fetch customer details by user ID from the SQLite database"""
    conn = get_db_connection()
    try:
        conn.row_factory = dict_factory
        cursor = conn.cursor()

        # Query user details from SQLite
        cursor.execute("""
        SELECT
            user_id as id,
            annual_income_inr as annual_income,
            credit_score,
            savings_balance_inr as savings_balance,
            cc_limit_inr as credit_card_limit,
            region,
            fd_balance_inr as fixed_deposit_balance,
            rd_balance_inr as recurring_deposit_balance,
            mf_value_inr as mutual_fund_value,
            demat_value_inr as demat_account_value
        FROM
            user_details
        WHERE
            user_id = ?
        """, (customer_id,))

        customer = cursor.fetchone()
        return customer or {}
    finally:
        conn.close()

def tool_list_transactions(customer_id: str):
    """Fetch transactions and user details for a specific user from SQLite"""
    conn = get_db_connection()
    try:
        conn.row_factory = dict_factory
        cursor = conn.cursor()

        # Query transactions from SQLite
        cursor.execute("""
        SELECT
            transaction_id as id,
            user_id as customer_id,
            date,
            category,
            merchant,
            amount_inr as amount,
            method as payment_method,
            month
        FROM
            transactions
        WHERE
            user_id = ?
        ORDER BY
            date DESC
        """, (customer_id,))

        transactions = cursor.fetchall()

        # Query user details
        cursor.execute("""
        SELECT
            user_id as id,
            annual_income_inr as annual_income,
            credit_score,
            savings_balance_inr as savings_balance,
            cc_limit_inr as credit_card_limit,
            region,
            fd_balance_inr as fixed_deposit_balance,
            rd_balance_inr as recurring_deposit_balance,
            mf_value_inr as mutual_fund_value,
            demat_value_inr as demat_account_value
        FROM
            user_details
        WHERE
            user_id = ?
        """, (customer_id,))

        details = cursor.fetchone() or {}

        return transactions, details
    finally:
        conn.close()



def tool_get_decision_log(customer_id: str) -> list[Dict[str, Any]]:
    """Fetch all decision logs for a user, including explanations, fairness, and compliance data."""
    conn = get_db_connection()
    try:
        conn.row_factory = dict_factory
        cursor = conn.cursor()

        # Get all decision logs for the user
        cursor.execute("""
        SELECT
            decision_id,
            user_id,
            outcome,
            confidence,
            timestamp,
            model_name,
            model_version,
            user_explanation
        FROM
            decision_logs
        WHERE
            user_id = ?
        ORDER BY
            timestamp DESC
        """, (customer_id,))

        logs = cursor.fetchall()
        all_logs = []

        for log in logs:
            # Format model information
            log["model"] = {
                "name": log.pop("model_name"),
                "version": log.pop("model_version")
            }

            decision_id = log["decision_id"]

            # Get explanation data
            cursor.execute("""
            SELECT
                explanation_id,
                method
            FROM
                explanations
            WHERE
                decision_id = ?
            """, (decision_id,))
            explanation_data = cursor.fetchone()

            if explanation_data:
                explanation_id = explanation_data["explanation_id"]
                cursor.execute("""
                SELECT
                    name,
                    value,
                    importance,
                    impact,
                    description
                FROM
                    explanation_factors
                WHERE
                    explanation_id = ?
                ORDER BY
                    ABS(importance) DESC
                """, (explanation_id,))
                factors = cursor.fetchall()
                for factor in factors:
                    value = factor["value"]
                    try:
                        if value.lower() in ("true", "false"):
                            factor["value"] = value.lower() == "true"
                        elif "." in value and value.replace(".", "").isdigit():
                            factor["value"] = float(value)
                        elif value.isdigit():
                            factor["value"] = int(value)
                    except (ValueError, AttributeError):
                        pass
                log["explanation"] = {
                    "method": explanation_data["method"],
                    "top_factors": factors
                }

            # Get fairness data
            cursor.execute("""
            SELECT
                fairness_id,
                bias_detected,
                fairness_score,
                message
            FROM
                fairness
            WHERE
                decision_id = ?
            """, (decision_id,))
            fairness_data = cursor.fetchone()
            if fairness_data:
                fairness_id = fairness_data["fairness_id"]
                fairness_data["bias_detected"] = bool(fairness_data["bias_detected"])
                cursor.execute("""
                SELECT
                    attribute_name
                FROM
                    fairness_protected_attributes
                WHERE
                    fairness_id = ?
                """, (fairness_id,))
                attributes = cursor.fetchall()
                protected_attributes = [attr["attribute_name"] for attr in attributes]
                log["fairness"] = {
                    "bias_detected": fairness_data["bias_detected"],
                    "fairness_score": fairness_data["fairness_score"],
                    "protected_attributes_used": protected_attributes,
                    "message": fairness_data["message"]
                }

            # Get compliance data
            cursor.execute("""
            SELECT
                compliance_id,
                is_compliant,
                policies_checked,
                audit_receipt_id
            FROM
                compliance
            WHERE
                decision_id = ?
            """, (decision_id,))
            compliance_data = cursor.fetchone()
            if compliance_data:
                compliance_id = compliance_data["compliance_id"]
                compliance_data["is_compliant"] = bool(compliance_data["is_compliant"])
                cursor.execute("""
                SELECT
                    violation_description
                FROM
                    compliance_violations
                WHERE
                    compliance_id = ?
                """, (compliance_id,))
                violations = cursor.fetchall()
                violation_list = [v["violation_description"] for v in violations]
                log["compliance"] = {
                    "is_compliant": compliance_data["is_compliant"],
                    "policies_checked": compliance_data["policies_checked"],
                    "violations": violation_list,
                    "audit_receipt_id": compliance_data["audit_receipt_id"]
                }

            all_logs.append(log)

        return all_logs
    finally:
        conn.close()


def tool_get_transaction_analytics(customer_id: str) -> Dict[str, Any]:
    """Get transaction analytics for a user including category breakdown and monthly trends"""
    conn = get_db_connection()
    try:
        conn.row_factory = dict_factory
        cursor = conn.cursor()

        # Get spending by category
        cursor.execute("""
        SELECT
            category,
            COUNT(*) as transaction_count,
            SUM(amount_inr) as total_amount,
            AVG(amount_inr) as average_amount
        FROM
            transactions
        WHERE
            user_id = ?
        GROUP BY
            category
        ORDER BY
            total_amount DESC
        """, (customer_id,))

        category_data = cursor.fetchall()

        # Get monthly spending
        cursor.execute("""
        SELECT
            month,
            COUNT(*) as transaction_count,
            SUM(amount_inr) as total_amount,
            AVG(amount_inr) as average_amount
        FROM
            transactions
        WHERE
            user_id = ?
        GROUP BY
            month
        ORDER BY
            month
        """, (customer_id,))

        monthly_data = cursor.fetchall()

        # Get payment method breakdown
        cursor.execute("""
        SELECT
            method as payment_method,
            COUNT(*) as transaction_count,
            SUM(amount_inr) as total_amount
        FROM
            transactions
        WHERE
            user_id = ?
        GROUP BY
            method
        ORDER BY
            total_amount DESC
        """, (customer_id,))

        payment_method_data = cursor.fetchall()

        # Get user financial summary
        cursor.execute("""
        SELECT
            COUNT(*) as total_transactions,
            SUM(amount_inr) as total_spend,
            MAX(amount_inr) as largest_transaction,
            MIN(date) as earliest_transaction,
            MAX(date) as latest_transaction
        FROM
            transactions
        WHERE
            user_id = ?
        """, (customer_id,))

        summary = cursor.fetchone()

        # Get transaction stats relative to user income
        cursor.execute("""
        SELECT
            u.annual_income_inr as annual_income,
            SUM(t.amount_inr) as total_spent,
            (SUM(t.amount_inr) / u.annual_income_inr * 100) as spend_percent_of_income
        FROM
            user_details u
        JOIN
            transactions t ON u.user_id = t.user_id
        WHERE
            u.user_id = ?
        GROUP BY
            u.user_id
        """, (customer_id,))

        income_stats = cursor.fetchone()

        return {
            "summary": summary or {},
            "income_stats": income_stats or {},
            "categories": category_data,
            "monthly_trend": monthly_data,
            "payment_methods": payment_method_data
        }
    finally:
        conn.close()


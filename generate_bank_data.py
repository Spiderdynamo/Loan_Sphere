"""
=================================================================================
 generate_bank_data.py - creates a small synthetic bank statement for testing
=================================================================================
Run this script directly (python generate_bank_data.py) to produce a CSV of
100 fake transactions in uploads/bank_statement.csv. It's handy for quickly
exercising the charting logic without needing a real bank statement PDF on
hand. It is NOT used by the Flask app itself - it's a standalone dev tool.
=================================================================================
"""

import os
import random

import pandas as pd

# Make sure the uploads folder exists before writing into it.
os.makedirs("uploads", exist_ok=True)

CATEGORIES = [
    "Food",
    "Shopping",
    "Electricity",
    "Travel",
    "Recharge",
    "Medical",
    "Entertainment",
    "Fuel",
]


def generate_sample_rows(count=100):
    """Build `count` random transaction rows for a fake bank statement."""
    rows = []
    for _ in range(count):
        rows.append({
            "Date": f"2026-07-{random.randint(1, 30):02d}",
            "Category": random.choice(CATEGORIES),
            "Amount": random.randint(100, 5000),
            "Type": "Debit",
        })
    return rows


if __name__ == "__main__":
    df = pd.DataFrame(generate_sample_rows())
    output_path = os.path.join("uploads", "bank_statement.csv")
    df.to_csv(output_path, index=False)
    print(f"Synthetic bank statement created successfully at {output_path}")

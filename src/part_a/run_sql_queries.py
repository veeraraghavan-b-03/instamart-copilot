import sqlite3
import pandas as pd

connection = sqlite3.connect("data/instamart.db")

query = """
WITH daily_revenue AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS day,
        SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    day,
    ROUND(revenue, 2) AS daily_revenue,
    ROUND(
        SUM(revenue) OVER (
            ORDER BY day
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        2
    ) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
"""

result = pd.read_sql_query(query, connection)

print("========== Q7 RESULTS ==========")

print("\nFirst 5 days:")
print(result.head().to_string(index=False))

print("\nLast 5 days:")
print(result.tail().to_string(index=False))

print("\nTotal daily records:", len(result))
print("Final cumulative revenue:", result["cumulative_revenue"].iloc[-1])

connection.close()

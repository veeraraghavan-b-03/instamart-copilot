
import sqlite3
import math

# Connect to our SQLite database
connection = sqlite3.connect("data/instamart.db")

print("========== DATABASE VALIDATION ==========\n")

try:
    # Check 1: Customer count
    customer_count = connection.execute(
        "SELECT COUNT(*) FROM customers"
    ).fetchone()[0]

    assert customer_count == 1200
    print("CHECK 1 PASSED: Customer count =", customer_count)


    # Check 2: Product count
    product_count = connection.execute(
        "SELECT COUNT(*) FROM products"
    ).fetchone()[0]

    assert product_count == 40
    print("CHECK 2 PASSED: Product count =", product_count)


    # Check 3: Order count
    order_count = connection.execute(
        "SELECT COUNT(*) FROM orders"
    ).fetchone()[0]

    assert order_count == 12309
    print("CHECK 3 PASSED: Order count =", order_count)


    # Check 4: Every order ID must be unique
    unique_orders = connection.execute(
        "SELECT COUNT(DISTINCT order_id) FROM orders"
    ).fetchone()[0]

    assert unique_orders == 12309
    print("CHECK 4 PASSED: All order IDs are unique")


    # Check 5: Completed orders
    completed_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM orders
        WHERE status = 'completed'
        """
    ).fetchone()[0]

    assert completed_count == 11110
    print("CHECK 5 PASSED: Completed orders =", completed_count)


    # Check 6: Cancelled orders
    cancelled_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM orders
        WHERE status = 'cancelled'
        """
    ).fetchone()[0]

    assert cancelled_count == 1199
    print("CHECK 6 PASSED: Cancelled orders =", cancelled_count)


    # Check 7: No pending orders
    pending_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM orders
        WHERE status = 'pending'
        """
    ).fetchone()[0]

    assert pending_count == 0
    print("CHECK 7 PASSED: No pending orders")


    # Check 8: No missing order amounts
    missing_amounts = connection.execute(
        """
        SELECT COUNT(*)
        FROM orders
        WHERE order_amount IS NULL
        """
    ).fetchone()[0]

    assert missing_amounts == 0
    print("CHECK 8 PASSED: No missing order amounts")


    # Check 9: No missing delivery times in completed orders
    missing_completed_delivery = connection.execute(
        """
        SELECT COUNT(*)
        FROM orders
        WHERE status = 'completed'
        AND delivery_minutes IS NULL
        """
    ).fetchone()[0]

    assert missing_completed_delivery == 0
    print("CHECK 9 PASSED: Completed delivery times are complete")


    # Check 10: Cancelled orders must have missing delivery times
    missing_cancelled_delivery = connection.execute(
        """
        SELECT COUNT(*)
        FROM orders
        WHERE status = 'cancelled'
        AND delivery_minutes IS NULL
        """
    ).fetchone()[0]

    assert missing_cancelled_delivery == 1199
    print("CHECK 10 PASSED: Cancelled delivery times remain missing")


    # Check 11: Completed order revenue
    revenue = connection.execute(
        """
        SELECT SUM(order_amount)
        FROM orders
        WHERE status = 'completed'
        """
    ).fetchone()[0]

    assert math.isclose(
        revenue,
        2998668.80,
        abs_tol=0.01
    )

    print("CHECK 11 PASSED: Completed revenue =", round(revenue, 2))


    # Final success message
    print("\nALL CHECKPOINTS PASSED")


finally:
    connection.close()


-- Q1: Customers with high-value completed orders

SELECT
    customer_id,
    SUM(order_amount) AS total_spend,
    COUNT(*) AS qualifying_orders
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;


-- Q2: Revenue by city and product category

SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS total_revenue
FROM orders AS o
JOIN customers AS c
    ON o.customer_id = c.customer_id
JOIN products AS p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY
    c.city,
    p.category
ORDER BY
    c.city,
    total_revenue DESC;


-- Q3: Top-selling products

SELECT
    p.product_id,
    p.product_name,
    p.category,
    SUM(o.quantity) AS total_quantity_sold,
    ROUND(SUM(o.order_amount), 2) AS total_revenue
FROM orders AS o
JOIN products AS p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY
    p.product_id,
    p.product_name,
    p.category
ORDER BY
    total_quantity_sold DESC
LIMIT 11;


-- Q4: Customers with no completed orders

SELECT
    c.customer_id,
    c.city,
    c.age,
    c.gender
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;




-- Q5: Customers spending above the average

WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT
        AVG(total_spend) AS avg_spend
    FROM customer_spend
)
SELECT
    COUNT(*) AS total_customers,
    ROUND((SELECT avg_spend FROM average_spend), 2) AS average_customer_spend
FROM customer_spend
WHERE total_spend > (SELECT avg_spend FROM average_spend);

-- Q6: Rank customers by total completed spend within each city

WITH customer_spend AS (
    SELECT
        c.customer_id,
        c.city,
        SUM(o.order_amount) AS total_spend
    FROM orders AS o
    JOIN customers AS c
        ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY
        c.customer_id,
        c.city
)
SELECT
    customer_id,
    city,
    ROUND(total_spend, 2) AS total_spend,

    RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS spend_rank,

    DENSE_RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS dense_rank,

    ROW_NUMBER() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS row_number

FROM customer_spend
ORDER BY
    city,
    spend_rank,
    customer_id;

    -- Q7: Daily revenue and cumulative running total

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
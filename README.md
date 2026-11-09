# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds a customer intelligence and support copilot using synthetic Instamart customer, product, order, and support-ticket data.

The project combines data engineering, SQL analytics, exploratory data analysis, machine learning, customer segmentation, support ticket classification, retrieval-augmented generation (RAG), and an automated refund-support agent.

### Project Components

- **Part A:** Data auditing, cleaning, SQLite database creation, SQL analytics, and Pandas analysis.
- **Part B:** RFM-based customer segmentation and machine-learning-based lapse prediction.
- **Part C:** Support ticket classification, PII redaction, retry handling, token-budget management, and ticket routing.
- **Part D:** Policy document retrieval, RAG evaluation, ReAct refund agent, tool execution, checkpointing, and recovery.

The project uses the supplied offline `mock_llm.py`. No paid LLM API or external model service is required.

## Reproduce

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd instamart-copilot
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate project data

```bash
python generate_data.py
```

This generates the synthetic customer, product, order, ticket, policy evaluation, and agent scenario datasets, along with the policy documents.

### 5. Run the project

Run the scripts from the project root in the following order.

**Part A – Data Engineering and SQL**

```powershell
python src/part_a/audit_raw.py
python src/part_a/audit_quality.py
python src/part_a/clean_customers.py
python src/part_a/clean_products.py
python src/part_a/clean_orders.py
python src/part_a/build_database.py
python src/part_a/validate_database.py
python src/part_a/run_sql_queries.py
python src/part_a/pandas_analysis.py
```

**Part B – Customer Intelligence**

```powershell
python src/part_b/segment_customers.py
python src/part_b/predict_lapse.py
```

**Part C – Support Ticket Triage**

```powershell
python src/part_c/triage_tickets.py
```

**Part D – Policy RAG and Refund Agent**

```powershell
python src/part_d/policy_rag.py
python src/part_d/refund_agent.py
python src/part_d/test_checkpoint.py
python src/part_d/test_safety_limits.py
```

Run the Part A data preparation scripts before the later parts because they create the cleaned datasets and SQLite database required by the downstream components.

The supplied `generate_data.py` and `mock_llm.py` should remain unchanged.

## Part A

### Data Foundation and SQL Analytics

Implemented raw-data auditing, cleaning, database creation, SQL analysis, and Pandas validation.

Key activities:

- Audited missing values, duplicates, invalid values, and inconsistent categorical data.
- Cleaned customer, product, and order datasets.
- Standardized city and gender values.
- Handled missing customer ages using city-level median imputation.
- Removed duplicate order records and retained the latest record for repeated order IDs.
- Recovered missing order amounts using product price, quantity, and discount.
- Corrected invalid delivery times and handled missing delivery times according to order status.
- Created and validated a SQLite database.
- Implemented seven SQL analytical queries.
- Cross-checked SQL results using Pandas.
- Performed revenue analysis, customer spending analysis, distribution analysis, outlier detection, and correlation analysis.

### Data Quality Results

| Dataset | Final records |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 12,309 |

Order cleaning:

- Original order rows: 13,084
- Exact duplicate rows removed: 119
- Final unique order IDs: 12,309
- Completed orders: 11,110
- Cancelled orders: 1,199

Database validation: **11 out of 11 checks passed.**

### Key SQL and Pandas Findings

| Metric | Result |
|---|---:|
| Completed-order revenue | ₹29,98,668.80 |
| Completed orders | 11,110 |
| Customers above average spending | 550 |
| Average customer spending | ₹2,552.06 |
| Customers with no completed orders | 25 |
| Upper outlier threshold | ₹645.30 |
| Upper outliers | 897 |

Additional analysis:

- City-wise revenue and product-category revenue were calculated using SQL joins.
- CTEs were used to identify customers spending above the average.
- Window functions `RANK()`, `DENSE_RANK()`, and `ROW_NUMBER()` were used to rank customer spending within each city.
- A daily revenue query calculated cumulative running revenue.
- Pandas `groupby`, `pivot_table`, descriptive statistics, IQR, and correlation analysis were used to validate and explore the results.

The SQL and Pandas revenue totals matched at ₹29,98,668.80.

The strongest absolute correlation among the selected numerical variables was between order amount and discount percentage (-0.095). This is a weak negative correlation and does not establish causation.

## Part B

### Customer Segmentation and Lapse Prediction

#### 1. RFM Customer Segmentation

Implemented customer segmentation using Recency, Frequency, and Monetary (RFM) analysis.

- Snapshot date: 1 October 2025.
- Used completed orders to calculate RFM metrics.
- Standardized the RFM features using `StandardScaler`.
- Evaluated K-Means clustering for K values from 2 to 8.
- Compared inertia and silhouette scores.
- Selected K=4 based on the highest silhouette score among the tested values and segment interpretability.
- Used PCA to inspect the cluster structure.

Selected clustering results:

| K | Inertia | Silhouette |
|---|---:|---:|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

PCA explained variance:

- First component: 63.4%
- Second component: 21.8%
- Combined: 85.2%

Final customer segments:

| Segment | Customer count | Mean Recency | Mean Frequency | Mean Monetary |
|---|---:|---:|---:|---:|
| Occasional Customers | 436 | 28.86 | 3.77 | ₹1,057.62 |
| High-Value Customers | 138 | 24.12 | 6.77 | ₹7,129.70 |
| Loyal Customers | 419 | 8.77 | 19.40 | ₹3,410.02 |
| At-Risk Customers | 182 | 109.04 | 2.20 | ₹685.99 |

A premium customer flag was also created to identify the high-value segment.

Output:

- `outputs/customer_segments.csv`

#### 2. Customer Lapse Prediction

Built machine-learning models to identify customers who did not place a completed order during September 2025.

Target definition:

- Cutoff date: 1 September 2025.
- Lapsed = 1 if the customer had no completed order from 1 September through 30 September 2025.
- Active = 0 otherwise.

Dataset:

| Metric | Result |
|---|---:|
| Eligible customers | 1,156 |
| Lapsed customers | 443 |
| Active customers | 713 |
| Lapse rate | 38.32% |
| Training records | 924 |
| Test records | 232 |

Features included:

- Recency
- Frequency
- Monetary value
- Average discount
- Cancellation rate
- Average delivery time
- Most frequent product category
- City
- Age
- Gender
- Customer tenure

All model features were calculated using information available before the cutoff date.

Models evaluated:

- Baseline classifier
- Logistic Regression
- Random Forest
- Oversampled Logistic Regression

Model comparison:

| Model | Test Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Baseline | 61.6% | — | 0.000 | 0.000 |
| Logistic Regression | 76.3% | 0.655 | 0.809 | 0.724 |
| Random Forest | 77.2% | 0.684 | 0.753 | 0.717 |
| Oversampled Logistic Regression | 72.8% | 0.592 | 0.944 | 0.727 |

The oversampled Logistic Regression model was selected because of its high recall. It identifies more potentially lapsed customers, although this comes with lower precision.

Oversampling was applied only to the training data. The test set remained unchanged.

Top Random Forest features:

1. Frequency
2. Monetary value
3. Recency
4. Cancellation rate
5. Average discount

The selected model was saved using Joblib and reloaded successfully. Predictions after reloading matched the original predictions.

A retention list was also generated by ranking premium test customers using predicted lapse probabilities.

Outputs:

- `outputs/lapse_features.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

## Part C

### Support Ticket Triage

Implemented an automated ticket classification and routing pipeline using the supplied offline mock LLM.

Dataset:

- Total support tickets: 164.
- Allowed categories: Billing, Delivery, Technical, and Fraud.

### Processing Pipeline

1. Load support tickets.
2. Detect empty and excessively long messages.
3. Redact personally identifiable information (PII).
4. Construct zero-shot and few-shot prompts.
5. Call the mock LLM.
6. Retry transient failures using exponential backoff and jitter.
7. Validate model output.
8. Enforce token budgets.
9. Classify and route tickets.
10. Save results to CSV.

PII redaction:

| PII type | Replacements |
|---|---:|
| Email | 55 |
| Phone | 91 |
| Order ID | 132 |
| Total | 278 |

No PII remained in the redacted ticket text.

Empty and long-message guardrails:

- Empty messages: 4
- Long messages: 4

### Zero-Shot vs Few-Shot Evaluation

| Metric | Zero-Shot | Few-Shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Correct classifications | 126 | 147 |
| Accuracy | 85.71% | 100% |
| Total tokens | 14,860 | 25,624 |
| Tokens per classified ticket | 101.09 | 174.31 |

Few-shot prompting improved accuracy by 14.29 percentage points, with approximately 72.4% more token usage.

Zero-shot classification errors:

- Technical classified as Billing: 10
- Fraud classified as Billing: 11

Few-shot classification correctly classified all 147 evaluated tickets.

### Reliability and Token Budget

- Transient errors: 19 per full run.
- All transient errors recovered through retries.
- Maximum retry attempts: 3 retries after the initial attempt.
- A separate zero-shot budget demonstration used a 6,000-token budget.
- The final model call caused token usage to reach 6,087 before the next budget check stopped further processing.

### Final Ticket Routing

| Route | Tickets |
|---|---:|
| Support Queue | 102 |
| Security Team | 34 |
| Human Review | 17 |
| Premium Desk | 11 |
| Total | 164 |

Fraud tickets were routed to the Security Team, premium customers to the Premium Desk, and invalid or unclassified cases to Human Review.

Output files:

- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`

## Part D

### Policy RAG Assistant and Refund Agent

#### 1. Policy Document Retrieval

Implemented a retrieval-augmented generation pipeline using five policy documents:

- Delivery policy
- Refund policy
- Payment policy
- Privacy policy
- Escalation policy

Chunking configuration:

- Chunk size: 40 words
- Overlap: 10 words
- Total chunks: 15
- Retrieval method: TF-IDF
- Similarity: Cosine similarity
- Retrieved chunks per query: Top 3

TF-IDF uses term frequency to represent how often a term appears in a chunk and inverse document frequency to reduce the influence of words common across documents.

Retrieved policy chunks are provided to the mock LLM to generate policy-grounded answers.

#### RAG Evaluation

The evaluation dataset contained 14 questions, including 2 questions that should be treated as unanswerable.

| Evaluation metric | 10-word overlap | Zero overlap |
|---|---:|---:|
| Expected document in top 3 | 12/12 | 12/12 |
| Expected fact in answer | 12/12 | 10/12 |
| Unanswerable questions handled with fallback | 2/2 | 2/2 |

Overlap improved factual answer coverage from 10/12 to 12/12.

The zero-overlap evaluation demonstrated that policy facts can become separated across chunk boundaries, making some answers harder to retrieve accurately.

#### 2. Refund Agent and Tool Calling

Implemented a ReAct-style refund-support agent using the supplied offline mock LLM.

Available tools:

1. `get_order` – retrieves order information from SQLite.
2. `search_policy` – retrieves relevant policy chunks.
3. `calculate_refund` – calculates the eligible refund.
4. `issue_refund` – records an approved refund.
5. `escalate_to_human` – records escalation to a human lead.

Business rules:

- Only completed orders are eligible.
- Delivery must exceed 45 minutes for a refund.
- Eligible refunds are calculated using the full order amount.
- Refunds above ₹1,000 are escalated to a human lead.
- Missing orders and invalid cases are handled safely.
- Cancelled orders are never refunded.

#### Refund Scenario Results

| Scenario | Expected | Actual | Model Calls | Tokens |
|---|---|---|---:|---:|
| S1 | REFUND | REFUND | 5 | 1,047 |
| S2 | REFUND | REFUND | 5 | 1,047 |
| S3 | ESCALATE | ESCALATE | 5 | 1,034 |
| S4 | DENIED | DENIED | 4 | 725 |
| S5 | DENIED | DENIED | 3 | 427 |
| S6 | NOT_FOUND | NOT_FOUND | 2 | 132 |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | 6 | 3,180 |

All seven scenarios produced the expected business outcome.

Refund and escalation ledger:

| Action | Order | Amount |
|---|---|---:|
| Refund issued | ORD-100048 | ₹128.25 |
| Refund issued | ORD-100067 | ₹135.00 |
| Human escalation | ORD-100145 | ₹1,999.00 |

Outputs:

- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`

#### 3. Checkpoint and Resume

Implemented checkpoint persistence so an agent can continue after an interruption without repeating completed tool executions.

The checkpoint stores:

- Run ID
- Scenario ID
- Complete conversation messages
- Tokens used
- Step number
- Tools already executed
- Execution history

Demonstration using S1:

1. Agent executed `get_order`.
2. Agent executed `search_policy`.
3. A simulated crash was triggered after the second tool.
4. The checkpoint was saved successfully.
5. `resume_agent()` loaded the checkpoint.
6. The agent continued with `calculate_refund` and `issue_refund`.

Final execution sequence:

`get_order → search_policy → calculate_refund → issue_refund`

Final result: `REFUND: INR 128.25 issued.`

The first two tools were not repeated after resuming.

#### 4. Safety Limit Tests

| Safety control | Test configuration | Result |
|---|---|---|
| Token limit | 200 tokens | FALLBACK_TOKEN_LIMIT |
| Time limit | 0 seconds | FALLBACK_TIMEOUT |
| Maximum iterations | 6 iterations | FALLBACK_MAX_ITERATIONS |

All three safety-limit tests passed.

#### 5. ReAct vs Plan-and-Execute

ReAct combines reasoning and tool execution in an iterative loop. The model selects a tool, receives its observation, and decides what to do next.

Plan-and-Execute creates a plan first and then executes the planned steps, potentially replanning when something goes wrong.

For this refund workflow, ReAct was selected because the agent must respond to intermediate results. For example, a missing order, cancelled order, late delivery, or high-value refund can require different actions.

Although ReAct can require more model calls, it provides a flexible way to handle branching decisions and tool observations before performing a refund or escalation.

## Limitations and Next Steps

### Limitations

- The project uses synthetic data and may not represent real-world customer behaviour.
- The supplied mock LLM is an offline simulation and does not represent the performance of a production language model.
- TF-IDF retrieval relies on lexical similarity and may struggle with semantic synonyms or paraphrased questions.
- The lapse prediction dataset is relatively small, and model performance may change with real customer data.
- The premium retention list is illustrative: the premium flag was derived from the full RFM window, which includes September, the month used to define the lapse outcome. It should not be treated as a leakage-free retention evaluation.
- The refund agent uses simulated tool actions and a local JSONL ledger rather than a real payment or refund service.
- The token budget is checked after each model response, so a final call can exceed the configured budget before processing stops.

### Next Steps

- Evaluate the models using larger and more representative customer datasets.
- Improve retrieval with semantic embeddings and more robust document evaluation.
- Add monitoring for model quality, token usage, and agent failures.
- Introduce stronger transaction controls and idempotency for real refund operations.
- Integrate production databases and approved support systems.
- Deploy the copilot through a secure API or user-facing application.

---

**Project status:** All four project parts implemented and tested using the supplied synthetic datasets and offline mock LLM.


import json
import math
import random
import re
import time
import sys
from pathlib import Path

# Locate the project root.
ROOT = Path(__file__).resolve().parents[2]

# Add project root to Python's module search path.
sys.path.insert(0, str(ROOT))

import pandas as pd
import mock_llm
from sklearn.metrics import accuracy_score, confusion_matrix



# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------


TICKETS_FILE = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS_FILE = ROOT / "outputs" / "customer_segments.csv"
OUTPUT_DIR = ROOT / "outputs"

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. PRIVACY REDACTION
# --------------------------------------------------

PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"\b[6-9]\d{9}\b"),
    "order_id": re.compile(r"ORD-\d{6}")
}

PII_REPLACEMENTS = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "order_id": "[ORDER_ID]"
}


def redact_text(text):
    """Redact email, phone and order ID in the required order."""

    redacted = text
    counts = {}

    for name, pattern in PII_PATTERNS.items():
        redacted, count = pattern.subn(
            PII_REPLACEMENTS[name],
            redacted
        )
        counts[name] = count

    return redacted, counts


def contains_pii(text):
    """Return True if any original PII pattern remains."""

    return any(
        pattern.search(text)
        for pattern in PII_PATTERNS.values()
    )


# --------------------------------------------------
# 3. SYSTEM PROMPTS
# --------------------------------------------------

ZERO_SHOT_PROMPT = """
You are a customer support ticket classifier.

CLASSIFY each customer message into exactly one category:
billing, delivery, technical, fraud.

Return JSON only, in exactly this form:
{"category": "...", "reason": "..."}

The category must be one of the four allowed categories.
Do not include markdown or any text outside the JSON object.
"""


FEW_SHOT_PROMPT = """
You are a customer support ticket classifier.

CLASSIFY each customer message into exactly one category:
billing, delivery, technical, fraud.

Return JSON only, in exactly this form:
{"category": "...", "reason": "..."}

The category must be one of the four allowed categories.
Do not include markdown or any text outside the JSON object.

EXAMPLES:
Example 1:
Customer message: My card was used without my permission.
Answer: {"category": "fraud", "reason": "Unauthorized card usage"}

Example 2:
Customer message: My invoice has the wrong amount.
Answer: {"category": "billing", "reason": "Incorrect invoice amount"}

Example 3:
Customer message: The application crashes when I log in.
Answer: {"category": "technical", "reason": "Application crash"}
"""


# --------------------------------------------------
# 4. RETRY FUNCTION
# --------------------------------------------------

def call_with_retry(messages, ticket_id, max_retries=3):
    """
    Retry temporary model errors.
    Maximum three retries after the initial attempt.
    """

    for retry_number in range(max_retries + 1):

        try:
            return mock_llm.chat(messages)

        except mock_llm.TransientLLMError:

            if retry_number == max_retries:
                raise

            wait_time = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"RETRY ticket={ticket_id} "
                f"retry={retry_number} "
                f"wait={wait_time:.4f}s"
            )

            time.sleep(wait_time)


# --------------------------------------------------
# 5. OUTPUT VALIDATION
# --------------------------------------------------

def validate_response(response_text):

    # First, try JSON validation.
    try:
        parsed = json.loads(response_text)

        if (
            isinstance(parsed, dict)
            and parsed.get("category") in ALLOWED_CATEGORIES
        ):
            return parsed["category"], "classified_json"

        # Valid JSON but invalid category: do not guess.
        return None, "flagged_invalid_output"

    except (json.JSONDecodeError, TypeError):

        # For non-JSON text, extract lowercase words.
        words = re.findall(r"[a-z]+", response_text.lower())

        found = ALLOWED_CATEGORIES.intersection(words)

        if len(found) == 1:
            return next(iter(found)), "classified_keyword"

        return None, "flagged_invalid_output"


# --------------------------------------------------
# 6. COMPLETE TICKET PIPELINE
# --------------------------------------------------

def run_pipeline(system_prompt, token_budget=None):

    tickets = pd.read_csv(
        TICKETS_FILE,
        keep_default_na=False
    )

    # Reset mock immediately before each independent run.
    mock_llm.reset()

    running_tokens = 0
    results = []

    for _, ticket in tickets.iterrows():

        ticket_id = ticket["ticket_id"]
        original_text = ticket["text"]

        # Apply privacy redaction.
        redacted_text, redaction_counts = redact_text(original_text)

        # Assert that no PII remains in the redacted text.
        assert not contains_pii(redacted_text), (
            f"PII remains in ticket {ticket_id}"
        )

        category = None
        status = None
        ticket_tokens = 0

        # Guardrail 1: empty input.
        if not original_text.strip():

            status = "skipped_empty"

        # Guardrail 2: long input.
        elif mock_llm.est_tokens(original_text) > MAX_INPUT_TOKENS:

            status = "flagged_long"

        else:

            # Budget check before making a model call.
            if (
                token_budget is not None
                and running_tokens >= token_budget
            ):
                status = "budget_exhausted"

            else:

                # User message contains ONLY redacted ticket text.
                assert not contains_pii(redacted_text)

                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": redacted_text
                    }
                ]

                response = call_with_retry(messages, ticket_id)

                ticket_tokens = response.usage.total_tokens
                running_tokens += ticket_tokens

                response_text = response.choices[0].message.content

                category, status = validate_response(response_text)

        results.append({
            "ticket_id": ticket_id,
            "customer_id": ticket["customer_id"],
            "status": status,
            "category": category if category else "",
            "tokens_used": ticket_tokens,
            "true_category": ticket["true_category"]
        })

    return pd.DataFrame(results), running_tokens


# --------------------------------------------------
# 7. REPORT RUN METRICS
# --------------------------------------------------

def report_metrics(name, results, total_tokens):

    classified = results[
        results["status"].isin([
            "classified_json",
            "classified_keyword"
        ])
    ].copy()

    correct = (
        classified["category"] == classified["true_category"]
    ).sum()

    count = len(classified)

    accuracy = (
        accuracy_score(
            classified["true_category"],
            classified["category"]
        )
        if count else 0
    )

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("Total tickets:", len(results))
    print("Classified tickets:", count)
    print("Correct classifications:", correct)
    print(f"Accuracy: {accuracy:.2%}")
    print("Total tokens:", total_tokens)
    print(
        "Tokens per classified ticket:",
        round(total_tokens / count, 2) if count else 0
    )

    print("\nStatus counts:")
    print(results["status"].value_counts().to_string())

    return classified


# --------------------------------------------------
# 8. ZERO-SHOT VS FEW-SHOT COMPARISON
# --------------------------------------------------

print("\nRUN 1: ZERO-SHOT WITHOUT TOKEN BUDGET")

zero_results, zero_tokens = run_pipeline(
    ZERO_SHOT_PROMPT
)

zero_classified = report_metrics(
    "ZERO-SHOT RESULTS",
    zero_results,
    zero_tokens
)

print("\nZERO-SHOT CONFUSION MATRIX")
print(
    pd.DataFrame(
        confusion_matrix(
            zero_classified["true_category"],
            zero_classified["category"],
            labels=["billing", "delivery", "technical", "fraud"]
        ),
        index=["actual_billing", "actual_delivery",
               "actual_technical", "actual_fraud"],
        columns=["pred_billing", "pred_delivery",
                 "pred_technical", "pred_fraud"]
    )
)


print("\nRUN 2: FEW-SHOT WITHOUT TOKEN BUDGET")

few_results, few_tokens = run_pipeline(
    FEW_SHOT_PROMPT
)

few_classified = report_metrics(
    "FEW-SHOT RESULTS",
    few_results,
    few_tokens
)


# --------------------------------------------------
# 9. REDACTION AUDIT
# --------------------------------------------------

tickets = pd.read_csv(
    TICKETS_FILE,
    keep_default_na=False
)

redaction_totals = {
    "email": 0,
    "phone": 0,
    "order_id": 0
}

remaining_pii = 0

for text in tickets["text"]:

    redacted, counts = redact_text(text)

    for name in redaction_totals:
        redaction_totals[name] += counts[name]

    remaining_pii += int(contains_pii(redacted))

print("\nREDACTION AUDIT")
print(redaction_totals)

total_redactions = sum(redaction_totals.values())

print("Total substitutions:", total_redactions)
print("Remaining PII matches:", remaining_pii)

assert remaining_pii == 0


# --------------------------------------------------
# 10. ZERO-SHOT TOKEN BUDGET DEMO
# --------------------------------------------------

print("\nRUN 3: ZERO-SHOT TOKEN BUDGET DEMO")

budget_results, budget_tokens = run_pipeline(
    ZERO_SHOT_PROMPT,
    token_budget=6000
)

budget_results.to_csv(
    OUTPUT_DIR / "triage_budget_demo.csv",
    index=False
)

print("Budget: 6000")
print("Tokens consumed:", budget_tokens)
print(
    "Budget exhausted tickets:",
    (budget_results["status"] == "budget_exhausted").sum()
)


# --------------------------------------------------
# 11. FINAL FEW-SHOT RUN AND CUSTOMER ROUTING
# --------------------------------------------------

print("\nRUN 4: FINAL FEW-SHOT RUN")

final_results, final_tokens = run_pipeline(
    FEW_SHOT_PROMPT,
    token_budget=100000
)

segments = pd.read_csv(SEGMENTS_FILE)

# Convert the premium flag safely to boolean.
segments["is_premium"] = (
    segments["is_premium"]
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
)

premium_lookup = (
    segments.set_index("customer_id")["is_premium"].to_dict()
)

segment_name_lookup = (
    segments.set_index("customer_id")["segment_name"].to_dict()
)


def assign_routing(row):

    customer_id = row["customer_id"]
    status = row["status"]
    category = row["category"]

    segment_name = segment_name_lookup.get(customer_id, "")

    # Human review takes priority over all other routing.
    if status in {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted"
    }:
        priority = "NORMAL"
        route = "human_review"

    elif category == "fraud":
        priority = "HIGH"
        route = "security_team"

    elif premium_lookup.get(customer_id, False):
        priority = "HIGH"
        route = "premium_desk"

    else:
        priority = "NORMAL"
        route = "support_queue"

    return pd.Series({
        "segment_name": segment_name,
        "priority": priority,
        "route": route
    })


routing = final_results.apply(assign_routing, axis=1)

final_results = pd.concat(
    [final_results, routing],
    axis=1
)

final_output = final_results[
    [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used"
    ]
]

final_output.to_csv(
    OUTPUT_DIR / "triage_results.csv",
    index=False
)


# --------------------------------------------------
# 12. FINAL CHECKS
# --------------------------------------------------

print("\nFINAL OUTPUT CHECKS")

print("Final rows:", len(final_output))
print("Budget demo rows:", len(budget_results))

print("\nFinal status counts:")
print(final_output["status"].value_counts().to_string())

print("\nRouting counts:")
print(final_output["route"].value_counts().to_string())

print("\nFiles created:")
print("outputs/triage_budget_demo.csv")
print("outputs/triage_results.csv")

assert len(final_output) == 164
assert len(budget_results) == 164

print("\nPART C PIPELINE COMPLETED")

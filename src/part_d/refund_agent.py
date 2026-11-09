
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from policy_rag import (
    load_policy_documents,
    create_chunks,
    PolicyRetriever
)

DB_FILE = ROOT / "data" / "instamart.db"
OUTPUT_DIR = ROOT / "outputs"
LEDGER_FILE = OUTPUT_DIR / "refund_ledger.jsonl"

STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(exist_ok=True)

OUTPUT_DIR.mkdir(exist_ok=True)

# Load policy documents and create the retriever
documents = load_policy_documents()
chunks = create_chunks(documents, overlap=10)
retriever = PolicyRetriever(chunks)


# TOOL 1: Get order details from SQLite
def get_order(order_id):
    connection = sqlite3.connect(DB_FILE)
    connection.row_factory = sqlite3.Row

    try:
        row = connection.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        ).fetchone()

        if row is None:
            return {"error": "order not found"}

        return {
            "order_id": row["order_id"],
            "status": row["status"],
            "order_amount": row["order_amount"],
            "delivery_minutes": row["delivery_minutes"]
        }
    finally:
        connection.close()


# TOOL 2: Search policy documents
def search_policy(query):
    results = retriever.retrieve(query, k=3)
    return {"results": results}


# TOOL 3: Calculate refund amount
def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None:
        return {"refund_inr": 0.0}

    if float(delivery_minutes) <= 45:
        return {"refund_inr": 0.0}

    refund = round(float(order_amount), 2)
    return {"refund_inr": refund}


# Shared function to write refund actions to ledger
def write_ledger(tool_name, order_id, refund_inr):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }

    with open(LEDGER_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return {"refund_inr": round(float(refund_inr), 2)}


# TOOL 4: Issue refund
def issue_refund(order_id, refund_inr):
    return write_ledger(
        "issue_refund", order_id, refund_inr
    )


# TOOL 5: Escalate refund to human
def escalate_to_human(order_id, refund_inr):
    return write_ledger(
        "escalate_to_human", order_id, refund_inr
    )


# Tool definitions
TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Retrieve an order from the SQLite database.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Search the policy documents for relevant rules.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate refund based on order amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Record an automatically approved refund.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Send a refund above INR 1000 to a human lead.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]


# Tool name -> Python function
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


import time
import mock_llm


def save_checkpoint(state):
    """Save the agent's complete progress so it can resume after a crash."""
    checkpoint_file = STATE_DIR / f"{state['run_id']}.json"

    with open(checkpoint_file, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)

    return checkpoint_file


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tools=None
):
    run_id = run_id or scenario_id

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "user_message": user_message,
        "messages": [
            {
                "role": "system",
                "content": (
                    "AGENT. You are an Instamart refund assistant. "
                    "Use the available tools to check orders, retrieve "
                    "refund policies, calculate refunds, and issue or "
                    "escalate approved refunds. Never refund cancelled "
                    "orders. Follow the tool results."
                )
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "steps": [],
        "tools_executed": [],
        "step_number": 0,
        "llm_calls": 0,
        "tokens_used": 0,
        "max_iterations": max_iterations,
        "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds
    }

    return continue_agent(state, crash_after_tools=crash_after_tools)


def continue_agent(state, crash_after_tools=None):
    """Continue an agent run from its current saved state."""
    start_time = time.monotonic()

    def fallback(status, reason):
        return {
            "scenario_id": state["scenario_id"],
            "status": status,
            "final_answer": (
                f"{status}: {reason} "
                "Please hand this case to a human support agent."
            ),
            "steps": state["steps"],
            "tools_executed": state["tools_executed"],
            "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"]
        }

    for iteration in range(
        state["step_number"],
        state["max_iterations"]
    ):
        if time.monotonic() - start_time >= state["time_limit_seconds"]:
            return fallback(
                "FALLBACK_TIMEOUT",
                "The agent exceeded its time limit."
            )

        response = mock_llm.chat(
            state["messages"],
            tools=TOOLS_SPEC
        )

        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens
        state["step_number"] = iteration + 1

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls

        if state["tokens_used"] > state["token_limit"]:
            return fallback(
                "FALLBACK_TOKEN_LIMIT",
                "The agent exceeded its token budget."
            )

        if not tool_calls:
            final_answer = assistant_message.content or (
                "FALLBACK: The model returned no final answer."
            )

            return {
                "scenario_id": state["scenario_id"],
                "status": final_answer.split(":")[0].strip(),
                "final_answer": final_answer,
                "steps": state["steps"],
                "tools_executed": state["tools_executed"],
                "llm_calls": state["llm_calls"],
                "tokens_used": state["tokens_used"]
            }

        assistant_tool_calls = []

        for tool_call in tool_calls:
            arguments = tool_call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            assistant_tool_calls.append({
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": arguments
            })

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_tool_calls
        })

        for tool_call in tool_calls:
            tool_name = tool_call.name
            arguments = tool_call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            if tool_name not in TOOL_MAP:
                observation = {"error": "unknown tool"}
            else:
                observation = TOOL_MAP[tool_name](**arguments)

            state["steps"].append({
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            })

            state["tools_executed"].append(tool_name)

            state["messages"].append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(observation)
            })

            # Save immediately after every tool observation.
            save_checkpoint(state)

            # Simulate a crash only after the checkpoint is safely saved.
            if (
                crash_after_tools is not None
                and len(state["tools_executed"]) >= crash_after_tools
            ):
                raise RuntimeError(
                    "SIMULATED CRASH: checkpoint saved successfully."
                )

    return fallback(
        "FALLBACK_MAX_ITERATIONS",
        "The agent reached its maximum number of iterations."
    )


def resume_agent(run_id):
    """Load a checkpoint and continue without repeating completed tools."""
    checkpoint_file = STATE_DIR / f"{run_id}.json"

    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"No checkpoint found for run: {run_id}"
        )

    with open(checkpoint_file, "r", encoding="utf-8") as file:
        state = json.load(file)

    print("Resuming run:", run_id)
    print("Tools already executed:", state["tools_executed"])

    return continue_agent(state)
    import pandas as pd

    scenario_file = ROOT / "data" / "raw" / "agent_scenarios.csv"
    runs_file = OUTPUT_DIR / "agent_runs.jsonl"

    scenarios = pd.read_csv(scenario_file)

    # Start fresh for this batch
    if runs_file.exists():
        runs_file.unlink()

    if LEDGER_FILE.exists():
        LEDGER_FILE.unlink()

    all_results = []

    for _, scenario in scenarios.iterrows():

        mock_llm.reset()

        result = run_agent(
            user_message=scenario["message"],
            scenario_id=scenario["scenario_id"],
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10
        )

        # Store each completed run
        with open(runs_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(result) + "\n")

        all_results.append(result)

        print("\n" + "=" * 60)
        print("Scenario:", scenario["scenario_id"])
        print("Expected:", scenario["expected_outcome"])
        print("Actual:", result["status"])
        print("Final answer:", result["final_answer"])
        print("Model calls:", result["llm_calls"])
        print("Tokens:", result["tokens_used"])

    print("\n" + "=" * 60)
    print("ALL SCENARIOS COMPLETED")
    print("Runs saved:", runs_file)
    print("Refund ledger saved:", LEDGER_FILE)

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mock_llm
import refund_agent as agent

scenarios = pd.read_csv(
    ROOT / "data" / "raw" / "agent_scenarios.csv"
)

s1 = scenarios.loc[
    scenarios["scenario_id"] == "S1"
].iloc[0]

s7 = scenarios.loc[
    scenarios["scenario_id"] == "S7"
].iloc[0]

# Test 1: Token limit
mock_llm.reset()
token_result = agent.run_agent(
    s1["message"],
    "TEST_TOKEN",
    token_limit=200
)

# Test 2: Time limit
mock_llm.reset()
timeout_result = agent.run_agent(
    s1["message"],
    "TEST_TIMEOUT",
    time_limit_seconds=0
)

# Test 3: Maximum iterations
mock_llm.reset()
iteration_result = agent.run_agent(
    s7["message"],
    "TEST_ITERATIONS",
    max_iterations=6
)

print("\nSAFETY LIMIT TEST RESULTS")
print("-" * 45)

for name, result in [
    ("Token limit", token_result),
    ("Time limit", timeout_result),
    ("Maximum iterations", iteration_result)
]:
    print(
        name,
        "=>",
        result["status"],
        "| Calls:",
        result["llm_calls"],
        "| Tokens:",
        result["tokens_used"]
    )
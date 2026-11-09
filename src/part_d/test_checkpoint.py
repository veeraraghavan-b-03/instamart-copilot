import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mock_llm
import refund_agent as agent

# Read the S1 scenario
scenarios = pd.read_csv(
    ROOT / "data" / "raw" / "agent_scenarios.csv"
)
scenario = scenarios.loc[
    scenarios["scenario_id"] == "S1"
].iloc[0]

run_id = "S1_checkpoint_demo"

# Use a separate ledger so the original refund ledger is untouched.
agent.LEDGER_FILE = ROOT / "outputs" / "checkpoint_demo_ledger.jsonl"

checkpoint_file = ROOT / "state" / f"{run_id}.json"

# Start this demonstration from a clean state.
for path in [checkpoint_file, agent.LEDGER_FILE]:
    if path.exists():
        path.unlink()

mock_llm.reset()

print("\nSTEP 1: Start S1 and simulate a crash after tool 2")

try:
    agent.run_agent(
        user_message=scenario["message"],
        scenario_id="S1",
        run_id=run_id,
        crash_after_tools=2
    )
except RuntimeError as error:
    print(error)

print("\nSTEP 2: Check saved checkpoint")

import json

with open(checkpoint_file, encoding="utf-8") as file:
    saved = json.load(file)

print("Scenario:", saved["scenario_id"])
print("Tools completed:", saved["tools_executed"])
print("Checkpoint exists:", checkpoint_file.exists())

print("\nSTEP 3: Resume the agent")

result = agent.resume_agent(run_id)

print("\nFINAL RESULT")
print("Status:", result["status"])
print("Answer:", result["final_answer"])
print("Tools executed:", result["tools_executed"])
print("Model calls:", result["llm_calls"])
print("Tokens:", result["tokens_used"])
"""Test that counterpart auto-discovery finds the correct run directories.

Usage:
    cd examples/baselines/ppo
    python test_counterpart_discovery.py
"""

import json
from pathlib import Path

BASE = Path("runs")

# Simulate k_suffix logic from ppo_outer_loop_full.py
def get_counterpart_types(eureka_mode: bool, failureselection: bool, k: int):
    k_suffix = f"_k_{k}" if k != 4 else ""
    if eureka_mode:
        return [f"outer-loop_full_failureselection{k_suffix}", f"outer-loop_full{k_suffix}"]
    elif failureselection:
        return [f"eureka_full{k_suffix}", f"outer-loop_full{k_suffix}"]
    else:
        return [f"eureka_full{k_suffix}"]


def find_counterpart(env_id: str, counterpart_types: list[str]) -> Path | None:
    for ct in counterpart_types:
        base = BASE / ct / env_id
        if not base.exists():
            continue
        candidates = [
            d for d in base.iterdir()
            if d.is_dir() and (d / "outer_loop_history.json").exists()
        ]
        if candidates:
            candidates.sort(key=lambda d: (d / "outer_loop_history.json").stat().st_mtime)
            return candidates[-1]
    return None


def check_history(run_dir: Path) -> dict:
    hist = json.load(open(run_dir / "outer_loop_history.json"))
    return {
        "iters": len(hist),
        "iter0_best_id": hist[0].get("best_candidate", {}).get("candidate_id", "?") if hist else "?",
    }


# === Tests ===
print("=" * 70)
print("Counterpart discovery test")
print("=" * 70)

# Test: Eureka K=16 should find outer-loop_full_failureselection_k_16
TASKS = ["PushCube-v1", "AnymalC-Reach-v1", "PushT-v1", "UnitreeG1PlaceAppleInBowl-v1", "PegInsertionSide-v1"]

all_ok = True
for task in TASKS:
    types = get_counterpart_types(eureka_mode=True, failureselection=False, k=16)
    result = find_counterpart(task, types)
    if result is None:
        print(f"  FAIL  {task}: no counterpart found (searched {types})")
        all_ok = False
    else:
        info = check_history(result)
        expected_prefix = "outer-loop_full_failureselection_k_16"
        actual_prefix = result.parent.parent.name
        ok = actual_prefix == expected_prefix
        status = "OK" if ok else "WRONG DIR"
        if not ok:
            all_ok = False
        print(f"  {status:5s}  {task}: {actual_prefix}/{result.name} (iters={info['iters']}, iter0_best={info['iter0_best_id']})")

print()

# Test: VLM+LLM failureselection K=16 should find eureka_full_k_16 (may not exist yet)
print("Reverse check (failureselection K=16 → eureka K=16):")
for task in TASKS[:2]:
    types = get_counterpart_types(eureka_mode=False, failureselection=True, k=16)
    result = find_counterpart(task, types)
    if result is None:
        print(f"  SKIP  {task}: no eureka_full_k_16 yet (expected)")
    else:
        print(f"  OK    {task}: {result.parent.parent.name}/{result.name}")

print()

# Test: K=4 should NOT find K=16 data
print("K=4 isolation check (should NOT find K=16):")
for task in TASKS[:2]:
    types = get_counterpart_types(eureka_mode=True, failureselection=False, k=4)
    result = find_counterpart(task, types)
    if result is None:
        print(f"  SKIP  {task}: no K=4 counterpart")
    else:
        actual_prefix = result.parent.parent.name
        if "k_16" in actual_prefix:
            print(f"  FAIL  {task}: K=4 search found K=16 data! ({actual_prefix})")
            all_ok = False
        else:
            print(f"  OK    {task}: {actual_prefix}/{result.name}")

print()
print("=" * 70)
if all_ok:
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED")
print("=" * 70)

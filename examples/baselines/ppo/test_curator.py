#!/usr/bin/env python3
"""
Test the RewardFamilyCurator using real candidate data from an outer_loop_history.json.

Usage:
    # Dry-run mode (no API key needed, uses mock classification):
    python test_curator.py --dry-run

    # Live mode (requires OPENAI_API_KEY):
    python test_curator.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

# Resolve imports from this directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppo_curator import RewardFamilyCurator


HISTORY_PATH = (
    Path(__file__).resolve().parent
    / "runs"
    / "outer-loop_full_failureselection_failure_and_near_miss_k_16"
    / "PushT-v1"
    / "ppo-vlm-full-failureselection-PushT-v1-9351-PushT-v1-20260401_093535_resume1_20260406_213015"
    / "outer_loop_history.json"
)

# Iterations 3-4 (0-indexed) where mode collapse was observed
TARGET_ITERS = [3, 4]


def load_candidates_from_history(history_path: Path, iter_indices: List[int]) -> List[Dict]:
    """Load candidate data from outer_loop_history.json for the given iterations."""
    with open(history_path) as f:
        history = json.load(f)

    candidates = []
    for idx in iter_indices:
        if idx >= len(history):
            print(f"[warn] Iteration {idx} not found in history (only {len(history)} iters)")
            continue
        entry = history[idx]
        all_cands = entry.get("all_candidates", [])
        print(f"Iter {idx}: {len(all_cands)} candidates, best fitness={entry.get('best_candidate', {}).get('fitness', 'N/A')}")
        for cand in all_cands:
            candidates.append({
                "id": cand.get("candidate_id", len(candidates)),
                "code": cand.get("code", ""),
                "rationale": cand.get("rationale", ""),
                "is_elite": cand.get("is_elite", False),
                "fitness": cand.get("fitness"),
                "source_iter": idx,
            })

    # Re-index
    for i, cand in enumerate(candidates):
        cand["id"] = i

    return candidates


def mock_classify(candidates: List[Dict]) -> Dict:
    """Mock classification for dry-run mode.

    Groups candidates by simple code heuristics (presence of key terms).
    """
    families: Dict[str, Dict] = {}
    duplicate_pairs = []

    # Simple heuristic classification based on code content
    for cand in candidates:
        code = cand.get("code", "") or ""
        cid = cand["id"]

        # Classify by structural signals in the code
        if "curriculum" in code.lower() or "stage" in code.lower() or "phase" in code.lower():
            label = "curriculum_staged"
        elif "align" in code.lower() and "orient" in code.lower():
            label = "distance_plus_alignment"
        elif "tanh" in code.lower() and code.lower().count("tanh") >= 3:
            label = "multi_tanh_shaping"
        elif "clamp" in code.lower() or "clip" in code.lower():
            label = "clamped_distance"
        else:
            label = "basic_distance"

        if label not in families:
            families[label] = {
                "description": f"Candidates using {label.replace('_', ' ')} strategy",
                "members": [],
            }
        families[label]["members"].append(cid)

    # Mock duplicate detection: if two candidates share the same normalized code body
    # (strip whitespace and comments, compare core logic)
    def _normalize_code(code: str) -> str:
        """Rough normalization: remove comments and collapse whitespace."""
        lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            lines.append(stripped)
        return "\n".join(lines)

    seen_codes: Dict[str, int] = {}
    for cand in candidates:
        code = _normalize_code(cand.get("code", "") or "")
        if code in seen_codes:
            duplicate_pairs.append([seen_codes[code], cand["id"]])
        else:
            seen_codes[code] = cand["id"]

    return {"families": families, "duplicate_pairs": duplicate_pairs}


def main():
    parser = argparse.ArgumentParser(description="Test RewardFamilyCurator")
    parser.add_argument("--dry-run", action="store_true", help="Use mock classification instead of LLM")
    parser.add_argument("--history", type=str, default=str(HISTORY_PATH), help="Path to outer_loop_history.json")
    parser.add_argument("--target-k", type=int, default=16, help="Target number of candidates after curation")
    parser.add_argument("--max-per-family", type=int, default=3, help="Max candidates per family")
    args = parser.parse_args()

    history_path = Path(args.history)
    if not history_path.exists():
        print(f"ERROR: History file not found: {history_path}")
        sys.exit(1)

    print(f"Loading candidates from: {history_path}")
    candidates = load_candidates_from_history(history_path, TARGET_ITERS)
    print(f"\nTotal candidates loaded: {len(candidates)}")

    elite_ids = {c["id"] for c in candidates if c.get("is_elite")}
    print(f"Elite candidates: {sorted(elite_ids)}")

    if args.dry_run:
        print("\n--- DRY RUN MODE (mock classification) ---\n")

        # Use mock classification
        classification = mock_classify(candidates)

        # Print family classification
        print("Family Classification:")
        for label, info in classification["families"].items():
            members = info["members"]
            print(f"  {label}: {len(members)} members -> {members}")
            print(f"    {info['description']}")

        print(f"\nDuplicate pairs: {classification['duplicate_pairs']}")

        # Create curator with mock - we monkey-patch classify_into_families
        curator = RewardFamilyCurator.__new__(RewardFamilyCurator)
        curator.target_k = args.target_k
        curator.max_per_family = args.max_per_family
        curator.enable_refill = True
        curator.last_debug_info = None
        curator.client = None
        curator.model = "mock"

        # Monkey-patch to use mock classification
        curator.classify_into_families = lambda cands: mock_classify(cands)

        curated = curator.curate(candidates, elite_ids=elite_ids)

    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set. Use --dry-run for testing without API key.")
            sys.exit(1)

        print(f"\n--- LIVE MODE (LLM classification) ---\n")
        curator = RewardFamilyCurator(
            api_key=api_key,
            model="gpt-5.4",
            target_k=args.target_k,
            max_per_family=args.max_per_family,
        )
        curated = curator.curate(candidates, elite_ids=elite_ids)

    # Print results
    print(f"\n{'='*60}")
    print(f"CURATION RESULTS")
    print(f"{'='*60}")
    print(f"Input:  {len(candidates)} candidates")
    print(f"Output: {len(curated)} candidates")

    if curator.last_debug_info:
        debug = curator.last_debug_info
        print(f"\nFamilies ({len(debug['families'])}):")
        for label, info in debug["families"].items():
            print(f"  {label}: {info['member_count']} members")
            print(f"    {info['description']}")
        print(f"\nDuplicate pairs removed: {debug['duplicate_pairs']}")
        print(f"Elite IDs preserved: {debug['elite_ids']}")
        print(f"Kept IDs: {debug['kept_ids']}")

    print(f"\nCurated candidates:")
    for cand in curated:
        elite_marker = " [ELITE]" if cand.get("is_elite") else ""
        family = cand.get("reward_family", "?")
        fitness = cand.get("fitness", "N/A")
        print(f"  id={cand['id']:2d}  family={family:<30s}  fitness={fitness}{elite_marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()

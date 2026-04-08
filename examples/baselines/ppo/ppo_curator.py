"""
Reward Family Curator: diversity-preserving filter for LLM-generated reward candidates.

Sits between LLM candidate generation and PPO training in the outer loop.
Uses an LLM call to classify candidates into "reward families" by structural
strategy, then enforces balanced allocation across families and removes
near-duplicates so that the PPO training pool stays diverse and avoids
mode collapse.

Elite candidates (is_elite=True) are never removed.

When families are underrepresented, the curator generates targeted refill
candidates by asking the LLM to write reward code for specific strategies.
"""

import json
import math
import textwrap
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import openai


class RewardFamilyCurator:
    """Classify reward candidates into families and select a diverse subset."""

    def __init__(
        self,
        api_key: str,
        model: str,
        target_k: int = 16,
        max_per_family: int = 3,
        enable_refill: bool = True,
        max_refill_attempts: int = 2,
    ):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.target_k = target_k
        self.max_per_family = max_per_family
        self.enable_refill = enable_refill
        self.max_refill_attempts = max_refill_attempts
        # Stores debug info from the last curate() call
        self.last_debug_info: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def curate(
        self,
        candidates: List[Dict],
        elite_ids: Optional[Set[int]] = None,
        training_summary: Optional[Dict] = None,
        compile_fn: Optional[Callable[[str], bool]] = None,
    ) -> List[Dict]:
        """Top-level curation pipeline.

        1. Classify candidates into reward families via LLM.
        2. Remove near-duplicates (identified by LLM).
        3. Allocate slots per family (balanced, elite always kept).
        4. If underrepresented families exist and enable_refill=True,
           generate targeted refill candidates via LLM.
        5. Return curated list with ``reward_family`` annotations.

        Args:
            candidates: List of candidate dicts with "id", "code", "rationale".
            elite_ids: Set of candidate IDs that must never be removed.
            training_summary: Context dict passed to LLM for refill generation
                (task description, history, etc.).
            compile_fn: Optional function to test-compile reward code.
                Signature: compile_fn(code: str) -> bool.
        """
        if elite_ids is None:
            elite_ids = set()

        # Step 1: classify
        classification = self.classify_into_families(candidates)

        families: Dict[str, Dict] = classification.get("families", {})
        duplicate_pairs: List[List[int]] = classification.get("duplicate_pairs", [])

        # Build id -> family label mapping
        family_map: Dict[int, str] = {}
        for label, info in families.items():
            for member_id in info.get("members", []):
                family_map[int(member_id)] = label

        # Step 2 + 3: select diverse subset with balanced allocation
        curated = self.select_diverse_subset(
            candidates, family_map, duplicate_pairs, elite_ids
        )

        # Step 4: targeted refill for underrepresented families
        refill_info = []
        if self.enable_refill and len(curated) < self.target_k and training_summary:
            curated, refill_info = self._targeted_refill(
                curated, families, family_map, elite_ids,
                training_summary, compile_fn,
            )

        # Annotate each candidate with its family label
        for cand in curated:
            cand_id = cand.get("id")
            if cand_id is not None and cand_id in family_map:
                cand["reward_family"] = family_map[cand_id]

        # Store debug info
        self.last_debug_info = {
            "input_count": len(candidates),
            "output_count": len(curated),
            "families": {
                label: {
                    "description": info.get("description", ""),
                    "member_count": len(info.get("members", [])),
                    "members": info.get("members", []),
                }
                for label, info in families.items()
            },
            "duplicate_pairs": duplicate_pairs,
            "elite_ids": sorted(elite_ids),
            "kept_ids": sorted(c["id"] for c in curated if "id" in c),
            "refill_info": refill_info,
        }

        return curated

    # ------------------------------------------------------------------
    # LLM classification
    # ------------------------------------------------------------------

    def classify_into_families(self, candidates: List[Dict]) -> Dict:
        """Single LLM API call to classify candidates into reward families.

        Returns::

            {
                "families": {
                    "label": {
                        "description": "...",
                        "members": [id1, id2, ...]
                    },
                    ...
                },
                "duplicate_pairs": [[id_a, id_b], ...]
            }
        """
        # Build a compact summary of each candidate for the prompt
        candidate_summaries = []
        for cand in candidates:
            cand_id = cand.get("id", cand.get("candidate_id", "?"))
            code = cand.get("code", "")
            rationale = cand.get("rationale", "")
            snippet = code[:800] if code else "(no code)"
            candidate_summaries.append(
                f"--- Candidate {cand_id} ---\n"
                f"Rationale: {rationale}\n"
                f"Code:\n{snippet}\n"
            )

        prompt = textwrap.dedent("""\
            You are analyzing reward function candidates for reinforcement learning.
            Your job is to group them into "reward families" based on their structural
            strategy (e.g., what signals they use, how they shape the reward, what
            sub-goals they define).

            For each family, provide:
            - A short descriptive label (e.g., "distance_plus_alignment", "curriculum_staged")
            - A one-sentence description of the shared strategy
            - The list of candidate IDs that belong to this family

            Also identify near-duplicate pairs: candidates whose reward logic is
            structurally identical or differs only in minor constant values.

            Respond with ONLY valid JSON in this exact format:
            {
                "families": {
                    "family_label": {
                        "description": "One sentence description",
                        "members": [0, 3, 7]
                    }
                },
                "duplicate_pairs": [[0, 3], [5, 12]]
            }

            CANDIDATES:
        """)
        prompt += "\n".join(candidate_summaries)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=4096,
        )

        raw_text = response.choices[0].message.content.strip()

        # Parse JSON from response (handle markdown code fences)
        json_text = raw_text
        if "```" in json_text:
            # Extract content between code fences
            parts = json_text.split("```")
            for part in parts[1:]:
                # Skip the language tag line if present
                lines = part.strip().split("\n")
                if lines[0].strip().lower() in ("json", ""):
                    lines = lines[1:]
                candidate_json = "\n".join(lines).strip()
                if candidate_json:
                    json_text = candidate_json
                    break

        try:
            result = json.loads(json_text)
        except json.JSONDecodeError:
            print(f"[Curator] WARNING: Failed to parse LLM classification JSON. Raw response:\n{raw_text[:500]}")
            # Fallback: put all candidates in one family
            all_ids = [c.get("id", c.get("candidate_id", i)) for i, c in enumerate(candidates)]
            result = {
                "families": {
                    "unclassified": {
                        "description": "All candidates (classification failed)",
                        "members": all_ids,
                    }
                },
                "duplicate_pairs": [],
            }

        return result

    # ------------------------------------------------------------------
    # Diverse subset selection
    # ------------------------------------------------------------------

    def select_diverse_subset(
        self,
        candidates: List[Dict],
        family_map: Dict[int, str],
        duplicate_pairs: List[List[int]],
        elite_ids: Set[int],
    ) -> List[Dict]:
        """Core filtering: remove duplicates, enforce per-family cap, round-robin trim.

        Elite candidates are never removed.
        """
        # Index candidates by id
        cand_by_id: Dict[int, Dict] = {}
        for cand in candidates:
            cid = cand.get("id")
            if cid is not None:
                cand_by_id[cid] = cand

        # Step 1: Mark near-duplicates for removal (keep the first in each pair,
        # unless the second is elite).
        removed_ids: Set[int] = set()
        for pair in duplicate_pairs:
            if len(pair) < 2:
                continue
            id_a, id_b = int(pair[0]), int(pair[1])
            # Never remove elites
            if id_b not in elite_ids and id_b not in removed_ids:
                removed_ids.add(id_b)
            elif id_a not in elite_ids and id_a not in removed_ids:
                removed_ids.add(id_a)

        # Step 2: Group remaining candidates by family
        family_groups: Dict[str, List[Dict]] = {}
        unassigned: List[Dict] = []
        for cand in candidates:
            cid = cand.get("id")
            if cid in removed_ids:
                continue
            label = family_map.get(cid)
            if label is not None:
                family_groups.setdefault(label, []).append(cand)
            else:
                unassigned.append(cand)

        # Step 3: Enforce max_per_family cap (elites are never removed)
        selected: List[Dict] = []
        overflow: List[Dict] = []  # candidates beyond the cap, available for refill

        for label, members in family_groups.items():
            # Separate elites from non-elites within the family
            elites = [c for c in members if c.get("id") in elite_ids]
            non_elites = [c for c in members if c.get("id") not in elite_ids]

            # Always keep all elites
            selected.extend(elites)

            # Fill remaining slots with non-elites up to max_per_family
            remaining_slots = max(0, self.max_per_family - len(elites))
            selected.extend(non_elites[:remaining_slots])
            overflow.extend(non_elites[remaining_slots:])

        # Add unassigned candidates
        selected.extend(unassigned)

        # Step 4: Round-robin trim if still over target_k
        if len(selected) > self.target_k:
            # Build family -> non-elite members in selected
            family_selected: Dict[str, List[Dict]] = {}
            always_keep: List[Dict] = []
            for cand in selected:
                cid = cand.get("id")
                label = family_map.get(cid)
                if cid in elite_ids:
                    always_keep.append(cand)
                elif label is not None:
                    family_selected.setdefault(label, []).append(cand)
                else:
                    always_keep.append(cand)

            # Sort families by size (largest first) and trim from the largest
            trimmed_count = len(selected) - self.target_k
            sorted_families = sorted(
                family_selected.keys(),
                key=lambda f: len(family_selected[f]),
                reverse=True,
            )

            removed_in_trim: Set[int] = set()
            while trimmed_count > 0 and sorted_families:
                for fam in sorted_families:
                    if trimmed_count <= 0:
                        break
                    members = family_selected[fam]
                    if len(members) > 1:
                        removed_cand = members.pop()
                        removed_in_trim.add(removed_cand.get("id"))
                        trimmed_count -= 1
                # If no family has >1 member, stop
                if all(len(family_selected[f]) <= 1 for f in sorted_families):
                    break

            selected = always_keep + [
                c for fam in sorted_families for c in family_selected[fam]
            ]

        # Step 5: Refill from overflow if under target_k
        if len(selected) < self.target_k:
            needed = self.target_k - len(selected)
            selected.extend(overflow[:needed])

        return selected

    # ------------------------------------------------------------------
    # Targeted refill
    # ------------------------------------------------------------------

    def _targeted_refill(
        self,
        curated: List[Dict],
        families: Dict[str, Dict],
        family_map: Dict[int, str],
        elite_ids: Set[int],
        training_summary: Dict,
        compile_fn: Optional[Callable[[str], bool]],
    ) -> Tuple[List[Dict], List[Dict]]:
        """Generate new candidates for underrepresented families.

        Returns (updated_curated, refill_info_list).
        """
        refill_info = []
        n_elite = sum(1 for c in curated if c.get("id") in elite_ids)
        non_elite_slots = self.target_k - n_elite
        n_families = max(len(families), 1)
        slots_per_family = max(1, non_elite_slots // n_families)

        # Count current allocation per family
        family_counts: Dict[str, int] = {}
        for cand in curated:
            cid = cand.get("id")
            if cid in elite_ids:
                continue
            label = family_map.get(cid, "unknown")
            family_counts[label] = family_counts.get(label, 0) + 1

        # Identify families that need more candidates
        deficit: Dict[str, int] = {}
        for label in families:
            current = family_counts.get(label, 0)
            needed = slots_per_family - current
            if needed > 0:
                deficit[label] = needed

        if not deficit:
            return curated, refill_info

        total_needed = min(
            sum(deficit.values()),
            self.target_k - len(curated),
        )
        if total_needed <= 0:
            return curated, refill_info

        print(f"  [Curator Refill] Need {total_needed} candidates for families: "
              f"{', '.join(f'{k}({v})' for k, v in deficit.items())}")

        # Build refill requests
        next_id = max((c.get("id", -1) for c in curated), default=-1) + 1
        for attempt in range(self.max_refill_attempts):
            if total_needed <= 0:
                break

            refill_candidates = self._generate_refill(
                deficit, families, training_summary, total_needed,
            )

            for rc in refill_candidates:
                code = rc.get("code", "")
                if not code:
                    continue
                # Compile check
                if compile_fn and not compile_fn(code):
                    continue

                rc["id"] = next_id
                next_id += 1
                family_label = rc.get("target_family", "refill")
                rc["reward_family"] = family_label
                rc["is_refill"] = True
                family_map[rc["id"]] = family_label
                curated.append(rc)

                refill_info.append({
                    "id": rc["id"],
                    "family": family_label,
                    "attempt": attempt,
                })

                # Update deficit
                if family_label in deficit:
                    deficit[family_label] -= 1
                    if deficit[family_label] <= 0:
                        del deficit[family_label]
                total_needed -= 1
                if total_needed <= 0:
                    break

            if not deficit:
                break

        if total_needed > 0:
            print(f"  [Curator Refill] Warning: still {total_needed} slots unfilled after {self.max_refill_attempts} attempts")

        return curated, refill_info

    def _generate_refill(
        self,
        deficit: Dict[str, int],
        families: Dict[str, Dict],
        training_summary: Dict,
        total_needed: int,
    ) -> List[Dict]:
        """Ask LLM to generate candidates for specific reward families."""

        task_desc = training_summary.get("llm_task_description", "")
        state_docs = training_summary.get("state_access_docs", "")
        reward_src = training_summary.get("reward_fn_source", "")

        family_requests = []
        for label, count in deficit.items():
            desc = families.get(label, {}).get("description", label)
            family_requests.append(f"- {count} candidate(s) using strategy: '{label}' ({desc})")

        prompt = textwrap.dedent(f"""\
            You are writing reward functions for a reinforcement learning task.

            Task: {task_desc}

            {f'State Access Documentation:{chr(10)}{state_docs}' if state_docs else ''}

            Generate reward function candidates with SPECIFIC strategies as requested below.
            Each candidate must use a DIFFERENT approach from the requested family.

            Requested candidates:
            {chr(10).join(family_requests)}

            Total candidates needed: {total_needed}

            For each candidate, write a complete `compute_reward(info, base)` function.
            Store per-component values in `info["_reward_components"]`.

            Respond with ONLY valid JSON:
            {{
                "candidates": [
                    {{
                        "target_family": "family_label",
                        "rationale": "Brief explanation of this candidate's strategy",
                        "code": "import torch\\ndef compute_reward(info, base):\\n    ...\\n    return reward"
                    }}
                ]
            }}
        """)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_completion_tokens=8192,
            )
            raw_text = response.choices[0].message.content.strip()

            json_text = raw_text
            if "```" in json_text:
                parts = json_text.split("```")
                for part in parts[1:]:
                    lines = part.strip().split("\n")
                    if lines[0].strip().lower() in ("json", ""):
                        lines = lines[1:]
                    candidate_json = "\n".join(lines).strip()
                    if candidate_json:
                        json_text = candidate_json
                        break

            result = json.loads(json_text)
            candidates = result.get("candidates", [])
            print(f"  [Curator Refill] LLM generated {len(candidates)} refill candidates")
            return candidates

        except Exception as e:
            print(f"  [Curator Refill] Error generating refill: {e}")
            return []

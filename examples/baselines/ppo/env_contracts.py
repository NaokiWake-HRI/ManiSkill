"""
Environment contracts: single source of truth for env_id -> control_mode + action shape.

Call validate_env_setup() after env creation + wrapper application.
Raises ValueError with actionable message on any mismatch.
"""

import gymnasium as gym

# ──────────────────────────────────────────────────────────────
# Contract table
#   key   : env_id (exact match first, then prefix match)
#   value : {
#     control_mode      : required control mode
#     raw_action_dim    : action dim BEFORE wrappers (from gym.make)
#     wrapped_action_dim: action dim AFTER wrappers  (what PPO sees)
#   }
# ──────────────────────────────────────────────────────────────
ENV_CONTRACTS = {
    "PickCubePandaAllegro-v2": {
        "control_mode": "pd_ee_delta_pose",
        "raw_action_dim": 22,       # CombinedController: arm(6) + hand(16)
        "wrapped_action_dim": 8,    # CoupledAllegroActionWrapper: arm(6) + scalars(2)
    },
    # NOTE: v1 は RL 訓練パイプライン未対応（CoupledAllegroActionWrapper が
    # "PandaAllegro" substring match で適用されるため、v1 を ppo.py で走らせると
    # pd_ee_delta_pose + 8D になる）。v1 用の契約は実態が確定してから追加する。
}


def _lookup(env_id: str) -> dict | None:
    """Exact match on env_id."""
    return ENV_CONTRACTS.get(env_id)


def validate_env_setup(env_id: str, control_mode: str, env) -> None:
    """Validate env configuration against contract. No-op for unknown env_ids.

    Call AFTER gym.make + wrapper application (before ManiSkillVectorEnv).

    Args:
        env_id: environment ID string
        control_mode: the control_mode that was passed to gym.make
        env: the gymnasium env (possibly wrapped)

    Raises:
        ValueError: on any contract violation, with fix instructions.
    """
    contract = _lookup(env_id)
    if contract is None:
        return  # no contract -> no validation

    errors = []

    # 1) control_mode check
    expected_cm = contract["control_mode"]
    if control_mode != expected_cm:
        errors.append(
            f"  control_mode: got '{control_mode}', expected '{expected_cm}'\n"
            f"  Fix: pass --control_mode={expected_cm}"
        )

    # 2) action space shape check (after wrappers)
    actual_dim = env.action_space.shape[-1]
    expected_dim = contract["wrapped_action_dim"]
    if actual_dim != expected_dim:
        errors.append(
            f"  action_space: got dim={actual_dim}, expected dim={expected_dim}\n"
            f"  Fix: ensure the correct wrapper is applied for {env_id}"
        )

    # 3) action space must be Box (not Dict)
    # Check the innermost space type (handle batched spaces)
    space = env.action_space
    if not isinstance(space, gym.spaces.Box):
        errors.append(
            f"  action_space type: got {type(space).__name__}, expected Box\n"
            f"  Fix: apply FlattenActionSpaceWrapper or CoupledAllegroActionWrapper"
        )

    if errors:
        raise ValueError(
            f"\n{'='*60}\n"
            f"ENV CONTRACT VIOLATION: {env_id}\n"
            f"{'='*60}\n"
            + "\n".join(errors) +
            f"\n{'='*60}"
        )

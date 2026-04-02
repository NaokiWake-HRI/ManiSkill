"""Shared utilities for PPO outer-loop scripts.

Contains the neural-network architecture, logger, video/VLM helpers, and
random weight generation that are common to both ``ppo_outer_loop.py`` and
``ppo_outer_loop_full.py``.
"""

import base64
import io
import random as py_random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from reward_wrapper import TASK_DEFAULTS, _resolve_task_id


# ============================================================================
# Network
# ============================================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# ============================================================================
# Logger
# ============================================================================

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            import wandb
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.close()


# ============================================================================
# Video / VLM utilities
# ============================================================================

def crop_tiled_frame(frame: np.ndarray, num_total_envs: int, num_show_envs: int) -> np.ndarray:
    """Crop a tiled frame to show only the first num_show_envs environments."""
    if num_show_envs >= num_total_envs:
        return frame
    h, w = frame.shape[:2]
    nrows = int(np.sqrt(num_total_envs))
    ncols = int(np.ceil(num_total_envs / nrows))
    env_h = h // nrows
    env_w = w // ncols
    show_rows = int(np.ceil(np.sqrt(num_show_envs)))
    show_cols = int(np.ceil(num_show_envs / show_rows))
    return frame[:env_h * show_rows, :env_w * show_cols]


def extract_frames_from_video(
    video_path: Path,
    max_frames: int = 8,
    num_total_envs: int = 1,
    num_show_envs: int = 1,
) -> List[np.ndarray]:
    """Extract evenly-sampled frames from MP4 video."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    if total_frames <= max_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if num_show_envs < num_total_envs:
                frame = crop_tiled_frame(frame, num_total_envs, num_show_envs)
            frames.append(frame)
    cap.release()
    return frames


def extract_single_env_tile(frame: np.ndarray, env_idx: int, num_total_envs: int) -> np.ndarray:
    """Extract a single environment's tile from a tiled frame.

    Args:
        frame: Tiled frame (H, W, 3) containing all envs in a grid.
        env_idx: Index of the environment to extract (0-based).
        num_total_envs: Total number of environments in the tile grid.

    Returns:
        Cropped frame showing only the specified environment.
    """
    h, w = frame.shape[:2]
    nrows = int(np.sqrt(num_total_envs))
    ncols = int(np.ceil(num_total_envs / nrows))
    env_h = h // nrows
    env_w = w // ncols
    # ManiSkill tile_images fills column-major: top-to-bottom, then left-to-right
    col = env_idx // nrows
    row = env_idx % nrows
    return frame[row * env_h : (row + 1) * env_h, col * env_w : (col + 1) * env_w].copy()


def categorize_env_outcomes(
    env_last_outcomes: Dict[int, Dict[str, bool]],
) -> Dict[str, List[int]]:
    """Categorize env indices into success / near_miss / failure.

    Args:
        env_last_outcomes: {env_idx: {"success_once": bool, "success_at_end": bool}}

    Returns:
        {"success": [env_indices], "near_miss": [env_indices], "failure": [env_indices]}
    """
    categories: Dict[str, List[int]] = {"success": [], "near_miss": [], "failure": []}
    for env_idx, outcome in sorted(env_last_outcomes.items(), key=lambda x: int(x[0])):
        env_idx = int(env_idx)  # JSON round-trip converts int keys to str
        if outcome.get("success_at_end", False):
            categories["success"].append(env_idx)
        elif outcome.get("success_once", False):
            categories["near_miss"].append(env_idx)
        else:
            categories["failure"].append(env_idx)
    return categories


def resolve_vlm_categories_to_show(
    env_categories: Dict[str, List[int]],
    focus: str = "all",
) -> List[str]:
    """Resolve which categorized outcome panels should be shown to VLM."""
    focus = focus.lower()
    if focus == "all":
        requested = ["failure", "near_miss", "success"]
    elif focus == "failure_and_near_miss":
        requested = ["failure", "near_miss"]
    elif focus in {"failure", "near_miss", "success"}:
        requested = [focus]
    else:
        raise ValueError(
            f"Unsupported vlm_category_focus={focus!r}. "
            "Expected one of: all, failure, near_miss, failure_and_near_miss, success."
        )

    return [cat for cat in requested if env_categories.get(cat)]


def extract_categorized_frames(
    video_path: Path,
    env_categories: Dict[str, List[int]],
    num_total_envs: int,
    max_frames: int = 8,
    categories_to_show: Optional[List[str]] = None,
) -> tuple:
    """Create composite frames showing one representative env per category.

    For each timestep, extracts tiles for one env per non-empty category,
    adds a text label, and concatenates them horizontally.

    Args:
        video_path: Path to the tiled MP4 video.
        env_categories: {"success": [env_indices], "near_miss": [...], "failure": [...]}
        num_total_envs: Total envs in the tile grid.
        max_frames: Maximum number of timestep frames to extract.
        categories_to_show: Ordered subset of categories to include. If None,
            show all non-empty categories in failure > near_miss > success order.

    Returns:
        (composite_frames, categories_shown, selected_envs)
        - composite_frames: list of composite RGB frames
        - categories_shown: list of category names included (e.g. ["failure", "near_miss"])
        - selected_envs: dict {category: env_idx} for the representative envs
    """
    # Pick one representative env per non-empty category.
    selected: Dict[str, int] = {}
    ordered_categories = categories_to_show or ["failure", "near_miss", "success"]
    for cat in ordered_categories:
        indices = env_categories.get(cat, [])
        if indices:
            selected[cat] = indices[0]

    if not selected:
        return [], [], {}

    # Label styling
    _LABEL_MAP = {
        "success": "SUCCESS",
        "near_miss": "NEAR_MISS (success_once but lost)",
        "failure": "FAILURE",
    }
    _COLOR_MAP = {
        "success": (0, 180, 0),      # green
        "near_miss": (220, 180, 0),   # yellow
        "failure": (220, 0, 0),       # red
    }

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return [], [], {}

    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()

    composite_frames = []
    for fidx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        tiles = []
        for cat, env_idx in selected.items():
            tile = extract_single_env_tile(frame, env_idx, num_total_envs)
            tiles.append(tile)

        # Concatenate tiles horizontally
        if tiles:
            # Pad to same height if needed
            max_h = max(t.shape[0] for t in tiles)
            padded = []
            for t in tiles:
                if t.shape[0] < max_h:
                    pad = np.zeros((max_h - t.shape[0], t.shape[1], 3), dtype=np.uint8)
                    t = np.concatenate([t, pad], axis=0)
                padded.append(t)
            composite_frames.append(np.concatenate(padded, axis=1))

    cap.release()
    categories_shown = list(selected.keys())
    return composite_frames, categories_shown, selected


def build_vlm_prompt(env_id: str) -> str:
    """Build a VLM prompt focused on failure analysis."""
    return f"""Analyze this robot manipulation video for the task: {env_id}.

Focus on FAILURE ANALYSIS:
1. What is the robot currently doing? Describe the behavior you see.
2. What is going WRONG? Be specific about failure modes:
   - Is the robot failing to reach the target?
   - Is it reaching but failing to grasp/push/open?
   - Is it succeeding partially but then losing the object?
   - Is it moving too fast, too slow, or in the wrong direction?
3. What reward signal adjustments might help fix the observed failures?

Be concise and specific. Focus on actionable observations.
Do NOT provide a numerical score - focus on qualitative analysis.

After your English analysis, provide a brief summary in Japanese (日本語での簡潔な要約も追加してください)."""


def build_vlm_prompt_categorized(env_id: str, categories_shown: List[str]) -> str:
    """Build a VLM prompt for comparing success / near_miss / failure episodes.

    Args:
        env_id: Environment ID string.
        categories_shown: List of categories visible in the frames
            (e.g. ["failure", "near_miss", "success"]).
    """
    if not categories_shown:
        raise ValueError("categories_shown must not be empty")

    if categories_shown == ["failure"]:
        return build_vlm_prompt(env_id)

    if categories_shown == ["near_miss"]:
        return f"""Analyze this robot manipulation video for the task: {env_id}.

This episode is labeled NEAR_MISS.
- NEAR_MISS: Episode where the robot achieved success_once but LOST it before the end.

Focus on INSTABILITY ANALYSIS:
1. What is the robot doing before, during, and after first reaching success?
2. What causes it to LOSE the success it once achieved?
3. What reward signal adjustments might help it achieve and maintain success?

Be concise and specific. Focus on actionable observations.
Do NOT provide a numerical score - focus on qualitative analysis.

After your English analysis, provide a brief summary in Japanese (日本語での簡潔な要約も追加してください)."""

    if categories_shown == ["success"]:
        return f"""Analyze this robot manipulation video for the task: {env_id}.

This episode is labeled SUCCESS.
- SUCCESS: Episode where the robot completed the task and held success at the end.

Focus on SUCCESS ANALYSIS:
1. What is the robot doing that makes this episode successful?
2. Which approach direction, contact pattern, or stopping behavior seems important?
3. What reward signal adjustments would preserve this behavior while avoiding regressions?

Be concise and specific. Focus on actionable observations.
Do NOT provide a numerical score - focus on qualitative analysis.

After your English analysis, provide a brief summary in Japanese (日本語での簡潔な要約も追加してください)."""

    panel_desc = " | ".join(cat.upper() for cat in categories_shown)
    cat_bullets = "\n".join(
        f"   - {cat.upper()}: "
        + {
            "success": "Episode where the robot completed the task and held success at the end.",
            "near_miss": "Episode where the robot achieved success_once but LOST it before the end.",
            "failure": "Episode where the robot never achieved success at all.",
        }[cat]
        for cat in categories_shown
    )

    return f"""Analyze these side-by-side robot manipulation episodes for the task: {env_id}.

Each frame shows panels left-to-right: {panel_desc}
{cat_bullets}

COMPARATIVE ANALYSIS:
1. Describe what each panel's robot is doing. How do their behaviors differ?
2. For FAILURE / NEAR_MISS panels: What specific behavior prevents success?
   - Reaching errors, grasping failures, object slippage, wrong direction, etc.
3. For NEAR_MISS (if present): What causes the robot to LOSE the success it once achieved?
   - This is especially valuable — it reveals instability in the current policy.
4. What reward signal adjustments could fix the observed failure modes?
   - Be specific: which reward component should increase/decrease and why.

Be concise and specific. Focus on actionable observations.
Do NOT provide a numerical score — focus on qualitative analysis.

After your English analysis, provide a brief summary in Japanese (日本語での簡潔な要約も追加してください)."""


def generate_reward_plot_html(
    step_rewards: torch.Tensor,
    num_envs: int,
    breakdowns: Optional[List[Dict[str, float]]] = None,
) -> str:
    """Generate an HTML snippet with per-step reward plot from an eval rollout.

    Args:
        step_rewards: tensor of shape (num_eval_steps, num_envs)
        num_envs: number of eval environments
        breakdowns: list of dicts (one per timestep) with component means
    Returns:
        HTML string with base64-embedded PNG plot
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rewards_np = step_rewards.cpu().numpy()  # (T, E)
    T = rewards_np.shape[0]
    timesteps = np.arange(T)

    has_breakdowns = breakdowns and len(breakdowns) == T
    fig, axes = plt.subplots(2 if has_breakdowns else 1, 1,
                             figsize=(10, 7 if has_breakdowns else 4),
                             sharex=True)
    if not has_breakdowns:
        axes = [axes]

    # Top: total reward
    ax = axes[0]
    for e in range(min(num_envs, 8)):
        ax.plot(timesteps, rewards_np[:, e], alpha=0.3, linewidth=0.8)
    mean_r = rewards_np.mean(axis=1)
    ax.plot(timesteps, mean_r, "k-", linewidth=2, label=f"mean (n={num_envs})")
    ax.set_ylabel("Total Reward")
    ax.set_title("Per-Step Reward During Eval Rollout")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Bottom: component breakdown
    if has_breakdowns:
        ax2 = axes[1]
        component_keys = [k for k in breakdowns[0] if k != "norm_scale"]
        for key in component_keys:
            values = [bd.get(key, 0.0) for bd in breakdowns]
            ax2.plot(timesteps, values, linewidth=1.5, label=key)
        ax2.set_xlabel("Episode Timestep")
        ax2.set_ylabel("Component (mean across envs)")
        ax2.set_title("Reward Component Breakdown")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)
    else:
        axes[0].set_xlabel("Episode Timestep")

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return (
        '<div style="margin-top:20px; border-top:1px solid #ccc; padding-top:10px;">'
        '<h3>Per-Step Reward (Eval Rollout)</h3>'
        f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%;">'
        '</div>'
    )


def append_html_to_file(file_path: Path, html_snippet: str):
    """Append an HTML snippet before the closing </body> tag."""
    content = file_path.read_text()
    if "</body>" in content:
        content = content.replace("</body>", html_snippet + "\n</body>")
    else:
        content += html_snippet
    file_path.write_text(content)


# ============================================================================
# Random weight generation
# ============================================================================

def generate_random_weights(
    env_id: str,
    seed: int = 42,
    w_min: float = 0.01,
    w_max: float = 10.0,
    ws_min: float = 0.1,
    ws_max: float = 20.0,
) -> Dict[str, float]:
    """Generate random reward weights for a task."""
    task_id = _resolve_task_id(env_id)
    rng = py_random.Random(seed)
    defaults = TASK_DEFAULTS[task_id]
    weights = {}
    for k in defaults:
        if k == "w_success":
            weights[k] = rng.uniform(ws_min, ws_max)
        else:
            weights[k] = rng.uniform(w_min, w_max)
    return weights

"""
SAC with Iterative VLM/LLM Reward Tuning for ManiSkill.

Based on sac.py. Splits training into segments and uses VLM for failure
analysis + LLM for reward weight adjustment between segments.

Usage:
    python sac_iterative.py --env_id PickCube-v1 --total_timesteps 4_000_000 --num_segments 10
    python sac_iterative.py --env_id PickCube-v1 --total_timesteps 4_000_000 --num_segments 10 --skip_vlm_llm
"""

from collections import defaultdict
from dataclasses import dataclass
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import tqdm

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro

import mani_skill.envs

# Import debug RewardWrapper (local, with component storage support)
from reward_wrapper_debug import RewardWrapper, _resolve_task_id


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "SAC-Iterative"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances"""
    save_trajectory: bool = False
    """whether to save trajectory data"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file"""
    log_freq: int = 1_000
    """logging frequency in terms of environment steps"""

    # Environment specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    eval_freq: int = 100
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    buffer_device: str = "cuda"
    """where the replay buffer is stored"""
    gamma: float = 0.8
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""
    learning_starts: int = 4_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    training_freq: int = 64
    """training frequency (in steps)"""
    utd: float = 0.5
    """update to data ratio"""
    bootstrap_at_done: str = "always"
    """the bootstrap method to use when done"""

    # Iterative VLM/LLM arguments
    num_segments: int = 10
    """number of training segments for VLM/LLM updates"""
    vlm_model: str = "gpt-5.2"
    """VLM model for video analysis"""
    llm_model: str = "gpt-5.2"
    """LLM model for reward tuning"""
    vlm_max_frames: int = 4
    """max frames to send to VLM"""
    vlm_num_envs: int = 2
    """number of eval envs to show in VLM frames"""
    rl_project_path: str = "/home/nwake/codes/RL_project"
    """path to RL_project for VLM/LLM imports"""
    skip_vlm_llm: bool = False
    """skip VLM/LLM calls (for testing reward wrapper only)"""
    vlm_reward_plot: bool = False
    """include reward time-series plot in VLM debug HTML"""
    clear_buffer_at_segment: bool = False
    """clear replay buffer at every segment boundary (even without VLM/LLM weight changes)"""
    critic_warmup_steps: int = 5000
    """number of critic-only updates after reward recompute (actor frozen)"""
    initial_weights_file: Optional[str] = None
    """path to JSON file with initial reward weights (overrides TASK_DEFAULTS)"""

    # to be filled in runtime
    grad_steps_per_iteration: int = 0
    steps_per_env: int = 0


@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, env, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device,
                 component_keys: Optional[List[str]] = None):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs
        self.obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
        self.next_obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
        self.actions = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_action_space.shape).to(storage_device)
        self.logprobs = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.rewards = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.values = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)

        # Component storage for reward recomputation
        self.component_keys = component_keys or []
        self.components: Dict[str, torch.Tensor] = {}
        for key in self.component_keys:
            self.components[key] = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.success = torch.zeros((self.per_env_buffer_size, self.num_envs), dtype=torch.bool).to(storage_device)

    def add(self, obs, next_obs, action, reward, done,
            components: Optional[Dict[str, torch.Tensor]] = None,
            success: Optional[torch.Tensor] = None):
        if self.storage_device == torch.device("cpu"):
            obs = obs.cpu()
            next_obs = next_obs.cpu()
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        if components is not None:
            for key in self.component_keys:
                val = components[key]
                if self.storage_device == torch.device("cpu"):
                    val = val.cpu()
                self.components[key][self.pos] = val
        if success is not None:
            s = success
            if self.storage_device == torch.device("cpu"):
                s = s.cpu()
            self.success[self.pos] = s
        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    def clear(self):
        """Clear the replay buffer (reset position, keep allocated memory)."""
        self.pos = 0
        self.full = False

    def recompute_rewards(self, reward_wrapper, normalize_stats: bool = True):
        """Recompute all stored rewards using current weights and stored components.

        Args:
            reward_wrapper: RewardWrapper instance with updated weights
            normalize_stats: If True, normalize new rewards to match old mean/std (Q stability)

        Returns:
            dict with statistics: old_mean, old_std, new_mean, new_std, etc.
        """
        n = self.per_env_buffer_size if self.full else self.pos
        if n == 0:
            return {}

        # Compute old statistics (exclude success rewards for stable statistics)
        success_mask = self.success[:n].flatten()
        old_rewards_flat = self.rewards[:n].flatten()

        if success_mask.any():
            old_non_success = old_rewards_flat[~success_mask]
            old_mean = old_non_success.mean().item() if len(old_non_success) > 0 else 0.0
            old_std = old_non_success.std().item() if len(old_non_success) > 1 else 1.0
        else:
            old_mean = old_rewards_flat.mean().item()
            old_std = old_rewards_flat.std().item() if n > 1 else 1.0

        old_std = max(old_std, 1e-6)  # Avoid division by zero

        # Recompute with new weights
        for i in range(n):
            comp_slice = {k: self.components[k][i] for k in self.component_keys}
            self.rewards[i] = reward_wrapper.compute_reward_from_components(
                comp_slice, self.success[i],
            )

        # Compute new statistics
        new_rewards_flat = self.rewards[:n].flatten()
        if success_mask.any():
            new_non_success = new_rewards_flat[~success_mask]
            new_mean_raw = new_non_success.mean().item() if len(new_non_success) > 0 else 0.0
            new_std_raw = new_non_success.std().item() if len(new_non_success) > 1 else 1.0
        else:
            new_mean_raw = new_rewards_flat.mean().item()
            new_std_raw = new_rewards_flat.std().item() if n > 1 else 1.0

        new_std_raw = max(new_std_raw, 1e-6)

        # Apply statistical normalization to maintain Q value stability
        if normalize_stats:
            # Normalize non-success rewards to match old distribution
            if success_mask.any():
                # Normalize only non-success part
                new_non_success_normalized = (new_non_success - new_mean_raw) / new_std_raw * old_std + old_mean
                new_rewards_flat[~success_mask] = new_non_success_normalized
                self.rewards[:n] = new_rewards_flat.view(n, self.num_envs)
            else:
                # Normalize all
                self.rewards[:n] = (self.rewards[:n] - new_mean_raw) / new_std_raw * old_std + old_mean

        # Final statistics
        final_mean = self.rewards[:n].mean().item()
        final_std = self.rewards[:n].std().item()

        stats = {
            "old_mean": old_mean,
            "old_std": old_std,
            "new_mean_raw": new_mean_raw,
            "new_std_raw": new_std_raw,
            "final_mean": final_mean,
            "final_std": final_std,
            "normalized": normalize_stats,
        }

        return stats

    def visualize_rewards(self, save_path: str, title: str = "Replay Buffer Rewards"):
        """Visualize buffer rewards as heatmap (rows=timesteps, cols=envs)."""
        import matplotlib.pyplot as plt
        n = self.per_env_buffer_size if self.full else self.pos
        if n == 0:
            return
        # Extract rewards: (n, num_envs)
        reward_data = self.rewards[:n].cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(reward_data, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_xlabel('Environment Index')
        ax.set_ylabel('Buffer Timestep')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Reward')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[DEBUG] Saved buffer visualization to {save_path}")

    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size,))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size,))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size,))
        return ReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
        )


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.net(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_eval_action(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.close()


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
    """Extract evenly-sampled frames from MP4 video, optionally cropping to show fewer envs."""
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


def generate_reward_plot_html(
    step_rewards: torch.Tensor,
    num_envs: int,
    breakdowns: Optional[List[Dict[str, float]]] = None,
) -> str:
    """Generate an HTML snippet with per-step reward plot from an eval rollout.

    Args:
        step_rewards: tensor of shape (num_eval_steps, num_envs) with per-step rewards
        num_envs: number of eval environments
        breakdowns: list of dicts (one per timestep) with component means
    Returns:
        HTML string with base64-embedded PNG plot
    """
    import base64
    import io
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

    # --- Top: total reward ---
    ax = axes[0]
    for e in range(min(num_envs, 8)):
        ax.plot(timesteps, rewards_np[:, e], alpha=0.3, linewidth=0.8)
    mean_r = rewards_np.mean(axis=1)
    ax.plot(timesteps, mean_r, "k-", linewidth=2, label=f"mean (n={num_envs})")
    ax.set_ylabel("Total Reward")
    ax.set_title("Per-Step Reward During Eval Rollout")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Bottom: component breakdown ---
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
    """Append an HTML snippet before the closing </body> tag of an existing HTML file."""
    content = file_path.read_text()
    if "</body>" in content:
        content = content.replace("</body>", html_snippet + "\n</body>")
    else:
        content += html_snippet
    file_path.write_text(content)


def build_vlm_prompt(env_id: str) -> str:
    """Build a VLM prompt focused on failure analysis (not scoring)."""
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


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
    vlm_suffix = "novlm" if args.skip_vlm_llm else "vlm"
    if args.vlm_reward_plot:
        vlm_suffix += "-includereward"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.exp_name}-{vlm_suffix}-{timestamp}"

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Environment setup #######
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="gpu",
        reward_mode="none",  # Disable built-in reward
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1,
                    reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs,
                         reconfiguration_freq=args.eval_reconfiguration_freq,
                         human_render_camera_configs=dict(shader_pack="default"), **env_kwargs)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    # Add RewardWrapper (BEFORE RecordEpisode and ManiSkillVectorEnv)
    # Load initial weights from file if provided (for random baseline experiments)
    initial_weights = None
    if args.initial_weights_file is not None:
        with open(args.initial_weights_file, "r") as f:
            initial_weights = json.load(f)
        print(f"[info] Loaded initial weights from {args.initial_weights_file}: {initial_weights}")

    reward_wrapper_train = RewardWrapper(envs, env_id=args.env_id, weights=initial_weights)
    reward_wrapper_eval = RewardWrapper(eval_envs, env_id=args.env_id, weights=initial_weights)
    envs = reward_wrapper_train
    eval_envs = reward_wrapper_eval

    # Video recording
    eval_output_dir = f"runs/{run_name}/videos"
    if args.capture_video or args.save_trajectory:
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos",
                                 save_trajectory=False, save_video_trigger=save_video_trigger,
                                 max_steps_per_video=args.num_steps, video_fps=30)
        # Limit eval videos to ~10 across entire training
        _total_iters = args.total_timesteps // max(1, args.num_envs * args.steps_per_env)
        _total_evals = max(1, _total_iters // args.eval_freq)
        _eval_video_interval = max(1, _total_evals // 10)
        eval_video_trigger = lambda x: (x // args.num_eval_steps) % _eval_video_interval == 0
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir,
                                  save_trajectory=args.save_trajectory,
                                  save_video_trigger=eval_video_trigger,
                                  trajectory_name="trajectory",
                                  max_steps_per_video=args.num_eval_steps, video_fps=30)

    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

    # --- Logging ---
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id,
                                     env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["initial_reward_weights"] = reward_wrapper_train.get_weights()
            if args.skip_vlm_llm:
                wandb_tags = ["sac", "iterative", "reward-wrapper-only"]
                wandb_group = "SAC-Iterative-NoVLM"
            else:
                wandb_tags = ["sac", "iterative", "vlm-llm"]
                wandb_group = "SAC-Iterative"
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=wandb_group,
                tags=wandb_tags,
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    max_action = float(envs.single_action_space.high[0])

    # --- Networks ---
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        actor.load_state_dict(ckpt['actor'])
        qf1.load_state_dict(ckpt['qf1'])
        qf2.load_state_dict(ckpt['qf2'])
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device,
        component_keys=reward_wrapper_train.component_keys,
    )

    # --- VLM/LLM setup ---
    vlm = None
    llm = None
    save_vlm_debug_html = None
    save_llm_debug_html = None
    if not args.skip_vlm_llm and not args.evaluate:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            import types
            sb3_mocks = {
                "stable_baselines3": {},
                "stable_baselines3.common": {},
                "stable_baselines3.common.callbacks": {"BaseCallback": object, "CallbackList": object},
                "stable_baselines3.common.base_class": {"BaseAlgorithm": object},
                "stable_baselines3.common.vec_env": {"VecNormalize": object},
                "stable_baselines3.common.logger": {"configure": lambda *a, **kw: None},
            }
            for mod_name, attrs in sb3_mocks.items():
                if mod_name not in sys.modules:
                    mock = types.ModuleType(mod_name)
                    for attr_name, attr_val in attrs.items():
                        setattr(mock, attr_name, attr_val)
                    sys.modules[mod_name] = mock

            sys.path.insert(0, args.rl_project_path)
            from experiments.callbacks.episode_collector import VLMEvaluator, LLMRewardTuner
            from experiments.iterative_learner import save_vlm_debug_html, save_llm_debug_html

            vlm = VLMEvaluator.from_openai(
                api_key=api_key,
                model=args.vlm_model,
                prompt=build_vlm_prompt(args.env_id),
                max_frames=args.vlm_max_frames,
                cache_results=False,
            )
            llm = LLMRewardTuner.from_openai(
                api_key=api_key,
                model=args.llm_model,
                enable_function_code=False,
                max_param_change=2.0,
            )
            print(f"[VLM/LLM] Initialized with {args.vlm_model} / {args.llm_model}")
        else:
            print("[warn] OPENAI_API_KEY not set, skipping VLM/LLM")
    else:
        if args.skip_vlm_llm:
            print("[info] VLM/LLM disabled (--skip_vlm_llm)")
        elif args.evaluate:
            print("[info] VLM/LLM disabled (evaluation mode)")

    print(f"[info] VLM/LLM active: {vlm is not None and llm is not None}")

    # --- Training ---
    obs, info = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    global_step = 0
    global_update = 0
    learning_has_started = False

    global_steps_per_iteration = args.num_envs * args.steps_per_env

    # Segment-level configuration
    steps_per_segment = args.total_timesteps // args.num_segments
    segment_history = []
    initial_weights = reward_wrapper_train.get_weights()
    latest_eval_metrics: Dict[str, Any] = {}
    latest_eval_step_rewards: List[torch.Tensor] = []  # per-step rewards from last eval
    latest_eval_reward_breakdowns: List[Dict[str, float]] = []  # per-step breakdown from last eval

    print(f"####")
    print(f"total_timesteps={args.total_timesteps} steps_per_segment={steps_per_segment} num_segments={args.num_segments}")
    print(f"num_envs={args.num_envs} num_eval_envs={args.num_eval_envs}")
    print(f"steps_per_env={args.steps_per_env} grad_steps_per_iteration={args.grad_steps_per_iteration}")
    print(f"Initial reward weights: {initial_weights}")
    print(f"####")

    pbar = tqdm.tqdm(range(args.total_timesteps))
    cumulative_times = defaultdict(float)
    current_segment = 0
    next_segment_step = steps_per_segment
    iteration = 0
    eval_count = 0
    # Save ~10 checkpoints across entire training
    _ckpt_total_evals = max(1, (args.total_timesteps // max(1, args.num_envs * args.steps_per_env)) // args.eval_freq)
    _ckpt_save_interval = max(1, _ckpt_total_evals // 10)

    # Critic warmup state (after reward recompute)
    critic_warmup_remaining = 0

    def run_eval():
        """Run evaluation, update latest_eval_metrics, log to wandb/tensorboard."""
        actor.eval()
        stime = time.perf_counter()
        eval_obs_local, _ = eval_envs.reset()
        eval_metrics = defaultdict(list)
        num_episodes = 0
        step_rewards_list = []
        step_breakdowns_list = []
        for _ in range(args.num_eval_steps):
            with torch.no_grad():
                eval_obs_local, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(actor.get_eval_action(eval_obs_local))
                step_rewards_list.append(eval_rew.detach())
                step_breakdowns_list.append(dict(reward_wrapper_eval._last_breakdown))
                if "final_info" in eval_infos:
                    mask = eval_infos["_final_info"]
                    num_episodes += mask.sum()
                    for k, v in eval_infos["final_info"]["episode"].items():
                        eval_metrics[k].append(v)
        # Store per-step rewards: (num_eval_steps, num_eval_envs)
        latest_eval_step_rewards.clear()
        latest_eval_step_rewards.extend(step_rewards_list)
        latest_eval_reward_breakdowns.clear()
        latest_eval_reward_breakdowns.extend(step_breakdowns_list)
        eval_metrics_mean = {}
        for k, v in eval_metrics.items():
            mean = torch.stack(v).float().mean()
            eval_metrics_mean[k] = mean
            if logger is not None:
                logger.add_scalar(f"eval/{k}", mean, global_step)
        latest_eval_metrics.clear()
        latest_eval_metrics.update({k: v.item() for k, v in eval_metrics_mean.items()})
        # Track how many eval episodes actually finished (used by LLM summary).
        if isinstance(num_episodes, torch.Tensor):
            latest_eval_metrics["num_episodes"] = int(num_episodes.item())
        else:
            latest_eval_metrics["num_episodes"] = int(num_episodes)
        pbar.set_description(
            f"success_once: {eval_metrics_mean.get('success_once', 0):.2f}, "
            f"return: {eval_metrics_mean.get('return', 0):.2f}"
        )
        if logger is not None:
            eval_time = time.perf_counter() - stime
            cumulative_times["eval_time"] += eval_time
            logger.add_scalar("time/eval_time", eval_time, global_step)
        actor.train()

        global eval_count
        eval_count += 1
        if args.save_model and eval_count % _ckpt_save_interval == 0:
            model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
            torch.save({
                'actor': actor.state_dict(),
                'qf1': qf1_target.state_dict(),
                'qf2': qf2_target.state_dict(),
                'log_alpha': log_alpha,
            }, model_path)
            print(f"model saved to {model_path}")

    while global_step < args.total_timesteps:
        iteration += 1

        # --- Evaluation (every eval_freq iterations) ---
        if args.eval_freq > 0 and iteration % args.eval_freq == 1:
            run_eval()
            if args.evaluate:
                break

        # --- Segment boundary check (AFTER eval, so fresh video/metrics exist) ---
        if global_step >= next_segment_step and current_segment < args.num_segments:
            # Force eval if not just run above
            if iteration % args.eval_freq != 1:
                print(f"[segment] Forcing eval at segment boundary")
                run_eval()

            print(f"\n{'='*60}")
            print(f"Segment {current_segment+1}/{args.num_segments} complete (global_step={global_step})")
            print(f"{'='*60}")

            # Log current weights
            current_weights = reward_wrapper_train.get_weights()
            if logger is not None:
                for wk, wv in current_weights.items():
                    logger.add_scalar(f"weights/{wk}", wv, global_step)

            if vlm is not None and llm is not None:
                print(f"[VLM/LLM] Analyzing segment {current_segment+1}...")

                # Extract frames from latest eval video
                video_dir = Path(eval_output_dir)
                video_files = sorted(video_dir.glob("*.mp4"))
                vlm_comment = None

                if video_files:
                    latest_video = video_files[-1]
                    frames = extract_frames_from_video(
                        latest_video,
                        max_frames=args.vlm_max_frames,
                        num_total_envs=args.num_eval_envs,
                        num_show_envs=args.vlm_num_envs,
                    )
                    if frames:
                        episode_info = {
                            "return": latest_eval_metrics.get("return", 0.0),
                            "success": latest_eval_metrics.get("success", 0.0),
                            "length": args.num_eval_steps,
                        }

                        try:
                            _, vlm_comment, _ = vlm.evaluate(frames, episode_info)
                            print(f"[VLM] Analysis:\n{vlm_comment}")

                            # Save VLM debug HTML
                            debug_dir = Path(f"runs/{run_name}/debug_html")
                            vlm_html_path = debug_dir / f"segment_{current_segment+1:02d}_vlm.html"
                            save_vlm_debug_html(
                                frames=frames,
                                prompt=build_vlm_prompt(args.env_id),
                                episode_info=episode_info,
                                vlm_score=0.0,
                                vlm_comment=vlm_comment,
                                save_path=vlm_html_path,
                                max_frames=args.vlm_max_frames,
                            )
                            if args.vlm_reward_plot and latest_eval_step_rewards:
                                step_rewards_tensor = torch.stack(latest_eval_step_rewards)
                                plot_html = generate_reward_plot_html(
                                    step_rewards_tensor, args.num_eval_envs,
                                    breakdowns=latest_eval_reward_breakdowns,
                                )
                                append_html_to_file(vlm_html_path, plot_html)
                        except Exception as e:
                            print(f"[VLM] Error: {e}")
                            vlm_comment = f"VLM analysis failed: {e}"
                    else:
                        print("[warn] No frames extracted from video")
                else:
                    print("[warn] No eval videos found")

                # LLM reward tuning
                if vlm_comment is not None:
                    # Build task description for LLM (appears in prompt via llm_task_description)
                    _task_id = _resolve_task_id(args.env_id)
                    _llm_task_descs = {
                        "PushCube": (
                            "The robot arm must push a cube to the goal position (PushCube).\n"
                            "Reward components: w_reach (TCP approach to push position behind cube), "
                            "w_push (cube movement toward goal, gated by reach), "
                            "w_z_keep (keep cube on table surface), w_success (success bonus).\n\n"
                            "日本語補足: ロボットアームがキューブを目標位置まで押すタスク。"
                            "w_reachはTCPのキューブ背面への接近、w_pushはゴールへの押し動作（到達後に有効）、"
                            "w_z_keepはキューブをテーブルから落とさない制約、w_successは成功ボーナス。"
                            "回答の末尾に日本語での簡潔な要約も追加してください。"
                        ),
                        "PickCube": (
                            f"The robot must pick up a cube and place it at the goal ({args.env_id}).\n"
                            "Reward components: w_reach (TCP approach to cube), w_grasp (grasp success), "
                            "w_place (cube toward goal, gated by grasp), "
                            "w_static (robot static with object placed), w_success (success bonus).\n\n"
                            "日本語補足: ロボットがキューブを掴んで目標位置に運ぶタスク。"
                            "w_reachは接近、w_graspは把持成功、w_placeは把持状態でのゴールへの配置、"
                            "w_staticは静止状態での配置完了、w_successは成功ボーナス。"
                            "回答の末尾に日本語での簡潔な要約も追加してください。"
                        ),
                        "OpenCabinetDoor": (
                            "The robot must open a cabinet door (OpenCabinetDoor).\n"
                            "Reward components: w_reach (TCP to handle), w_open (door opening progress), "
                            "w_static (maintain open state), w_success (success bonus).\n\n"
                            "日本語補足: キャビネットのドアを開けるタスク。回答の末尾に日本語での簡潔な要約も追加してください。"
                        ),
                        "OpenCabinetDrawer": (
                            "The robot must open a cabinet drawer (OpenCabinetDrawer).\n"
                            "Reward components: w_reach (TCP to handle), w_open (drawer opening progress), "
                            "w_static (maintain open state), w_success (success bonus).\n\n"
                            "日本語補足: キャビネットの引き出しを開けるタスク。回答の末尾に日本語での簡潔な要約も追加してください。"
                        ),
                    }

                    training_summary = {
                        "current_weights": current_weights,
                        "initial_weights": initial_weights,
                        "current_iteration": current_segment,
                        "total_iterations": args.num_segments,
                        "total_timesteps": global_step,
                        "avg_return": latest_eval_metrics.get("return", 0.0),
                        "success_rate": latest_eval_metrics.get("success", 0.0),
                        "vlm_avg_score": 0.5,
                        "vlm_comments": [vlm_comment],
                        "num_episodes": int(latest_eval_metrics.get("num_episodes", 0)),
                        "llm_task_description": _llm_task_descs.get(_task_id, ""),
                    }

                    try:
                        suggestions = llm.suggest_parameters(training_summary)

                        # Save LLM debug HTML
                        query_info = llm.get_last_query_info() if hasattr(llm, 'get_last_query_info') else None
                        llm_prompt = query_info.get("prompt", "(no prompt)") if query_info else "(no query info)"
                        llm_response = query_info.get("response_text", "(no response)") if query_info else "(no query info)"
                        debug_dir = Path(f"runs/{run_name}/debug_html")
                        save_llm_debug_html(
                            iteration=current_segment,
                            prompt=llm_prompt,
                            response_text=llm_response,
                            suggestions=suggestions,
                            summary_for_llm=training_summary,
                            save_path=debug_dir / f"segment_{current_segment+1:02d}_llm.html",
                        )

                        if suggestions and suggestions.get("type") == "params":
                            new_params = suggestions.get("params", {})
                            rationale = suggestions.get("rationale", "No rationale")
                            if new_params:
                                print(f"[LLM] Rationale: {rationale}")
                                print(f"[LLM] Suggested weights: {new_params}")
                                old_weights = reward_wrapper_train.get_weights()

                                # Visualize buffer BEFORE recompute
                                vis_dir = debug_dir / f"segment_{current_segment+1:02d}_buffer_vis"
                                vis_dir.mkdir(exist_ok=True, parents=True)
                                rb.visualize_rewards(
                                    str(vis_dir / "before_recompute.png"),
                                    title=f"Segment {current_segment+1} - Before Recompute (old weights)"
                                )

                                reward_wrapper_train.update_weights(new_params)
                                reward_wrapper_eval.update_weights(new_params)

                                # Recompute rewards in replay buffer with new weights.
                                # Statistical normalization ensures Q value stability by maintaining
                                # reward distribution (mean/std) even when weight changes affect component scales.
                                recompute_stats = rb.recompute_rewards(reward_wrapper_train, normalize_stats=True)
                                print(f"[SAC] Replay buffer rewards recomputed with new weights "
                                      f"({rb.per_env_buffer_size if rb.full else rb.pos} steps retained)")
                                if recompute_stats:
                                    print(f"  Old: mean={recompute_stats['old_mean']:.3f}, std={recompute_stats['old_std']:.3f}")
                                    print(f"  New (raw): mean={recompute_stats['new_mean_raw']:.3f}, std={recompute_stats['new_std_raw']:.3f}")
                                    print(f"  Final (normalized): mean={recompute_stats['final_mean']:.3f}, std={recompute_stats['final_std']:.3f}")

                                # Log statistics to wandb/tensorboard
                                if logger is not None and recompute_stats:
                                    logger.add_scalar("recompute/old_mean", recompute_stats['old_mean'], global_step)
                                    logger.add_scalar("recompute/old_std", recompute_stats['old_std'], global_step)
                                    logger.add_scalar("recompute/new_mean_raw", recompute_stats['new_mean_raw'], global_step)
                                    logger.add_scalar("recompute/new_std_raw", recompute_stats['new_std_raw'], global_step)
                                    logger.add_scalar("recompute/final_mean", recompute_stats['final_mean'], global_step)
                                    logger.add_scalar("recompute/final_std", recompute_stats['final_std'], global_step)

                                # Visualize buffer AFTER recompute
                                rb.visualize_rewards(
                                    str(vis_dir / "after_recompute.png"),
                                    title=f"Segment {current_segment+1} - After Recompute (normalized)"
                                )

                                # Activate critic warmup: freeze actor updates for N steps
                                critic_warmup_remaining = args.critic_warmup_steps
                                print(f"[SAC] Critic warmup activated: actor frozen for {critic_warmup_remaining} updates")

                                segment_history.append({
                                    "segment": current_segment,
                                    "global_step": global_step,
                                    "old_weights": old_weights,
                                    "new_weights": reward_wrapper_train.get_weights(),
                                    "rationale": rationale,
                                    "vlm_comment": vlm_comment,
                                    "eval_metrics": latest_eval_metrics,
                                })

                                if logger is not None:
                                    for wk, wv in reward_wrapper_train.get_weights().items():
                                        logger.add_scalar(f"weights/{wk}", wv, global_step)
                            else:
                                print("[LLM] No weight changes suggested")
                        else:
                            print(f"[LLM] Unexpected suggestion type: {suggestions}")
                    except Exception as e:
                        print(f"[LLM] Error: {e}")
            else:
                print(f"[info] VLM/LLM not active, skipping analysis for segment {current_segment+1}")
                if args.clear_buffer_at_segment:
                    rb.clear()
                    print(f"[DEBUG] Replay buffer cleared at segment boundary "
                          f"(--clear_buffer_at_segment, no weight change)")

            current_segment += 1
            next_segment_step = steps_per_segment * (current_segment + 1)

        # --- Collect samples ---
        rollout_time = time.perf_counter()
        for local_step in range(args.steps_per_env):
            global_step += 1 * args.num_envs

            if not learning_has_started:
                actions = 2 * torch.rand(size=envs.action_space.shape, dtype=torch.float32, device=device) - 1
            else:
                actions, _, _ = actor.get_action(obs)
                actions = actions.detach()

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = next_obs.clone()
            if args.bootstrap_at_done == 'never':
                need_final_obs = torch.ones_like(terminations, dtype=torch.bool)
                stop_bootstrap = truncations | terminations
            else:
                if args.bootstrap_at_done == 'always':
                    need_final_obs = truncations | terminations
                    stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool)
                else:
                    need_final_obs = truncations & (~terminations)
                    stop_bootstrap = terminations
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                real_next_obs[need_final_obs] = infos["final_observation"][need_final_obs]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

            rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap,
                   components=reward_wrapper_train._last_components,
                   success=reward_wrapper_train._last_success)
            obs = next_obs

        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.num_envs * args.steps_per_env)

        # --- Training ---
        if global_step < args.learning_starts:
            continue
        # Skip training if buffer was recently cleared and has too few samples
        if not rb.full and rb.pos < args.batch_size // args.num_envs:
            continue

        update_time = time.perf_counter()
        learning_has_started = True
        for local_update in range(args.grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(args.batch_size)

            # update the value networks
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_obs)
                qf1_next_target = qf1_target(data.next_obs, next_state_actions)
                qf2_next_target = qf2_target(data.next_obs, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.obs, data.actions).view(-1)
            qf2_a_values = qf2(data.obs, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # Critic warmup: skip actor/alpha updates after reward recompute
            if critic_warmup_remaining > 0:
                critic_warmup_remaining -= 1
                if critic_warmup_remaining % 1000 == 0 or critic_warmup_remaining == 0:
                    print(f"[Critic warmup] {critic_warmup_remaining} updates remaining (actor frozen)")
                # Skip actor/alpha updates during warmup
                actor_loss = torch.tensor(0.0)  # For logging
            else:
                # update the policy network
                if global_update % args.policy_frequency == 0:
                    pi, log_pi, _ = actor.get_action(data.obs)
                    qf1_pi = qf1(data.obs, pi)
                    qf2_pi = qf2(data.obs, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.obs)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_update % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # Log training-related data
        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            logger.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            logger.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            logger.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            logger.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            logger.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            logger.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            logger.add_scalar("losses/alpha", alpha, global_step)
            logger.add_scalar("critic_warmup/remaining", critic_warmup_remaining, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", global_steps_per_iteration / rollout_time, global_step)
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)
            logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
            if args.autotune:
                logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    # --- Save final model and history ---
    if not args.evaluate and args.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save({
            'actor': actor.state_dict(),
            'qf1': qf1_target.state_dict(),
            'qf2': qf2_target.state_dict(),
            'log_alpha': log_alpha,
        }, model_path)
        print(f"model saved to {model_path}")

    if not args.evaluate:
        # Save segment history
        history_path = f"runs/{run_name}/segment_history.json"
        with open(history_path, "w") as f:
            json.dump(segment_history, f, indent=2, default=str)
        print(f"segment history saved to {history_path}")

        # Save final weights
        final_weights_path = f"runs/{run_name}/final_weights.json"
        with open(final_weights_path, "w") as f:
            json.dump(reward_wrapper_train.get_weights(), f, indent=2)
        print(f"final weights saved to {final_weights_path}")

        logger.close()

    envs.close()
    eval_envs.close()

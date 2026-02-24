"""
PPO Outer Loop: Eureka Full Implementation (LLM-generated Reward Functions).

This script implements the complete Eureka algorithm from the paper:
- Generate K reward function candidates per iteration
- Train and evaluate each candidate
- Select the best candidate based on fitness (success_rate)
- Use Reward Reflection to improve next iteration

Unlike ppo_outer_loop.py (params-only), this script allows LLM to generate
entire reward functions, not just adjust weights.

Algorithm (Eureka Paper, Algorithm 1):
    for N outer iterations:
        1. Generate K reward function candidates (LLM)
        2. Train each candidate (PPO)
        3. Evaluate fitness (success_rate)
        4. Select best candidate
        5. (Optional) VLM analysis of best candidate
        6. Reward Reflection: analyze results, provide feedback to LLM
        7. Update prompt for next iteration

Usage:
    # Full Eureka mode (with VLM):
    export OPENAI_API_KEY=sk-... && python ppo_outer_loop_full.py \
        --env_id PushCube-v1 --num_reward_candidates=4 --num_outer_iters=5

    # Eureka without VLM (LLM-only):
    export OPENAI_API_KEY=sk-... && python ppo_outer_loop_full.py \
        --env_id PushCube-v1 --eureka_mode --num_reward_candidates=4

    # Debug mode (no LLM):
    python ppo_outer_loop_full.py --env_id PushCube-v1 --skip_vlm_llm --num_outer_iters=1
"""

import base64
import inspect
import io
import json
import os
import pickle
import subprocess
import random as py_random
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from coupled_allegro_wrapper import CoupledAllegroActionWrapper
from env_contracts import validate_env_setup
from reward_wrapper_dynamic import RewardWrapperDynamic, TASK_DEFAULTS, _resolve_task_id


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "PushCube-v1"
    """the id of the environment"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    control_mode: Optional[str] = None
    """the control mode to use for the environment (default: pd_ee_delta_pose for PandaAllegro, pd_joint_delta_pos for others)"""
    anneal_lr: bool = False
    gamma: float = 0.8
    gae_lambda: float = 0.9
    num_minibatches: int = 32
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.1
    reward_scale: float = 1.0
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    finite_horizon_gae: bool = False

    # Outer loop arguments
    num_outer_iters: int = 5
    """number of outer loop iterations (each is a full PPO training from scratch)"""
    total_timesteps_per_iter: int = 2_000_000
    """total timesteps per outer iteration"""
    initial_weights_file: Optional[str] = None
    """path to JSON file with initial weights (if None, generate random)"""
    weight_seed: int = 42
    """seed for random weight generation"""
    resume_dir: Optional[str] = None
    """path to a previous run directory to resume from (e.g., runs/outer-loop_full/PushCube-v1/...). Creates a new branched directory; original is not modified."""

    # VLM/LLM arguments
    vlm_model: str = "gpt-5.2"
    """VLM model for video analysis"""
    llm_model: str = "gpt-5.2"
    """LLM model for reward tuning"""
    vlm_max_frames: int = 8
    """max frames to send to VLM"""
    vlm_num_envs: int = 1
    """number of eval envs to show in VLM frames"""
    vlm_reward_plot: bool = False
    """if toggled, append per-step reward plot to VLM debug HTML"""
    rl_project_path: str = "/home/nwake/codes/RL_project"
    """path to RL_project for VLM/LLM imports"""
    skip_vlm_llm: bool = False
    """skip VLM/LLM calls (for testing)"""
    eureka_mode: bool = False
    """pure Eureka mode: use LLM only without VLM (learning curve based optimization)"""
    enable_function_code: bool = True
    """allow LLM to generate custom reward code (Eureka-style). When False, params-only mode."""
    num_reward_candidates: int = 4
    """number of reward function candidates to generate per iteration (K in Eureka paper)"""
    enable_reward_reflection: bool = True
    """enable Reward Reflection: analyze learning curves and provide feedback to LLM"""

    # Parallel K-candidate training
    gpus: Optional[str] = None
    """comma-separated GPU IDs for parallel K-candidate training (e.g., '0,1'). If None, sequential on current device."""

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


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
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

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
# Parallel K-candidate worker mode
# ============================================================================

def _run_worker_mode(task_path: str):
    """Internal subprocess worker: train a single candidate on the assigned GPU.

    Called via: python ppo_outer_loop_full.py _worker <task_path>
    CUDA_VISIBLE_DEVICES is set by the parent process before spawning.
    """
    with open(task_path, "rb") as f:
        task = pickle.load(f)

    args = task["args"]
    cand = task["cand"]
    outer_iter = task["outer_iter"]
    cand_run_dir = task["run_dir"]
    global_step_offset = task["global_step_offset"]
    result_path = task["result_path"]

    device = torch.device("cuda")

    # Create per-candidate tensorboard logger
    writer = SummaryWriter(f"runs/{cand_run_dir}")
    worker_logger = Logger(log_wandb=False, tensorboard=writer)

    try:
        result = run_ppo_training(
            args=args,
            weights=None,
            outer_iter=outer_iter,
            run_dir=cand_run_dir,
            logger=worker_logger,
            device=device,
            global_step_offset=global_step_offset,
            custom_code=cand["code"],
        )

        # Convert step_rewards tensors to serializable reward_stats
        step_rewards = result.get("step_rewards", [])
        if step_rewards:
            stacked = torch.stack(step_rewards).cpu()
            reward_stats = {
                "mean": stacked.mean().item(),
                "std": stacked.std().item(),
                "min": stacked.min().item(),
                "max": stacked.max().item(),
            }
        else:
            reward_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        output = {
            "success": True,
            "candidate_id": cand["id"],
            "code": cand["code"],
            "rationale": cand["rationale"],
            "is_elite": cand.get("is_elite", False),
            "eval_metrics": result["eval_metrics"],
            "learning_curve": result["learning_curve"],
            "reward_stats": reward_stats,
            "eval_video_dir": result["eval_video_dir"],
            "final_global_step": result["final_global_step"],
        }

    except Exception as e:
        error_tb = traceback.format_exc()
        print(f"[Worker] Candidate {cand['id']+1} FAILED: {type(e).__name__}: {e}")
        print(error_tb)
        output = {
            "success": False,
            "candidate_id": cand["id"],
            "code": cand["code"],
            "rationale": cand["rationale"],
            "is_elite": cand.get("is_elite", False),
            "error": f"{type(e).__name__}: {e}",
            "traceback": error_tb,
        }

    worker_logger.close()

    with open(result_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[Worker] Result written to {result_path}")


def _train_candidates_parallel(
    args: "Args",
    candidates: List[Dict],
    outer_iter: int,
    run_dir: str,
    global_step_offset: int,
    gpu_list: List[int],
    llm: Any,
    training_summary: Dict,
    save_llm_debug_html: Any,
    debug_dir: Path,
) -> List[Dict[str, Any]]:
    """Train K candidates in parallel across GPUs using subprocesses.

    Candidates are batched by len(gpu_list). Failed candidates are retried
    sequentially with LLM error-fixing (same as sequential mode).

    Returns list of candidate_results (same format as sequential mode).
    """
    MAX_RUNTIME_RETRIES = 2
    candidate_results = []
    failed_candidates = []  # (cand, error, traceback)

    # --- Phase 1: Parallel training ---
    for batch_start in range(0, len(candidates), len(gpu_list)):
        batch = candidates[batch_start:batch_start + len(gpu_list)]
        procs = []

        print(f"\n  [Parallel] Batch: candidates {[c['id']+1 for c in batch]} "
              f"on GPUs {gpu_list[:len(batch)]}")

        for i, cand in enumerate(batch):
            gpu_id = gpu_list[i]
            cand_run_dir = f"{run_dir}/cand_{cand['id']}"

            Path(f"runs/{cand_run_dir}").mkdir(parents=True, exist_ok=True)

            task_path = f"runs/{run_dir}/_cand_{cand['id']}_task.pkl"
            result_path = f"runs/{run_dir}/_cand_{cand['id']}_result.json"

            task = {
                "args": args,
                "cand": cand,
                "outer_iter": outer_iter,
                "run_dir": cand_run_dir,
                "global_step_offset": global_step_offset,
                "result_path": result_path,
            }
            with open(task_path, "wb") as f:
                pickle.dump(task, f)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            log_path = f"runs/{run_dir}/_cand_{cand['id']}_stdout.log"
            log_file = open(log_path, "w")

            cmd = [sys.executable, "-u", os.path.abspath(__file__), "_worker", task_path]
            proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
            procs.append((proc, cand, result_path, log_file, log_path))
            print(f"    Candidate {cand['id']+1} -> GPU {gpu_id} (PID: {proc.pid})")

        # Wait for all processes in this batch
        for proc, cand, result_path, log_file, log_path in procs:
            proc.wait()
            log_file.close()

            if proc.returncode == 0 and Path(result_path).exists():
                with open(result_path) as f:
                    result = json.load(f)

                if result.get("success", False):
                    fitness = result["eval_metrics"].get("success_at_end", 0.0)
                    _lc = result.get("learning_curve", [])
                    _peak_success_once = max(
                        (lc.get("success_once", 0.0) for lc in _lc),
                        default=result["eval_metrics"].get("success_once", 0.0),
                    )
                    candidate_results.append({
                        "candidate_id": result["candidate_id"],
                        "code": result["code"],
                        "rationale": result["rationale"],
                        "is_elite": result["is_elite"],
                        "fitness": fitness,
                        "fitness_success_at_end": fitness,
                        "fitness_success_once": _peak_success_once,
                        "fitness_return": result["eval_metrics"].get("return", float("-inf")),
                        "eval_metrics": result["eval_metrics"],
                        "learning_curve": result["learning_curve"],
                        "reward_statistics": result["reward_stats"],
                        "eval_video_dir": result["eval_video_dir"],
                    })
                    print(f"    Candidate {cand['id']+1} OK (fitness={fitness:.4f})")
                else:
                    # Runtime error captured by worker
                    print(f"    Candidate {cand['id']+1} FAILED (runtime error)")
                    failed_candidates.append((
                        cand,
                        result.get("error", "Unknown error"),
                        result.get("traceback", ""),
                    ))
            else:
                # Process-level failure
                print(f"    Candidate {cand['id']+1} FAILED (exit code: {proc.returncode})")
                print(f"    See log: {log_path}")
                failed_candidates.append((cand, f"Process exited with code {proc.returncode}", ""))

    # --- Phase 2: Sequential LLM retry for failed candidates ---
    if failed_candidates and llm is not None:
        print(f"\n  [Retry] {len(failed_candidates)} failed candidate(s), attempting LLM fix...")

        for cand, error_msg, error_tb in failed_candidates:
            current_code = cand["code"]
            current_rationale = cand["rationale"]

            for attempt in range(MAX_RUNTIME_RETRIES):
                print(f"\n  [Retry {attempt+1}/{MAX_RUNTIME_RETRIES}] Candidate {cand['id']+1}")

                fix_summary = {
                    **training_summary,
                    "previous_code_error": {
                        "code": current_code,
                        "error": f"{error_msg}\n\n{error_tb}",
                        "instruction": (
                            "前回のコードでランタイムエラーが発生しました。\n"
                            "エラーを修正した新しいコードを生成してください。\n\n"
                            "よくあるランタイムエラー:\n"
                            "1. base.device が存在しない → base.obj.pose.p.device を使う\n"
                            "2. テンソルのshape不一致（(B,3)に対してスカラー操作等）\n"
                            "3. 存在しない属性へのアクセス（State Access Docsを参照）\n"
                            "4. torch演算のdevice不一致（CPU/CUDA混在）\n"
                            "5. info dictのキーが存在しない\n"
                            "6. hasattr/setattr on batched env objects\n"
                            "7. 型アノテーションで __import__ を使用 → 'torch.Tensor' を使う\n\n"
                            "修正後のコードをfunction_code形式で返してください。"
                        )
                    }
                }

                try:
                    suggestions = llm.suggest_parameters(fix_summary)
                except (ValueError, SyntaxError) as e:
                    print(f"    LLM suggest_parameters failed: {e}")
                    continue

                # Save retry debug HTML
                if save_llm_debug_html is not None:
                    query_info = llm.get_last_query_info() if hasattr(llm, 'get_last_query_info') else None
                    llm_prompt = query_info.get("prompt", "(no prompt)") if query_info else "(no query info)"
                    llm_response = query_info.get("response_text", "(no response)") if query_info else "(no query info)"
                    save_llm_debug_html(
                        iteration=outer_iter,
                        prompt=llm_prompt,
                        response_text=llm_response,
                        suggestions=suggestions,
                        summary_for_llm=fix_summary,
                        save_path=debug_dir / f"iter_{outer_iter+1:02d}_cand_{cand['id']}_parallel_retry{attempt}_llm.html",
                    )

                if suggestions and suggestions.get("type") == "function_code":
                    new_code = suggestions["custom_code"]
                    test_fn, compile_error = RewardWrapperDynamic._compile_custom_function_with_error(new_code)
                    if test_fn is not None:
                        current_code = new_code
                        current_rationale = suggestions.get("rationale", f"LLM fix (retry {attempt+1})")
                        print(f"    LLM fix compiled, re-training on GPU {gpu_list[0]}...")

                        # Re-train via subprocess on first GPU
                        retry_cand = {**cand, "code": current_code, "rationale": current_rationale}
                        retry_run_dir = f"{run_dir}/cand_{cand['id']}_retry{attempt+1}"
                        Path(f"runs/{retry_run_dir}").mkdir(parents=True, exist_ok=True)

                        retry_task_path = f"runs/{run_dir}/_cand_{cand['id']}_retry{attempt+1}_task.pkl"
                        retry_result_path = f"runs/{run_dir}/_cand_{cand['id']}_retry{attempt+1}_result.json"
                        retry_task = {
                            "args": args,
                            "cand": retry_cand,
                            "outer_iter": outer_iter,
                            "run_dir": retry_run_dir,
                            "global_step_offset": global_step_offset,
                            "result_path": retry_result_path,
                        }
                        with open(retry_task_path, "wb") as f:
                            pickle.dump(retry_task, f)

                        retry_env = os.environ.copy()
                        retry_env["CUDA_VISIBLE_DEVICES"] = str(gpu_list[0])
                        retry_log_path = f"runs/{run_dir}/_cand_{cand['id']}_retry{attempt+1}_stdout.log"
                        retry_log_file = open(retry_log_path, "w")

                        retry_cmd = [sys.executable, "-u", os.path.abspath(__file__), "_worker", retry_task_path]
                        retry_proc = subprocess.Popen(retry_cmd, env=retry_env, stdout=retry_log_file, stderr=subprocess.STDOUT)
                        retry_proc.wait()
                        retry_log_file.close()

                        if retry_proc.returncode == 0 and Path(retry_result_path).exists():
                            with open(retry_result_path) as f:
                                retry_result = json.load(f)

                            if retry_result.get("success", False):
                                fitness = retry_result["eval_metrics"].get("success_at_end", 0.0)
                                _lc = retry_result.get("learning_curve", [])
                                _peak_success_once = max(
                                    (lc.get("success_once", 0.0) for lc in _lc),
                                    default=retry_result["eval_metrics"].get("success_once", 0.0),
                                )
                                candidate_results.append({
                                    "candidate_id": retry_result["candidate_id"],
                                    "code": retry_result["code"],
                                    "rationale": retry_result["rationale"],
                                    "is_elite": retry_result["is_elite"],
                                    "fitness": fitness,
                                    "fitness_success_at_end": fitness,
                                    "fitness_success_once": _peak_success_once,
                                    "fitness_return": retry_result["eval_metrics"].get("return", float("-inf")),
                                    "eval_metrics": retry_result["eval_metrics"],
                                    "learning_curve": retry_result["learning_curve"],
                                    "reward_statistics": retry_result["reward_stats"],
                                    "eval_video_dir": retry_result["eval_video_dir"],
                                })
                                print(f"    Retry OK (fitness={fitness:.4f})")
                                break  # Success, stop retrying this candidate
                            else:
                                error_msg = retry_result.get("error", "Unknown error")
                                error_tb = retry_result.get("traceback", "")
                                print(f"    Retry failed: {error_msg[:200]}")
                        else:
                            print(f"    Retry process failed (exit code: {retry_proc.returncode})")
                            error_msg = f"Retry process exited with code {retry_proc.returncode}"
                            error_tb = ""
                    else:
                        print(f"    LLM fix failed compilation: {compile_error[:200]}")
                else:
                    sug_type = suggestions.get("type", "N/A") if suggestions else "N/A"
                    print(f"    LLM returned wrong type: {sug_type}")

    return candidate_results


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


# ============================================================================
# Single PPO training run
# ============================================================================

def run_ppo_training(
    args: Args,
    weights: Optional[Dict[str, float]],
    outer_iter: int,
    run_dir: str,
    logger: Optional["Logger"],
    device: torch.device,
    global_step_offset: int,
    custom_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single PPO training from scratch with fixed weights or custom reward function.

    Args:
        custom_code: If provided, replaces reward computation with custom function (Eureka mode).
                     If None, uses YAML-based rewards with weights.

    Returns:
        Dict with eval metrics, per-step reward data, and video path.
    """
    # Reset RNG so each iteration starts from identical state
    py_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps_per_iter // args.batch_size

    print(f"\n[Outer Iter {outer_iter+1}] Starting PPO training")
    if custom_code is not None:
        print(f"  Mode: Eureka full replacement (custom function)")
    else:
        print(f"  Mode: Weight-based (weights={weights})")
    print(f"  num_iterations={args.num_iterations}, batch_size={args.batch_size}")

    # Create output directory (for model saving, videos, etc.)
    from pathlib import Path
    Path(f"runs/{run_dir}").mkdir(parents=True, exist_ok=True)

    # --- Environment setup ---
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        reward_mode="none",
    )
    # Resolve control_mode: CLI explicit > task default
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    elif "PandaAllegro" in args.env_id:
        env_kwargs["control_mode"] = "pd_ee_delta_pose"
    else:
        env_kwargs["control_mode"] = "pd_joint_delta_pos"

    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,
    )
    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **env_kwargs,
    )

    if "PandaAllegro" in args.env_id:
        # Use coupled 8D action space (6 arm EE + 2 hand scalars)
        envs = CoupledAllegroActionWrapper(envs)
        eval_envs = CoupledAllegroActionWrapper(eval_envs)
    elif isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    validate_env_setup(args.env_id, env_kwargs["control_mode"], envs)

    # RewardWrapperDynamic with fixed weights (supports custom functions)
    # In Eureka mode (custom_code provided), raise on runtime errors so the
    # outer loop's retry mechanism can detect and fix broken reward functions.
    raise_on_err = custom_code is not None
    reward_wrapper_train = RewardWrapperDynamic(envs, env_id=args.env_id, weights=weights, raise_on_custom_fn_error=raise_on_err)
    reward_wrapper_eval = RewardWrapperDynamic(eval_envs, env_id=args.env_id, weights=weights, raise_on_custom_fn_error=raise_on_err)
    envs = reward_wrapper_train
    eval_envs = reward_wrapper_eval

    # Set custom reward function if provided (Eureka full replacement mode)
    # Must be called BEFORE ManiSkillVectorEnv wrapping
    # ManiSkill uses GPU batch environments, not separate processes, so one call applies to all
    if custom_code is not None:
        print(f"  Setting custom reward function (Eureka mode)")
        reward_wrapper_train.set_custom_function(custom_fn=None, code=custom_code)
        reward_wrapper_eval.set_custom_function(custom_fn=None, code=custom_code)

    # Video recording for eval
    eval_output_dir = f"runs/{run_dir}/videos/iter_{outer_iter+1:02d}"
    if args.capture_video:
        print(f"  Saving eval videos to {eval_output_dir}")
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=False,
            max_steps_per_video=args.num_eval_steps,
            video_fps=30,
        )

    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    # --- Agent (fresh initialization) ---
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # --- Storage ---
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # --- Training ---
    global_step = global_step_offset
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    action_space_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_space_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    latest_eval_metrics: Dict[str, Any] = {}
    latest_eval_step_rewards: List[torch.Tensor] = []
    latest_eval_reward_breakdowns: List[Dict[str, float]] = []

    # Collect learning curve: eval metrics at each eval point (Eureka-style)
    learning_curve: List[Dict[str, Any]] = []

    for iteration in range(1, args.num_iterations + 1):
        print(f"  Epoch: {iteration}/{args.num_iterations}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()

        # --- Evaluation ---
        is_last_iter = (iteration == args.num_iterations)
        if iteration % args.eval_freq == 1 or is_last_iter:
            print("  Evaluating")
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            step_rewards_list = []
            step_breakdowns_list = []
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(
                        agent.get_action(eval_obs, deterministic=True)
                    )
                    step_rewards_list.append(eval_rew.detach())
                    step_breakdowns_list.append(dict(reward_wrapper_eval._last_breakdown))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"  Evaluated {args.num_eval_steps * args.num_eval_envs} steps, {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"  eval_{k}_mean={mean}")
            latest_eval_metrics = {
                k: torch.stack(v).float().mean().item() for k, v in eval_metrics.items()
            }
            if isinstance(num_episodes, torch.Tensor):
                latest_eval_metrics["num_episodes"] = int(num_episodes.item())
            else:
                latest_eval_metrics["num_episodes"] = int(num_episodes)

            # Store per-step reward data
            latest_eval_step_rewards = step_rewards_list
            latest_eval_reward_breakdowns = step_breakdowns_list

            # Record learning curve point (Eureka-style)
            lc_point = {
                "step": global_step,
                "avg_return": latest_eval_metrics.get("return", 0.0),
                "success_at_end": latest_eval_metrics.get("success_at_end", 0.0),
                "success_once": latest_eval_metrics.get("success_once", 0.0),
                "success_rate": latest_eval_metrics.get("success_at_end", 0.0),
                "episode_len": latest_eval_metrics.get("episode_len", 0.0),
            }
            # Add per-component reward means (Eureka-style)
            if step_breakdowns_list:
                comp_means = {}
                for key in step_breakdowns_list[0]:
                    if key == "norm_scale":
                        continue
                    vals = [bd[key] for bd in step_breakdowns_list]
                    comp_means[key] = sum(vals) / len(vals)
                lc_point["reward_components"] = comp_means
            learning_curve.append(lc_point)

        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_dir}/iter_{outer_iter+1:02d}_ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)

        # LR annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # --- Rollout ---
        rollout_time = time.time()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    if logger is not None:
                        logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(
                        infos["final_observation"][done_mask]
                    ).view(-1)
        rollout_time = time.time() - rollout_time

        # --- GAE ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                if args.finite_horizon_gae:
                    if t == args.num_steps - 1:
                        lam_coef_sum = 0.0
                        reward_term_sum = 0.0
                        value_term_sum = 0.0
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done
                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values
                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
            returns = advantages + values

        # --- PPO Update ---
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.time()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_time

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if logger is not None:
            logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            logger.add_scalar("losses/explained_variance", explained_var, global_step)
            sps = int(global_step / (time.time() - start_time))
            logger.add_scalar("charts/SPS", sps, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)

    # Save final model for this iteration
    if args.save_model:
        model_path = f"runs/{run_dir}/iter_{outer_iter+1:02d}_final.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"  Model saved to {model_path}")

    # Close envs
    envs.close()
    eval_envs.close()

    return {
        "eval_metrics": latest_eval_metrics,
        "step_rewards": latest_eval_step_rewards,
        "reward_breakdowns": latest_eval_reward_breakdowns,
        "eval_video_dir": eval_output_dir,
        "final_global_step": global_step,
        "learning_curve": learning_curve,
    }


# ============================================================================
# Main: Outer Loop
# ============================================================================

if __name__ == "__main__":
    # Internal worker mode for parallel K-candidate training.
    # Invoked as: python ppo_outer_loop_full.py _worker <task_pickle_path>
    # CUDA_VISIBLE_DEVICES is set by the parent process.
    if len(sys.argv) >= 3 and sys.argv[1] == "_worker":
        _run_worker_mode(sys.argv[2])
        sys.exit(0)

    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.exp_name}-{args.env_id}-{timestamp}"
    experiment_type = "eureka_full" if args.eureka_mode else "outer-loop_full"
    run_dir = f"{experiment_type}/{args.env_id}/{run_name}"

    # --- Resume from previous run (creates a new branched directory) ---
    _resumed_history = None
    _resume_start_iter = 0
    _resume_global_step_offset = 0
    if args.resume_dir is not None:
        # Resolve the source directory (accept both "runs/..." and bare paths)
        _src_dir = Path(args.resume_dir)
        if not _src_dir.exists():
            _src_dir = Path("runs") / args.resume_dir
        if not _src_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {args.resume_dir}")

        _hist_path = _src_dir / "outer_loop_history.json"
        if not _hist_path.exists():
            raise FileNotFoundError(f"No outer_loop_history.json in {_src_dir}")

        with open(_hist_path) as f:
            _resumed_history = json.load(f)
        _resume_start_iter = len(_resumed_history)
        _resume_global_step_offset = _resume_start_iter * args.total_timesteps_per_iter

        # Branch: new directory name derived from original + resume timestamp
        _orig_name = _src_dir.name
        run_name = f"{_orig_name}_resume{_resume_start_iter}_{timestamp}"
        run_dir = f"{experiment_type}/{args.env_id}/{run_name}"

        # Copy debug_html from source so VLM/LLM history files are accessible
        import shutil
        _src_debug = _src_dir / "debug_html"
        _dst_debug = Path(f"runs/{run_dir}/debug_html")
        _dst_debug.mkdir(parents=True, exist_ok=True)
        if _src_debug.exists():
            for _f in _src_debug.iterdir():
                shutil.copy2(_f, _dst_debug / _f.name)

        # Copy TensorBoard events from each cand_* directory (for plot_outer_loop_full_summary.py)
        # Only copies event files (small), NOT checkpoints or videos (large).
        for _cand_dir in sorted(_src_dir.glob("cand_*")):
            if _cand_dir.is_dir():
                _dst_cand = Path(f"runs/{run_dir}") / _cand_dir.name
                _dst_cand.mkdir(parents=True, exist_ok=True)
                for _ev in _cand_dir.glob("events.out.tfevents.*"):
                    shutil.copy2(_ev, _dst_cand / _ev.name)

        # Copy root-level TensorBoard events
        _dst_run = Path(f"runs/{run_dir}")
        _dst_run.mkdir(parents=True, exist_ok=True)
        for _ev in _src_dir.glob("events.out.tfevents.*"):
            shutil.copy2(_ev, _dst_run / _ev.name)

        # Copy summary image if exists
        _src_img = _src_dir / "outer_loop_full_summary.png"
        if _src_img.exists():
            shutil.copy2(_src_img, _dst_run / "outer_loop_full_summary.png")

        # Adjust num_outer_iters to be total (start_iter + additional)
        # User specifies --num_outer_iters as ADDITIONAL iterations to run
        args.num_outer_iters = _resume_start_iter + args.num_outer_iters

        print(f"\n{'='*60}")
        print(f"RESUMING from: {_src_dir}")
        print(f"  Previous iterations: {_resume_start_iter}")
        print(f"  Additional iterations: {args.num_outer_iters - _resume_start_iter}")
        print(f"  Total iterations: {args.num_outer_iters}")
        print(f"  Branched to: runs/{run_dir}")
        print(f"{'='*60}\n")

    # Seeding
    py_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # --- Initial weights ---
    if args.initial_weights_file:
        with open(args.initial_weights_file) as f:
            current_weights = json.load(f)
        print(f"Loaded initial weights from {args.initial_weights_file}: {current_weights}")
    else:
        current_weights = generate_random_weights(args.env_id, seed=args.weight_seed)
        print(f"Generated random weights (seed={args.weight_seed}): {current_weights}")

    initial_weights = dict(current_weights)

    # --- Logging ---
    logger = None
    if args.track:
        import wandb
        config = vars(args)
        config["initial_weights"] = initial_weights
        # Set tags based on mode
        tags = ["ppo", "outer-loop"]
        if args.eureka_mode:
            tags.extend(["eureka-full", "llm-only"])
        else:
            tags.extend(["vlm-llm"])
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=config,
            name=run_name,
            save_code=True,
            group="PPO-OuterLoop",
            tags=tags,
        )
    writer = SummaryWriter(f"runs/{run_dir}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    logger = Logger(log_wandb=args.track, tensorboard=writer)

    # --- VLM/LLM setup (once) ---
    vlm = None
    llm = None
    save_vlm_debug_html = None
    save_llm_debug_html = None
    if not args.skip_vlm_llm:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            # Mock stable_baselines3 to avoid dependency
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
            from experiments.iterative_learner import (
                save_vlm_debug_html as _save_vlm_debug_html,
                save_llm_debug_html as _save_llm_debug_html,
            )
            save_vlm_debug_html = _save_vlm_debug_html
            save_llm_debug_html = _save_llm_debug_html

            # Initialize VLM only if not in Eureka mode
            if not args.eureka_mode:
                vlm = VLMEvaluator.from_openai(
                    api_key=api_key,
                    model=args.vlm_model,
                    prompt=build_vlm_prompt(args.env_id),
                    max_frames=args.vlm_max_frames,
                    cache_results=False,
                )
                print(f"[VLM/LLM] Initialized VLM: {args.vlm_model}")
            else:
                print("[Eureka Mode] Skipping VLM initialization (LLM-only mode)")

            # Always initialize LLM
            llm = LLMRewardTuner.from_openai(
                api_key=api_key,
                model=args.llm_model,
                enable_function_code=args.enable_function_code,
                max_param_change=2.0,
                temperature=1.0,  # High temperature for diverse reward function candidates
            )
            print(f"[VLM/LLM] Initialized LLM: {args.llm_model} (temperature=1.0)")
        else:
            print("[warn] OPENAI_API_KEY not set, skipping VLM/LLM")
    else:
        print("[info] VLM/LLM disabled (--skip_vlm_llm)")

    print(f"\n{'='*60}")
    print(f"PPO Outer Loop: {args.num_outer_iters} iterations x {args.total_timesteps_per_iter} steps")
    print(f"Initial weights: {current_weights}")
    print(f"{'='*60}\n")

    # --- Outer Loop (Eureka Full Replacement Mode) ---
    outer_loop_history = []
    global_step_offset = 0
    training_summary = {}  # Carries Reflection context across iterations
    last_good_code = None  # Track last successful reward code across iterations

    # Restore state from previous run if resuming
    if _resumed_history is not None:
        outer_loop_history = list(_resumed_history)
        global_step_offset = _resume_global_step_offset
        # Restore last_good_code from the best candidate of the last iteration
        if outer_loop_history:
            _prev_best = outer_loop_history[-1]["best_candidate"]
            last_good_code = _prev_best.get("code")
        print(f"[Resume] Restored {len(outer_loop_history)} iterations, global_step_offset={global_step_offset}")

    for outer_iter in range(_resume_start_iter, args.num_outer_iters):
        print(f"\n{'='*60}")
        print(f"OUTER ITERATION {outer_iter+1}/{args.num_outer_iters}")
        print(f"{'='*60}")

        # Prepare debug directory (used for LLM/VLM logs and candidate comparison)
        debug_dir = Path(f"runs/{run_dir}/debug_html")
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Special handling for Iteration 0: no Reflection, no VLM
        if outer_iter == 0:
            print("  [Iteration 0] Initial iteration: no Reflection, no VLM")
            training_summary.pop("reflection_history", None)

        # ========== STEP 1: Generate K reward function candidates ==========
        candidates = []

        # Elite carry-over: Iteration 1+ reserves 1 slot for previous best
        if outer_iter > 0:
            prev_best = outer_loop_history[-1]["best_candidate"]
            if prev_best.get("code") is not None:
                candidates.append({
                    "code": prev_best["code"],
                    "rationale": f"Elite carry-over from Iteration {outer_iter} (fitness={prev_best['fitness']:.4f})",
                    "id": 0,
                    "is_elite": True,
                })
                print(f"  Elite candidate added (prev best fitness={prev_best['fitness']:.4f})")
            else:
                print(f"  Skipping elite carry-over (previous iteration had no valid candidates)")

        llm_candidate_count = args.num_reward_candidates - len(candidates)

        if llm is not None and args.enable_function_code:
            print(f"\n[Eureka] Generating {llm_candidate_count} LLM candidates (total slots: {args.num_reward_candidates})...")

            # Build task description for LLM
            task_id = _resolve_task_id(args.env_id)
            _llm_task_descs = {
                "PushCube": (
                    "The robot arm must push a cube to the goal position (PushCube).\n"
                    "Reward components: w_reach (TCP approach to push position behind cube), "
                    "w_push (cube movement toward goal, gated by reach), "
                    "w_z_keep (keep cube on table surface), w_success (success bonus).\n\n"
                    "日本語補足: ロボットアームがキューブを目標位置まで押すタスク。"
                    "回答の末尾に日本語での簡潔な要約も追加してください。"
                ),
                "PickCube": (
                    f"The robot must pick up a cube and place it at the goal ({args.env_id}).\n"
                    "Reward components: w_reach (TCP approach to cube), w_grasp (grasp success), "
                    "w_place (cube toward goal, gated by grasp), "
                    "w_static (robot static with object placed), w_success (success bonus).\n\n"
                    "日本語補足: ロボットがキューブを掴んで目標位置に運ぶタスク。"
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
                "PegInsertionSide": (
                    "The robot must grasp a peg and insert it into a hole from the side (PegInsertionSide).\n"
                    "This is a multi-stage task: reach the peg, grasp it, align with the hole, then insert.\n"
                    "Reward components: w_reach (gripper approach to peg tail), "
                    "w_grasp (binary grasp success), "
                    "w_pre_insertion (peg-hole yz alignment, gated by grasp), "
                    "w_insertion (peg head into hole, gated by grasp AND pre-insertion alignment), "
                    "w_success (success bonus).\n\n"
                    "日本語補足: ペグを掴んで横方向から穴に挿入するタスク。"
                    "reach→grasp→alignment→insertionの段階的な報酬構造。"
                    "回答の末尾に日本語での簡潔な要約も追加してください。"
                ),
                "PushT": (
                    "The robot must push a T-shaped block to match the goal position and rotation (PushT).\n"
                    "This is a 2D pushing task requiring both position and rotation alignment.\n"
                    "Reward components: w_rotation (cos similarity of tee vs goal z-rotation, squared), "
                    "w_position (tee-to-goal xy distance, tanh-shaped and squared), "
                    "w_tcp_guide (encourage TCP to stay near the tee block), "
                    "w_success (success bonus).\n\n"
                    "日本語補足: T字ブロックを目標位置・回転に合わせるタスク。"
                    "位置と回転の両方を合わせる必要がある。"
                    "回答の末尾に日本語での簡潔な要約も追加してください。"
                ),
                "AnymalC": (
                    "A quadruped robot (AnymalC) must walk to a goal position (AnymalC-Reach).\n"
                    "The robot must maintain balance while locomoting toward the target.\n"
                    "Reward components: w_reach (robot-to-goal distance, tanh-shaped), "
                    "w_vel_z_penalty (penalize vertical velocity oscillation), "
                    "w_ang_vel_penalty (penalize angular velocity in xy), "
                    "w_contact_penalty (penalize undesired knee/body contacts with ground), "
                    "w_qpos_penalty (penalize deviation from default standing pose).\n"
                    "Note: reward has a base of +1.0 per step; fails (falls) give 0.\n\n"
                    "日本語補足: 四脚ロボットが目標位置まで歩行するタスク。"
                    "バランスを保ちながら移動する必要がある。"
                    "回答の末尾に日本語での簡潔な要約も追加してください。"
                ),
                "UnitreeG1PlaceAppleInBowl": (
                    "A humanoid robot (UnitreeG1) must pick up an apple and place it into a bowl "
                    "(UnitreeG1PlaceAppleInBowl).\n"
                    "Multi-stage: reach apple, grasp it, carry to above the bowl, then release.\n"
                    "Reward components: w_reach (TCP-to-apple distance), "
                    "w_grasp (binary grasp success), "
                    "w_place (apple-to-bowl distance with +0.15m z-offset, gated by grasp), "
                    "w_above_bowl (binary bonus when apple is within 0.025m of above-bowl target), "
                    "w_release (encourage opening hand, gated by above_bowl), "
                    "w_success (success bonus).\n"
                    "Note: the bowl target has a +0.15m z-offset to encourage bringing the apple "
                    "above the bowl before releasing.\n\n"
                    "日本語補足: ヒューマノイドロボットがリンゴを掴んでボウルに入れるタスク。"
                    "reach→grasp→carry→above_bowl→releaseの段階的な報酬構造。"
                    "回答の末尾に日本語での簡潔な要約も追加してください。"
                ),
                "PickCubePandaAllegro": (
                    "A Panda arm with Allegro dexterous hand must pick up a box and place it at "
                    "the goal position (PickCubePandaAllegro-v2).\n"
                    "The hand uses coupled control: 2D action (finger group + thumb open/close) "
                    "mapped to 16D joint positions. Arm uses 6D EE delta pose.\n"
                    "Reward components: w_reach (TCP approach to box), w_grasp (>=2 fingertips in contact), "
                    "w_place (box toward goal, gated by grasp), "
                    "w_static (arm static with object placed), w_success (success bonus).\n\n"
                    "日本語補足: Pandaアーム+Allegro多指ハンドでボックスを掴んで目標位置に運ぶタスク。"
                    "指は2自由度（親指・対抗指の開閉）で制御。"
                    "回答の末尾に日本語での簡潔な要約も追加してください。"
                ),
            }

            # Get reward function source code for LLM context
            # Iteration 0: Use minimal sparse reward example to avoid bias
            # Iteration 1+: Use best candidate from previous iteration
            if outer_iter == 0:
                # Sparse reward baseline: only success bonus
                # LLM must design dense reward shaping from scratch
                reward_fn_source = '''def compute_reward_sparse_example(info, base):
    """
    Sparse reward baseline: only provides reward at goal achievement.

    Your task: Design dense reward shaping to guide learning efficiently.
    Consider:
    - Distance-based rewards (reach, approach, proximity)
    - Multi-stage rewards (reach → manipulate → succeed)
    - Normalization (torch.exp, torch.tanh) with temperature parameters
    - Gating (reward stages only when prerequisites are met)
    """
    import torch

    # Sparse reward: only at success
    reward = torch.zeros(info["success"].shape[0], device=base.device, dtype=torch.float32)
    reward[info["success"]] = 1.0

    return reward
'''
            else:
                # Use previous iteration's best candidate as reference
                # (will be set in training_summary later)
                _reward_method_map = {
                    "PickCube": "_compute_pick_cube",
                    "PushCube": "_compute_push_cube",
                    "OpenCabinetDoor": "_compute_open_cabinet",
                    "OpenCabinetDrawer": "_compute_open_cabinet",
                    "UnitreeG1PlaceAppleInBowl": "_compute_unitree_place_apple",
                    "AnymalC": "_compute_anymalc_reach",
                    "PegInsertionSide": "_compute_peg_insertion",
                    "PushT": "_compute_push_t",
                    "PickCubePandaAllegro": "_compute_pick_cube_allegro",
                }
                try:
                    method_name = _reward_method_map[task_id]
                    reward_fn_source = inspect.getsource(
                        getattr(RewardWrapperDynamic, method_name)
                    )
                except Exception as e:
                    print(f"[Warning] Could not get reward_fn_source: {e}")
                    reward_fn_source = "N/A"

            # State access documentation for LLM (Eureka full replacement mode)
            _state_access_docs = {
                "PushCube": """
Available state attributes (base = env.unwrapped):
- base.obj.pose.p: Cube position, torch.Tensor (batch_size, 3)
- base.agent.tcp.pose.p: End effector position, torch.Tensor (batch_size, 3)
- base.goal_region.pose.p: Goal position, torch.Tensor (batch_size, 3)
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, n_joints)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool

Success condition (from environment):
- Cube XY distance to goal < 0.1m (goal_radius=0.1) AND cube Z < 0.025m (on table).
- Key constants: cube_half_size=0.02m, goal_radius=0.1m.

Reward design guidelines:
- Total reward MUST be in [0, 4] range. On success, override reward to exactly 4.
- ALWAYS use full 3D Euclidean distances (not just XY) for reach/approach components.

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
                "PickCube": """
Available state attributes (base = env.unwrapped):
- base.cube.pose.p: Object position, torch.Tensor (batch_size, 3)
- base.agent.tcp_pose.p: End effector position, torch.Tensor (batch_size, 3)
- base.goal_site.pose.p: Goal position, torch.Tensor (batch_size, 3)
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, n_joints)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool
- info["is_grasped"]: torch.Tensor (batch_size,) bool
- info["is_obj_placed"]: torch.Tensor (batch_size,) bool

Success condition (from environment):
- Cube within 0.025m (3D Euclidean) of goal AND robot is static (joint velocities <= 0.2 rad/s).
- Key constants: cube_half_size=0.02m, goal_thresh=0.025m.

Reward design guidelines:
- Total reward MUST be in [0, 4] range. On success, override reward to exactly 4.
- ALWAYS use full 3D Euclidean distances for reach/approach components.
- Use info["is_grasped"] to gate place rewards (only reward placing when grasped).

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
                "OpenCabinetDoor": """
Available state attributes (base = env.unwrapped):
- base.agent.tcp.pose.p: End effector position, torch.Tensor (batch_size, 3)
- base.handle_link.pose.p: Door handle link position, torch.Tensor (batch_size, 3)
- base.handle_link.joint.qpos: Door joint position (angle), torch.Tensor (batch_size, 1)
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, n_joints)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool
- info["handle_link_pos"]: torch.Tensor (batch_size, 3) handle center position (computed in evaluate())
- info["open_enough"]: torch.Tensor (batch_size,) bool (door opened beyond threshold)

Success condition (from environment):
- Door joint opened >= 75% of max range (min_open_frac=0.75) AND door is static (angular velocity <= 1 rad/s, linear velocity <= 0.1 m/s).

Reward design guidelines:
- Total reward MUST be in [0, 4] range. On success, override reward to exactly 4.
- ALWAYS use full 3D Euclidean distances for reaching the handle.
- Reward door opening progress (joint position increase) after reaching the handle.

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
                "OpenCabinetDrawer": """
Available state attributes (base = env.unwrapped):
- base.agent.tcp.pose.p: End effector position, torch.Tensor (batch_size, 3)
- base.handle_link.pose.p: Drawer handle link position, torch.Tensor (batch_size, 3)
- base.handle_link.joint.qpos: Drawer joint position (distance), torch.Tensor (batch_size, 1)
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, n_joints)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool
- info["handle_link_pos"]: torch.Tensor (batch_size, 3) handle center position (computed in evaluate())
- info["open_enough"]: torch.Tensor (batch_size,) bool (drawer opened beyond threshold)

Success condition (from environment):
- Drawer joint opened >= 75% of max range (min_open_frac=0.75) AND drawer is static (angular velocity <= 1 rad/s, linear velocity <= 0.1 m/s).

Reward design guidelines:
- Total reward MUST be in [0, 4] range. On success, override reward to exactly 4.
- ALWAYS use full 3D Euclidean distances for reaching the handle.
- Reward drawer opening progress (joint position increase) after reaching the handle.

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
                "PegInsertionSide": """
Available state attributes (base = env.unwrapped):
- base.peg.pose.p: Peg position, torch.Tensor (batch_size, 3)
- base.agent.tcp.pose.p: End effector position, torch.Tensor (batch_size, 3)
- base.box_hole_pose.p: Hole position, torch.Tensor (batch_size, 3)
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, n_joints)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool (peg fully inserted)
- info["peg_head_pos_at_hole"]: torch.Tensor (batch_size, 3) computed peg head position

Success condition (from environment):
- Peg head inserted >= 15mm into hole (X-axis in hole frame) AND peg head Y,Z within hole radius.
- Key constants: peg radius 0.015-0.025m (randomized), hole clearance 0.003m, peg half-length 0.085-0.125m.

Reward design guidelines:
- Total reward MUST be in [0, 4] range. On success, override reward to exactly 4.
- ALWAYS use full 3D Euclidean distances for reach and alignment components.
- Multi-stage: reach peg -> grasp -> align with hole -> insert.

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
                "PushT": """
Available state attributes (base = env.unwrapped):
- base.tee.pose.p: T-block position, torch.Tensor (batch_size, 3)
- base.tee.pose.q: T-block rotation (quaternion), torch.Tensor (batch_size, 4)
- base.agent.tcp.pose.p: End effector position, torch.Tensor (batch_size, 3)
- base.goal_tee.pose.p: Goal T-block position, torch.Tensor (batch_size, 3)
- base.goal_z_rot: Goal z-axis rotation (Euler angle), torch.Tensor (batch_size,)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool

Success condition (from environment):
- T-block overlaps >= 90% of goal T region area (2D intersection). Requires BOTH position AND rotation alignment.
- intersection_thresh = 0.90.

Reward design guidelines:
- Total reward MUST be in [0, 4] range. On success, override reward to exactly 4.
- ALWAYS use full 3D Euclidean distances for reach/approach components.
- Reward both position proximity and rotation alignment toward goal.

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
                "AnymalC": """
Available state attributes (base = env.unwrapped):
- base.agent.robot.pose.p: Robot position, torch.Tensor (batch_size, 3)
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, n_joints)
- base.goal.pose.p: Goal sphere position, torch.Tensor (batch_size, 3)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool (reached goal without falling)
- info["fail"]: torch.Tensor (batch_size,) bool (robot fell)
- info["robot_to_goal_dist"]: torch.Tensor (batch_size,) float (xy distance to goal)
- info["reached_goal"]: torch.Tensor (batch_size,) bool (dist < 0.35m)
- info["is_fallen"]: torch.Tensor (batch_size,) bool

Success condition (from environment):
- Robot XY distance to goal < 0.35m AND robot has NOT fallen.
- Goal is typically 2.0-3.0m away from the start position.

Reward design guidelines:
- Total reward MUST be in [0, 4] range. On success, override reward to exactly 4.
- Penalize falling (info["is_fallen"]).
- Reward progress toward goal (reducing XY distance).

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
                "UnitreeG1PlaceAppleInBowl": """
Available state attributes (base = env.unwrapped):
- base.apple.pose.p: Apple position, torch.Tensor (batch_size, 3)
- base.agent.right_tcp.pose.p: Right hand end effector position, torch.Tensor (batch_size, 3)
- base.bowl.pose.p: Bowl position, torch.Tensor (batch_size, 3)
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, n_joints)
- base.agent.right_hand_dist_to_open_grasp(): Distance to open grasp, torch.Tensor (batch_size,)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool (apple in bowl AND hand outside)
- info["is_grasped"]: torch.Tensor (batch_size,) bool
- info["hand_outside_bowl"]: torch.Tensor (batch_size,) bool (hand z > bowl z + 0.125m)

Success condition (from environment):
- Apple within 0.05m (3D Euclidean) of bowl position AND right hand Z > bowl Z + 0.125m (hand retracted above bowl).

Reward design guidelines:
- Total reward MUST be in [0, 4] range. On success, override reward to exactly 4.
- ALWAYS use full 3D Euclidean distances for reach/approach components.
- Use info["is_grasped"] to gate place rewards.
- After placing, reward hand retraction (moving hand above bowl).

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
                "PickCubePandaAllegro": """
Available state attributes (base = env.unwrapped):
- base.cube.pose.p: Box position, torch.Tensor (batch_size, 3)
- base.agent.tcp_pose.p: End effector (TCP) position, torch.Tensor (batch_size, 3)
- base.goal_site.pose.p: Goal position, torch.Tensor (batch_size, 3)
- base.agent.robot.get_qvel(): Joint velocities, torch.Tensor (batch_size, 23) (7 arm + 16 hand)
- base.agent.tip_links: List of 4 fingertip links [thumb, index, middle, ring]
- base.agent.tip_links[i].pose.p: Fingertip position, torch.Tensor (batch_size, 3)
- base.agent.palm_link.pose.p: Palm position, torch.Tensor (batch_size, 3)

info dict keys:
- info["success"]: torch.Tensor (batch_size,) bool
- info["is_grasped"]: torch.Tensor (batch_size,) bool (>=2 fingertips in contact with box)
- info["is_obj_placed"]: torch.Tensor (batch_size,) bool (box within 0.025m of goal)
- info["is_robot_static"]: torch.Tensor (batch_size,) bool

Action space: 8D (6 arm EE delta + 2 hand scalars via CoupledAllegroActionWrapper).
  action[0:6]: arm end-effector delta pose (position + rotation)
  action[6]: finger group scalar [-1=open, 1=closed] (controls index/middle/ring together)
  action[7]: thumb scalar [-1=open, 1=closed]

Success condition (from environment):
- Box within 0.025m (3D Euclidean) of goal AND robot arm is static (joint velocities <= 0.2 rad/s).
- Key constants: cube_half_size=[0.02, 0.04, 0.02], goal_thresh=0.025m.

Reward design guidelines:
- Total reward MUST be in [0, 5] range. On success, override reward to exactly 5.
- ALWAYS use full 3D Euclidean distances for reach/approach components.
- Use info["is_grasped"] to gate place rewards (only reward placing when grasped).
- Only use arm joint velocities (first 7) for static penalty, not hand joints.

Required function signature:
def compute_reward(info: dict, base) -> torch.Tensor:
    # Return: torch.Tensor, shape (batch_size,)
    pass
""",
            }

            # Base training_summary for candidate generation
            if outer_iter == 0:
                # Initial iteration: minimal context
                training_summary = {
                    "current_iteration": outer_iter,
                    "total_iterations": args.num_outer_iters,
                    "llm_task_description": _llm_task_descs.get(task_id, ""),
                    "reward_fn_source": reward_fn_source,
                    "state_access_docs": _state_access_docs.get(task_id, ""),
                    "eureka_full_replacement": True,
                    "num_reward_candidates": 1,  # Each LLM call produces exactly 1 candidate
                }
            else:
                # Iteration 1+: Include statistics from previous iteration
                prev_iter = outer_loop_history[-1]
                prev_best = prev_iter["best_candidate"]

                training_summary = {
                    "current_iteration": outer_iter,
                    "total_iterations": args.num_outer_iters,
                    "total_timesteps": global_step_offset,
                    "llm_task_description": _llm_task_descs.get(task_id, ""),
                    # Use prev best code → last known good → fallback method source
                    "reward_fn_source": prev_best.get("code") or last_good_code or reward_fn_source,
                    "state_access_docs": _state_access_docs.get(task_id, ""),
                    "eureka_full_replacement": True,
                    "num_reward_candidates": 1,  # Each LLM call produces exactly 1 candidate
                    # Statistics from previous iteration
                    "avg_return": prev_best["eval_metrics"].get("return", 0.0),
                    "success_rate": prev_best["fitness"],
                    "success_at_end": prev_best["fitness"],
                    "success_once": prev_best["eval_metrics"].get("success_once", 0.0),
                    "episode_len": prev_best["eval_metrics"].get("episode_len", 0.0),
                    "learning_curve": prev_best["learning_curve"],
                    "num_episodes": int(prev_best["eval_metrics"].get("num_episodes", 0)),
                }

                # Add Reward Reflection from previous iteration (if available)
                if "reflection_history" in prev_iter:
                    training_summary["reflection_history"] = prev_iter["reflection_history"]

                # Add structured history for LLM (past changes and observed outcomes)
                history_summary = []
                vlm_comments = []
                for hist in outer_loop_history:
                    hist_best = hist.get("best_candidate", {})
                    hist_eval = hist_best.get("eval_metrics", {})
                    rationale = hist_best.get("rationale", "No rationale provided")
                    history_summary.append(
                        {
                            "iteration": hist.get("outer_iter", 0),
                            "changes": rationale,
                            "rationale": rationale,
                            "result": {
                                "success_rate": hist_best.get("fitness", 0.0),
                                "avg_return": hist_eval.get("return", 0.0),
                                "success_once": hist_eval.get("success_once", 0.0),
                                "episode_len": hist_eval.get("episode_len", 0.0),
                            },
                        }
                    )

                    hist_vlm_comment = hist_best.get("vlm_comment")
                    if not hist_vlm_comment and "reflection_history" in hist:
                        hist_vlm_comment = hist["reflection_history"].get("vlm_feedback")
                    if (
                        isinstance(hist_vlm_comment, str)
                        and hist_vlm_comment.strip()
                        and hist_vlm_comment.strip() != "N/A"
                    ):
                        vlm_comments.append(
                            f"Iter {hist.get('outer_iter', 0) + 1}: {hist_vlm_comment.strip()}"
                        )

                training_summary["history_summary"] = history_summary
                training_summary["vlm_comments"] = vlm_comments[-5:]

                # Add performance trend across iterations
                if len(outer_loop_history) > 0:
                    training_summary["performance_trend"] = [
                        {
                            "iteration": h["outer_iter"],
                            "fitness": h["best_candidate"]["fitness"],
                            "avg_return": h["best_candidate"]["eval_metrics"].get("return", 0.0),
                            "success_rate": h["best_candidate"]["fitness"],
                            "success_at_end": h["best_candidate"]["fitness"],
                            "success_once": h["best_candidate"]["eval_metrics"].get("success_once", 0.0),
                            "episode_len": h["best_candidate"]["eval_metrics"].get("episode_len", 0.0),
                            "learning_curve": h["best_candidate"]["learning_curve"],
                        }
                        for h in outer_loop_history
                    ]

            # Generate LLM candidates (remaining slots after elite)
            MAX_COMPILE_RETRIES = 5
            elite_count = sum(1 for c in candidates if c.get("is_elite", False))

            for k in range(llm_candidate_count):
                cand_id = elite_count + k  # Fixed ID per slot (not len(candidates))
                print(f"  LLM Candidate {cand_id+1}/{args.num_reward_candidates} generation...")

                compiled = False
                error_context = None

                for compile_attempt in range(MAX_COMPILE_RETRIES):
                    if compile_attempt > 0:
                        print(f"    Compile retry {compile_attempt}/{MAX_COMPILE_RETRIES-1}...")

                    # Build summary (with error context if retrying)
                    gen_summary = training_summary
                    if error_context is not None:
                        gen_summary = {**training_summary, "previous_code_error": error_context}

                    # Seed fixing for Iteration 0, first attempt only (reproducibility)
                    llm_seed = args.weight_seed + k if outer_iter == 0 and compile_attempt == 0 else None

                    try:
                        suggestions = llm.suggest_parameters(gen_summary, seed=llm_seed)
                    except (ValueError, SyntaxError) as e:
                        # suggest_parameters internally validates/compiles code and may raise
                        # on syntax errors. Catch here so the retry loop can handle it.
                        print(f"    ✗ suggest_parameters failed (attempt {compile_attempt+1}): {e}")
                        # Save debug HTML even on failure
                        query_info = llm.get_last_query_info() if hasattr(llm, 'get_last_query_info') else None
                        llm_prompt = query_info.get("prompt", "(no prompt)") if query_info else "(no query info)"
                        llm_response = query_info.get("response_text", "(no response)") if query_info else "(no query info)"
                        suffix = f"_attempt{compile_attempt}" if compile_attempt > 0 else ""
                        save_llm_debug_html(
                            iteration=outer_iter,
                            prompt=llm_prompt,
                            response_text=llm_response,
                            suggestions=None,
                            summary_for_llm=gen_summary,
                            save_path=debug_dir / f"iter_{outer_iter+1:02d}_cand_{cand_id}{suffix}_llm.html",
                        )
                        error_context = {
                            "code": "N/A",
                            "error": str(e),
                            "instruction": (
                                "前回のコードでエラーが発生しました。\n"
                                "エラーを修正した新しいコードを生成してください。\n"
                                "Pythonとして正しい構文の完全な関数を返してください。\n\n"
                                f"エラー内容: {e}"
                            )
                        }
                        continue

                    # Save LLM debug HTML (unique per attempt)
                    query_info = llm.get_last_query_info() if hasattr(llm, 'get_last_query_info') else None
                    llm_prompt = query_info.get("prompt", "(no prompt)") if query_info else "(no query info)"
                    llm_response = query_info.get("response_text", "(no response)") if query_info else "(no query info)"
                    suffix = f"_attempt{compile_attempt}" if compile_attempt > 0 else ""
                    save_llm_debug_html(
                        iteration=outer_iter,
                        prompt=llm_prompt,
                        response_text=llm_response,
                        suggestions=suggestions,
                        summary_for_llm=gen_summary,
                        save_path=debug_dir / f"iter_{outer_iter+1:02d}_cand_{cand_id}{suffix}_llm.html",
                    )

                    if suggestions and suggestions.get("type") == "function_code":
                        custom_code = suggestions["custom_code"]
                        rationale = suggestions.get("rationale", "No rationale")

                        test_fn, compile_error = RewardWrapperDynamic._compile_custom_function_with_error(custom_code)

                        if test_fn is not None:
                            candidates.append({
                                "code": custom_code,
                                "rationale": rationale,
                                "id": cand_id,
                                "is_elite": False,
                            })
                            print(f"    ✓ Candidate {cand_id+1} compiled (attempt {compile_attempt+1})")
                            compiled = True
                            break
                        else:
                            print(f"    ✗ Compile failed (attempt {compile_attempt+1}): {compile_error[:200]}")
                            error_context = {
                                "code": custom_code,
                                "error": compile_error,
                                "instruction": (
                                    "前回のコードでコンパイルエラーが発生しました。\n"
                                    "エラーを修正した新しいコードを生成してください。\n\n"
                                    "よくあるエラー:\n"
                                    "1. 関数名が 'compute_reward' でない\n"
                                    "2. インデントエラー\n"
                                    "3. torch/npのインポート忘れ\n"
                                    "4. 戻り値がtorch.Tensorでない\n"
                                    "5. batch_size次元の処理ミス"
                                )
                            }
                    else:
                        sug_type = suggestions.get("type", "N/A") if suggestions else "None"
                        print(f"    ✗ Wrong response type (attempt {compile_attempt+1}): {sug_type}")
                        error_context = {
                            "code": "N/A",
                            "error": f"LLM did not return function_code type (got: {sug_type})",
                            "instruction": (
                                "function_code形式で報酬関数を返してください。\n"
                                "type: 'function_code', custom_code: 'def compute_reward(info, base): ...'"
                            )
                        }

                if not compiled:
                    print(f"    ✗ Candidate slot {cand_id+1} exhausted {MAX_COMPILE_RETRIES} attempts")

            if len(candidates) == 0:
                print("\n[ERROR] No valid candidates generated. Terminating experiment.")
                sys.exit(1)

            final_elite_count = sum(1 for c in candidates if c.get("is_elite", False))
            llm_count = len(candidates) - final_elite_count
            print(f"\n  Valid candidates: {len(candidates)}/{args.num_reward_candidates} (elite={final_elite_count}, llm={llm_count})")

        else:
            # Fallback: use default weights (params-only mode)
            # Preserve elite candidates if already added
            if not candidates:
                print("[INFO] LLM disabled or function_code=False, using default weights")
                candidates.append({
                    "code": None,
                    "rationale": "Default weights (no custom function)",
                    "id": 0,
                    "is_elite": False,
                })
            else:
                print(f"[INFO] LLM disabled, proceeding with {len(candidates)} elite candidate(s)")

        # ========== STEP 2: Train and evaluate each candidate ==========
        gpu_list = [int(g) for g in args.gpus.split(",")] if args.gpus else []

        if len(gpu_list) > 1:
            # --- Parallel K-candidate training across GPUs ---
            print(f"\n[Parallel Mode] Training {len(candidates)} candidates across {len(gpu_list)} GPUs: {gpu_list}")
            candidate_results = _train_candidates_parallel(
                args=args,
                candidates=candidates,
                outer_iter=outer_iter,
                run_dir=run_dir,
                global_step_offset=global_step_offset,
                gpu_list=gpu_list,
                llm=llm,
                training_summary=training_summary,
                save_llm_debug_html=save_llm_debug_html,
                debug_dir=debug_dir,
            )
            # Advance global_step_offset by one training run worth of steps
            global_step_offset += args.total_timesteps_per_iter
        else:
            # --- Sequential K-candidate training (original code path) ---
            candidate_results = []
            MAX_RUNTIME_RETRIES = 2

            for cand in candidates:
                print(f"\n{'='*50}")
                print(f"[Candidate {cand['id']+1}] Training PPO")
                print(f"{'='*50}")

                current_code = cand["code"]
                current_rationale = cand["rationale"]
                training_succeeded = False

                for attempt in range(MAX_RUNTIME_RETRIES + 1):
                    if attempt > 0:
                        print(f"\n  [Runtime Retry {attempt}/{MAX_RUNTIME_RETRIES}] Retrying with LLM-fixed code")

                    print(f"Rationale: {current_rationale}")

                    try:
                        cand_run_dir = f"{run_dir}/cand_{cand['id']}"
                        if attempt > 0:
                            cand_run_dir = f"{run_dir}/cand_{cand['id']}_retry{attempt}"

                        result = run_ppo_training(
                            args=args,
                            weights=None,
                            outer_iter=outer_iter,
                            run_dir=cand_run_dir,
                            logger=logger,
                            device=device,
                            global_step_offset=global_step_offset,
                            custom_code=current_code,
                        )

                        # Training succeeded
                        global_step_offset = result["final_global_step"]

                        eval_metrics = result["eval_metrics"]
                        fitness = eval_metrics.get("success_at_end", 0.0)
                        # Use peak success_once across all eval checkpoints (not just final)
                        # so candidates that briefly succeed during training are preferred
                        # over candidates that never succeed but have higher final return.
                        _lc = result.get("learning_curve", [])
                        fitness_success_once = max(
                            (lc.get("success_once", 0.0) for lc in _lc),
                            default=eval_metrics.get("success_once", 0.0),
                        )
                        fitness_return = eval_metrics.get("return", float("-inf"))

                        step_rewards = result.get("step_rewards", [])
                        if step_rewards:
                            step_rewards_tensor = torch.stack(step_rewards)
                            reward_stats = {
                                "mean": step_rewards_tensor.mean().item(),
                                "std": step_rewards_tensor.std().item(),
                                "min": step_rewards_tensor.min().item(),
                                "max": step_rewards_tensor.max().item(),
                            }
                        else:
                            reward_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

                        candidate_results.append({
                            "candidate_id": cand["id"],
                            "code": current_code,
                            "rationale": current_rationale,
                            "is_elite": cand.get("is_elite", False),
                            "fitness": fitness,
                            "fitness_success_at_end": fitness,
                            "fitness_success_once": fitness_success_once,
                            "fitness_return": fitness_return,
                            "eval_metrics": eval_metrics,
                            "learning_curve": result["learning_curve"],
                            "reward_statistics": reward_stats,
                            "eval_video_dir": result["eval_video_dir"],
                        })

                        print(f"\n[Candidate {cand['id']+1}] Results:")
                        print(f"  Fitness (success_at_end): {fitness:.4f}")
                        print(f"  Success once: {eval_metrics.get('success_once', 0.0):.4f}")
                        print(f"  Avg return: {eval_metrics.get('return', 0.0):.4f}")
                        print(f"  Reward stats: mean={reward_stats['mean']:.4f}, std={reward_stats['std']:.4f}")

                        training_succeeded = True
                        break

                    except Exception as e:
                        error_tb = traceback.format_exc()
                        print(f"\n[Candidate {cand['id']+1}] Runtime error (attempt {attempt+1}/{MAX_RUNTIME_RETRIES+1}):")
                        print(f"  {type(e).__name__}: {e}")
                        print(f"  Traceback (last 15 lines):")
                        for line in error_tb.strip().split('\n')[-15:]:
                            print(f"    {line}")

                        # Free GPU resources from failed training
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        if attempt < MAX_RUNTIME_RETRIES and llm is not None:
                            print(f"  Requesting LLM to fix the runtime error...")
                            fix_summary = {
                                **training_summary,
                                "previous_code_error": {
                                    "code": current_code,
                                    "error": f"{type(e).__name__}: {e}\n\n{error_tb}",
                                    "instruction": (
                                        "前回のコードでランタイムエラーが発生しました。\n"
                                        "エラーを修正した新しいコードを生成してください。\n\n"
                                        "よくあるランタイムエラー:\n"
                                        "1. base.device が存在しない → base.obj.pose.p.device を使う\n"
                                        "2. テンソルのshape不一致（(B,3)に対してスカラー操作等）\n"
                                        "3. 存在しない属性へのアクセス（State Access Docsを参照）\n"
                                        "4. torch演算のdevice不一致（CPU/CUDA混在）\n"
                                        "5. info dictのキーが存在しない\n"
                                        "6. hasattr/setattr on batched env objects\n"
                                        "7. 型アノテーションで __import__ を使用 → 'torch.Tensor' を使う\n\n"
                                        "修正後のコードをfunction_code形式で返してください。"
                                    )
                                }
                            }
                            suggestions = llm.suggest_parameters(fix_summary)

                            # Save retry debug HTML
                            if save_llm_debug_html is not None:
                                query_info = llm.get_last_query_info() if hasattr(llm, 'get_last_query_info') else None
                                llm_prompt = query_info.get("prompt", "(no prompt)") if query_info else "(no query info)"
                                llm_response = query_info.get("response_text", "(no response)") if query_info else "(no query info)"
                                save_llm_debug_html(
                                    iteration=outer_iter,
                                    prompt=llm_prompt,
                                    response_text=llm_response,
                                    suggestions=suggestions,
                                    summary_for_llm=fix_summary,
                                    save_path=debug_dir / f"iter_{outer_iter+1:02d}_cand_{cand['id']}_runtime_retry{attempt}_llm.html",
                                )

                            if suggestions and suggestions.get("type") == "function_code":
                                new_code = suggestions["custom_code"]
                                test_fn, compile_error = RewardWrapperDynamic._compile_custom_function_with_error(new_code)
                                if test_fn is not None:
                                    current_code = new_code
                                    current_rationale = suggestions.get("rationale", f"Runtime error fix (attempt {attempt+1})")
                                    print(f"  ✓ LLM generated fixed code, retrying training...")
                                    continue
                                else:
                                    print(f"  ✗ LLM fix failed compilation: {compile_error}")
                            else:
                                sug_type = suggestions.get("type", "N/A") if suggestions else "N/A"
                                print(f"  ✗ LLM returned wrong type: {sug_type}")

                        # No more retries or LLM fix failed
                        print(f"  ✗ Candidate {cand['id']+1} skipped (runtime error)")
                        break

                if not training_succeeded:
                    print(f"[Candidate {cand['id']+1}] FAILED after {attempt+1} attempt(s)")

        # ========== STEP 3: Select best candidate ==========
        if len(candidate_results) == 0:
            print(f"\n[ERROR] All candidates failed for iteration {outer_iter+1}. Skipping to next iteration.")
            # Record empty iteration
            outer_loop_history.append({
                "outer_iter": outer_iter,
                "best_candidate": {"candidate_id": -1, "fitness": 0.0, "eval_metrics": {}, "code": None,
                                   "rationale": "All candidates failed", "learning_curve": []},
                "all_candidates": [],
                "num_valid_candidates": 0,
            })
            # Incremental save even on failure
            history_path = f"runs/{run_dir}/outer_loop_history.json"
            with open(history_path, "w") as f:
                json.dump(outer_loop_history, f, indent=2, default=str)
            continue

        # Primary fitness is success_at_end (paper-style), but this often ties at 0.0
        # in early experiments. Break ties with success_once, then avg return.
        best = max(
            candidate_results,
            key=lambda x: (
                x.get("fitness_success_at_end", x["fitness"]),
                x.get("fitness_success_once", x["eval_metrics"].get("success_once", 0.0)),
                x.get("fitness_return", x["eval_metrics"].get("return", float("-inf"))),
            ),
        )
        fitness_end_values = [c.get("fitness_success_at_end", c["fitness"]) for c in candidate_results]
        if len(fitness_end_values) > 1 and max(fitness_end_values) == min(fitness_end_values):
            print("  [Selection] success_at_end tied across candidates; tie-break used: peak_success_once -> return")
        if best.get("code") is not None:
            last_good_code = best["code"]
        print(f"\n{'='*60}")
        print(
            f"[BEST] Candidate {best['candidate_id']+1} selected "
            f"(fitness_end={best.get('fitness_success_at_end', best['fitness']):.4f}, "
            f"peak_success_once={best.get('fitness_success_once', best['eval_metrics'].get('success_once', 0.0)):.4f}, "
            f"return={best.get('fitness_return', best['eval_metrics'].get('return', 0.0)):.2f})"
        )
        print(f"{'='*60}")

        # Save candidate comparison summary
        comparison_html = f"""<h2>Iteration {outer_iter+1} - Candidate Comparison</h2>
<table border="1" style="border-collapse: collapse; width: 100%;">
<tr><th>Candidate</th><th>Fitness</th><th>Success@End</th><th>Success Once</th><th>Avg Return</th><th>Reward Stats</th><th>Rationale</th></tr>
"""
        for cand in sorted(
            candidate_results,
            key=lambda x: (
                x.get("fitness_success_at_end", x["fitness"]),
                x.get("fitness_success_once", x["eval_metrics"].get("success_once", 0.0)),
                x.get("fitness_return", x["eval_metrics"].get("return", float("-inf"))),
            ),
            reverse=True,
        ):
            cid = cand["candidate_id"] + 1
            fit = cand["fitness"]
            s_end = cand["eval_metrics"].get("success_at_end", 0.0)
            s_once = cand["eval_metrics"].get("success_once", 0.0)
            avg_ret = cand["eval_metrics"].get("return", 0.0)
            r_stats = cand["reward_statistics"]
            rationale = cand["rationale"]
            best_mark = "⭐ BEST" if cand["candidate_id"] == best["candidate_id"] else ""
            comparison_html += f"""<tr>
<td>{cid} {best_mark}</td>
<td>{fit:.4f}</td>
<td>{s_end:.4f}</td>
<td>{s_once:.4f}</td>
<td>{avg_ret:.2f}</td>
<td>mean={r_stats['mean']:.3f}, std={r_stats['std']:.3f}</td>
<td>{rationale}</td>
</tr>
"""
        comparison_html += "</table>"

        comparison_path = debug_dir / f"iter_{outer_iter+1:02d}_candidates_comparison.html"
        with open(comparison_path, "w") as f:
            f.write(f"<html><body>{comparison_html}</body></html>")
        print(f"  Saved candidate comparison: {comparison_path}")

        # Log best candidate metrics (global_step_offset already updated in training loop)
        if logger is not None:
            logger.add_scalar("outer_iter/best_fitness", best["fitness"], global_step_offset)
            logger.add_scalar("outer_iter/best_success_at_end", best["eval_metrics"].get("success_at_end", 0.0), global_step_offset)
            logger.add_scalar("outer_iter/best_avg_return", best["eval_metrics"].get("return", 0.0), global_step_offset)
            logger.add_scalar("outer_iter/iteration", outer_iter, global_step_offset)

        # ========== STEP 4: VLM analysis ==========
        vlm_comment = None
        if vlm is not None:
            print(f"\n[VLM] Analyzing best candidate's video (iteration {outer_iter+1})...")

            video_dir = Path(best["eval_video_dir"])
            video_files = sorted(video_dir.glob("*.mp4"))

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
                        "return": best["eval_metrics"].get("return", 0.0),
                        "success": best["eval_metrics"].get("success_at_end", best["fitness"]),
                        "success_at_end": best["fitness"],
                        "success_once": best["eval_metrics"].get("success_once", 0.0),
                        "length": args.num_eval_steps,
                    }

                    vlm_score, vlm_comment, _ = vlm.evaluate(frames, episode_info)
                    print(f"[VLM] Analysis:\n{vlm_comment}")
                    best["vlm_comment"] = vlm_comment
                    best["vlm_score"] = vlm_score

                    # Save VLM debug HTML
                    debug_dir = Path(f"runs/{run_dir}/debug_html")
                    vlm_html_path = debug_dir / f"iter_{outer_iter+1:02d}_vlm.html"
                    save_vlm_debug_html(
                        frames=frames,
                        prompt=build_vlm_prompt(args.env_id),
                        episode_info=episode_info,
                        vlm_score=vlm_score,
                        vlm_comment=vlm_comment,
                        save_path=vlm_html_path,
                        max_frames=args.vlm_max_frames,
                    )
                else:
                    print("[warn] No frames extracted from video")
            else:
                print("[warn] No eval videos found")

        # ========== STEP 5: Reward Reflection ==========
        reflection_summary = None
        if args.enable_reward_reflection:
            print(f"\n[Reflection] Preparing feedback for next iteration...")
            reflection_summary = {
                "best_candidate": best,
                "all_candidates": candidate_results,
                "reward_statistics": best["reward_statistics"],
                "vlm_feedback": best.get("vlm_comment", "N/A"),
            }
            print(f"  Reflection data prepared (best fitness={best['fitness']:.4f})")

        # ========== STEP 6: Record history ==========
        iter_record = {
            "outer_iter": outer_iter,
            "best_candidate": best,
            "all_candidates": candidate_results,
            "num_valid_candidates": len(candidates),
        }
        # Save reflection for next iteration
        if reflection_summary is not None:
            iter_record["reflection_history"] = reflection_summary
        outer_loop_history.append(iter_record)

        # Incremental save after each iteration (enables mid-run visualization)
        history_path = f"runs/{run_dir}/outer_loop_history.json"
        with open(history_path, "w") as f:
            json.dump(outer_loop_history, f, indent=2, default=str)

    # --- Save final results ---
    history_path = f"runs/{run_dir}/outer_loop_history.json"
    print(f"\nOuter loop history saved to {history_path}")

    # Save best candidate from final iteration
    if outer_loop_history:
        final_best = outer_loop_history[-1]["best_candidate"]
        final_best_path = f"runs/{run_dir}/final_best_candidate.json"
        with open(final_best_path, "w") as f:
            json.dump({
                "candidate_id": final_best["candidate_id"],
                "rationale": final_best["rationale"],
                "fitness": final_best["fitness"],
                "code": final_best["code"],
            }, f, indent=2)
        print(f"Final best candidate saved to {final_best_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("OUTER LOOP SUMMARY (Eureka Full Replacement)")
    print(f"{'='*60}")
    for record in outer_loop_history:
        i = record["outer_iter"]
        best = record["best_candidate"]
        s_end = best["eval_metrics"].get("success_at_end", 0.0)
        s_once = best["eval_metrics"].get("success_once", 0.0)
        ar = best["eval_metrics"].get("return", 0.0)
        fitness = best["fitness"]
        num_cands = record["num_valid_candidates"]
        print(f"  Iter {i+1}: fitness={fitness:.4f}, success_end={s_end:.4f}, "
              f"success_once={s_once:.4f}, return={ar:.4f}, "
              f"candidates={num_cands}, best_id={best['candidate_id']+1}")
    print(f"{'='*60}")

    logger.close()

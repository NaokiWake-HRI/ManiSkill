"""
PPO with Iterative VLM/LLM Reward Tuning for ManiSkill.

Based on ppo.py. Splits training into segments and uses VLM for failure
analysis + LLM for reward weight adjustment between segments.

Usage:
    python ppo_iterative.py --env_id PickCube-v1 --total_timesteps 10_000_000 --num_segments 10
"""

from collections import defaultdict
import json
import os
import random
import sys
import time
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

from reward_wrapper import RewardWrapper


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
    evaluate: bool = False
    """if toggled, only runs evaluation"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file"""

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
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
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
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

    # Iterative VLM/LLM arguments
    num_segments: int = 10
    """number of training segments for VLM/LLM updates"""
    vlm_model: str = "gpt-5.2"
    """VLM model for video analysis"""
    llm_model: str = "gpt-5.2"
    """LLM model for reward tuning"""
    vlm_max_frames: int = 8
    """max frames to send to VLM"""
    vlm_num_envs: int = 1
    """number of eval envs to show in VLM frames (crops tiled image)"""
    rl_project_path: str = "/home/nwake/codes/RL_project"
    """path to RL_project for VLM/LLM imports"""
    skip_vlm_llm: bool = False
    """skip VLM/LLM calls (for testing reward wrapper only)"""
    initial_weights_file: Optional[str] = None
    """path to JSON file with initial reward weights (for random baseline experiments)"""

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5)

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
    """Crop a tiled frame to show only the first num_show_envs environments.

    ManiSkill tiles envs in a grid with nrows=int(sqrt(num_total_envs)).
    This function extracts a sub-grid showing num_show_envs envs.
    """
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
Do NOT provide a numerical score - focus on qualitative analysis."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
    vlm_suffix = "novlm" if args.skip_vlm_llm else "vlm"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.exp_name}-{vlm_suffix}-{timestamp}"
    run_dir = f"iterative/{args.env_id}/{run_name}"

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # --- Environment setup ---
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        reward_mode="none",  # Disable built-in reward
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,
    )
    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **env_kwargs,
    )

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    # Load custom initial weights if provided
    initial_weights = None
    if args.initial_weights_file is not None:
        with open(args.initial_weights_file, "r") as f:
            initial_weights = json.load(f)
        print(f"[info] Loaded initial weights from {args.initial_weights_file}: {initial_weights}")

    # Add RewardWrapper (BEFORE RecordEpisode and ManiSkillVectorEnv)
    reward_wrapper_train = RewardWrapper(envs, env_id=args.env_id, weights=initial_weights)
    reward_wrapper_eval = RewardWrapper(eval_envs, env_id=args.env_id, weights=initial_weights)
    envs = reward_wrapper_train
    eval_envs = reward_wrapper_eval

    # Video recording for eval
    eval_output_dir = f"runs/{run_dir}/videos"
    if args.capture_video:
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=f"runs/{run_dir}/train_videos",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=30,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.evaluate,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=30,
        )

    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

    # --- Logging ---
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_envs,
                env_id=args.env_id,
                env_horizon=max_episode_steps,
                partial_reset=args.partial_reset,
            )
            config["initial_reward_weights"] = reward_wrapper_train.get_weights()
            if args.skip_vlm_llm:
                wandb_tags = ["ppo", "iterative", "reward-wrapper-only"]
                wandb_group = "PPO-Iterative-NoVLM"
            else:
                wandb_tags = ["ppo", "iterative", "vlm-llm"]
                wandb_group = "PPO-Iterative"
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
        writer = SummaryWriter(f"runs/{run_dir}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # --- Agent ---
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    # --- VLM/LLM setup ---
    vlm = None
    llm = None
    if not args.skip_vlm_llm and not args.evaluate:
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

    # --- Storage ---
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # --- Training ---
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    action_space_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_space_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    # Segment-level configuration
    iters_per_segment = args.num_iterations // args.num_segments
    if iters_per_segment < 1:
        iters_per_segment = 1
    segment_history = []
    initial_weights = reward_wrapper_train.get_weights()

    print(f"####")
    print(f"args.num_iterations={args.num_iterations} iters_per_segment={iters_per_segment} num_segments={args.num_segments}")
    print(f"args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size}")
    print(f"Initial reward weights: {initial_weights}")
    print(f"####")

    global_iteration = 0
    latest_eval_metrics: Dict[str, Any] = {}

    for segment in range(args.num_segments):
        print(f"\n{'='*60}")
        print(f"Segment {segment+1}/{args.num_segments}")
        print(f"Current weights: {reward_wrapper_train.get_weights()}")
        print(f"{'='*60}")

        # --- Inner PPO training loop ---
        for local_iter in range(iters_per_segment):
            global_iteration += 1
            iteration = global_iteration
            print(f"Epoch: {iteration}, global_step={global_step}")
            final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
            agent.eval()

            # Evaluation
            if iteration % args.eval_freq == 1 or local_iter == iters_per_segment - 1:
                print("Evaluating")
                eval_obs, _ = eval_envs.reset()
                eval_metrics = defaultdict(list)
                num_episodes = 0
                for _ in range(args.num_eval_steps):
                    with torch.no_grad():
                        eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(
                            agent.get_action(eval_obs, deterministic=True)
                        )
                        if "final_info" in eval_infos:
                            mask = eval_infos["_final_info"]
                            num_episodes += mask.sum()
                            for k, v in eval_infos["final_info"]["episode"].items():
                                eval_metrics[k].append(v)
                print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
                for k, v in eval_metrics.items():
                    mean = torch.stack(v).float().mean()
                    if logger is not None:
                        logger.add_scalar(f"eval/{k}", mean, global_step)
                    print(f"eval_{k}_mean={mean}")
                latest_eval_metrics = {
                    k: torch.stack(v).float().mean().item() for k, v in eval_metrics.items()
                }
                if args.evaluate:
                    break

            if args.save_model and iteration % args.eval_freq == 1:
                model_path = f"runs/{run_dir}/ckpt_{iteration}.pt"
                torch.save(agent.state_dict(), model_path)
                print(f"model saved to {model_path}")

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

            logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            logger.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            logger.add_scalar("time/step", global_step, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)

        if args.evaluate:
            break

        # --- Segment boundary: VLM/LLM hook ---
        print(f"\n--- Segment {segment+1} complete (global_step={global_step}) ---")

        # Log current weights
        current_weights = reward_wrapper_train.get_weights()
        if logger is not None:
            for wk, wv in current_weights.items():
                logger.add_scalar(f"weights/{wk}", wv, global_step)

        if vlm is not None and llm is not None:
            print(f"[VLM/LLM] Analyzing segment {segment+1}...")

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
                        "success_at_end": latest_eval_metrics.get("success_at_end", 0.0),
                        "success_once": latest_eval_metrics.get("success_once", 0.0),
                        "length": args.num_eval_steps,
                    }

                    try:
                        _, vlm_comment, _ = vlm.evaluate(frames, episode_info)
                        print(f"[VLM] Analysis:\n{vlm_comment}")

                        # Save VLM debug HTML
                        debug_dir = Path(f"runs/{run_dir}/debug_html")
                        save_vlm_debug_html(
                            frames=frames,
                            prompt=build_vlm_prompt(args.env_id),
                            episode_info=episode_info,
                            vlm_score=0.0,  # not used in iterative mode
                            vlm_comment=vlm_comment,
                            save_path=debug_dir / f"segment_{segment+1:02d}_vlm.html",
                            max_frames=args.vlm_max_frames,
                        )
                    except Exception as e:
                        print(f"[VLM] Error: {e}")
                        vlm_comment = f"VLM analysis failed: {e}"
                else:
                    print("[warn] No frames extracted from video")
            else:
                print("[warn] No eval videos found")

            # LLM reward tuning
            if vlm_comment is not None:
                training_summary = {
                    "current_weights": current_weights,
                    "initial_weights": initial_weights,
                    "current_iteration": segment,
                    "total_iterations": args.num_segments,
                    "total_timesteps": global_step,
                    "avg_return": latest_eval_metrics.get("return", 0.0),
                    "success_at_end": latest_eval_metrics.get("success_at_end", 0.0),
                    "success_once": latest_eval_metrics.get("success_once", 0.0),
                    "vlm_avg_score": 0.5,  # placeholder (we don't use scores)
                    "vlm_comments": [vlm_comment],
                    "num_episodes": int(latest_eval_metrics.get("episode_len", 0)),
                }

                try:
                    suggestions = llm.suggest_parameters(training_summary)

                    # Save LLM debug HTML
                    query_info = llm.get_last_query_info() if hasattr(llm, 'get_last_query_info') else None
                    llm_prompt = query_info.get("prompt", "(no prompt)") if query_info else "(no query info)"
                    llm_response = query_info.get("response_text", "(no response)") if query_info else "(no query info)"
                    debug_dir = Path(f"runs/{run_dir}/debug_html")
                    save_llm_debug_html(
                        iteration=segment,
                        prompt=llm_prompt,
                        response_text=llm_response,
                        suggestions=suggestions,
                        summary_for_llm=training_summary,
                        save_path=debug_dir / f"segment_{segment+1:02d}_llm.html",
                    )

                    if suggestions and suggestions.get("type") == "params":
                        new_params = suggestions.get("params", {})
                        rationale = suggestions.get("rationale", "No rationale")
                        if new_params:
                            print(f"[LLM] Rationale: {rationale}")
                            print(f"[LLM] Suggested weights: {new_params}")
                            old_weights = reward_wrapper_train.get_weights()
                            reward_wrapper_train.update_weights(new_params)
                            reward_wrapper_eval.update_weights(new_params)

                            segment_history.append({
                                "segment": segment,
                                "global_step": global_step,
                                "old_weights": old_weights,
                                "new_weights": reward_wrapper_train.get_weights(),
                                "rationale": rationale,
                                "vlm_comment": vlm_comment,
                                "eval_metrics": latest_eval_metrics,
                            })

                            # Log new weights
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
            print(f"[info] VLM/LLM not active, skipping analysis for segment {segment+1}")

    # --- Save final model and history ---
    if not args.evaluate:
        if args.save_model:
            model_path = f"runs/{run_dir}/final_ckpt.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

        # Save segment history
        history_path = f"runs/{run_dir}/segment_history.json"
        with open(history_path, "w") as f:
            json.dump(segment_history, f, indent=2, default=str)
        print(f"segment history saved to {history_path}")

        # Save final weights
        final_weights_path = f"runs/{run_dir}/final_weights.json"
        with open(final_weights_path, "w") as f:
            json.dump(reward_wrapper_train.get_weights(), f, indent=2)
        print(f"final weights saved to {final_weights_path}")

        logger.close()

    envs.close()
    eval_envs.close()

# Proximal Policy Optimization (PPO)

Code for running the PPO RL algorithm is adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl/) and [LeanRL](https://github.com/pytorch-labs/LeanRL/). It is written to be single-file and easy to follow/read, and supports state-based RL and visual-based RL code.

Note that ManiSkill is still in beta, so we have not finalized training scripts for every pre-built task (some of which are simply too hard to solve with RL anyway).

Official baseline results can be run by using the scripts in the baselines.sh file. Results are organized and published to our [wandb report](https://api.wandb.ai/links/stonet2000/k6lz966q)

There is also now experimental support for PPO compiled and with CUDA Graphs enabled based on LeanRL. The code is in ppo_fast.py and you need to install [torchrl](https://github.com/pytorch/rl) and [tensordict](https://github.com/pytorch/tensordict/):

```bash
pip install torchrl tensordict
```

## State Based RL

Below is a sample of various commands you can run to train a state-based policy to solve various tasks with PPO that are lightly tuned already. The fastest one is the PushCube-v1 task which can take less than a minute to train on the GPU and the PickCube-v1 task which can take 2-5 minutes on the GPU.

The PPO baseline is not guaranteed to work for all tasks as some tasks do not have dense rewards yet or well tuned ones, or simply are too hard with standard PPO.


```bash
python ppo.py --env_id="PushCube-v1" \
  --num_envs=2048 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=2_000_000 --eval_freq=10 --num-steps=20
```

To evaluate, you can run
```bash
python ppo.py --env_id="PushCube-v1" \
   --evaluate --checkpoint=path/to/model.pt \
   --num_eval_envs=1 --num-eval-steps=1000
```

Note that with `--evaluate`, trajectories are saved from a GPU simulation. In order to support replaying these trajectories correctly with the `maniskill.trajectory.replay_trajectory` tool for some task, the number of evaluation environments must be fixed to `1`. This is necessary in order to ensure reproducibility for tasks that have randomizations on geometry (e.g. PickSingleYCB). Other tasks without geometrical randomization like PushCube are fine and you can increase the number of evaluation environments. 

The examples.sh file has a full list of tested commands for running state based PPO successfully on many tasks.

The results of running the baseline scripts for state based PPO are here: https://api.wandb.ai/links/stonet2000/k6lz966q.

## Visual (RGB) Based RL

Below is a sample of various commands for training a image-based policy with PPO that are lightly tuned. The fastest again is also PushCube-v1 which can take about 1-5 minutes and PickCube-v1 which takes 15-45 minutes. You will need to tune the `--num_envs` argument according to how much GPU memory you have as rendering visual observations uses a lot of memory. The settings below should all take less than 15GB of GPU memory. The examples.sh file has a full list of tested commands for running visual based PPO successfully on many tasks.


```bash
python ppo_rgb.py --env_id="PushCube-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=8 \
  --total_timesteps=1_000_000 --eval_freq=10 --num-steps=20
python ppo_rgb.py --env_id="PickCube-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=8 \
  --total_timesteps=10_000_000
python ppo_rgb.py --env_id="AnymalC-Reach-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=10_000_000 --num-steps=200 --num-eval-steps=200 \
  --gamma=0.99 --gae_lambda=0.95
```

To evaluate a trained policy you can run

```bash
python ppo_rgb.py --env_id="PickCube-v1" \
  --evaluate --checkpoint=path/to/model.pt \
  --num_eval_envs=1 --num-eval-steps=1000
```

and it will save videos to the `path/to/test_videos`.

The examples.sh file has a full list of tested commands for running RGB based PPO successfully on many tasks.

The results of running the baseline scripts for RGB based PPO are here: https://api.wandb.ai/links/stonet2000/k6lz966q

## Visual (RGB+Depth) Based RL

WIP

## Visual (Pointcloud) Based RL

WIP

## Replaying Evaluation Trajectories

It might be useful to get some nicer looking videos. A simple way to do that is to first use the evaluation scripts provided above. It will then save a .h5 and .json file with a name equal to the date and time that you can then replay with different settings as so

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path=path/to/trajectory.h5 --use-env-states --shader="rt-fast" \
  --save-video --allow-failure -o "none"
```

This will use environment states to replay trajectories, turn on the ray-tracer (There is also "rt" which is higher quality but slower), and save all videos including failed trajectories.

---

# Outer Loop Reward Optimization (Custom Extension)

Below is **not** part of upstream ManiSkill. It implements an Eureka-style
([Ma et al., 2023](https://eureka-research.github.io/)) outer-loop reward
optimization system that uses LLM (and optionally VLM) to iteratively improve
reward functions for ManiSkill tasks.

## Architecture

The system has **3 operating modes**, all sharing the same PPO training core:

```
                         ┌──────────────────────────────────────────────────────┐
                         │            Outer Loop (N iterations)                 │
                         │                                                      │
  ┌──────────┐  random   │  ┌───────────────┐   train K    ┌────────────────┐  │
  │  Init    │──weights──►│  │ Generate K    │──candidates──►│  PPO Training  │  │
  │  (Iter 0)│           │  │ Candidates    │  (parallel)  │  (per cand.)   │  │
  └──────────┘           │  └───────┬───────┘              └───────┬────────┘  │
                         │          │                              │            │
                         │          │ LLM                  eval    │            │
                         │          │                    metrics   │            │
                         │  ┌───────┴───────┐              ┌──────┴─────────┐  │
                         │  │  Reward       ◄──────────────│  Select Best   │  │
                         │  │  Reflection   │  fitness     │  Candidate     │  │
                         │  └───────┬───────┘              └───────┬────────┘  │
                         │          │                              │            │
                         │          │ (VLM mode only)      ┌──────┴─────────┐  │
                         │          ◄──────────────────────│  VLM Video     │  │
                         │                                 │  Analysis      │  │
                         │                                 └────────────────┘  │
                         └──────────────────────────────────────────────────────┘
```

### 3 Modes

| Mode | Script | LLM Output | VLM | Description |
|------|--------|-----------|-----|-------------|
| **Params-only** | `ppo_outer_loop.py` | JSON weight dict | optional | LLM tunes `{w_reach: 1.5, w_grasp: 3.0, ...}` weights for a fixed reward template |
| **Eureka-full** | `ppo_outer_loop_full.py --eureka_mode` | Python function | no | LLM generates entire `compute_reward()` function (Eureka paper style) |
| **VLM-full** | `ppo_outer_loop_full.py` | Python function | yes | Same as Eureka-full, but VLM analyzes eval videos and feeds failure analysis to LLM |

**Key concepts:**
- Each iteration trains **K candidates** (default 4) with different reward functions, then selects the best by `success_at_end` fitness
- **Reward Reflection**: LLM receives per-component reward statistics and learning curves from the previous iteration to guide improvements
- **Elite carry-over**: The best candidate from iteration N is carried into iteration N+1 as one of the K slots
- Parallel training across multiple GPUs is supported (`--gpus="0,1,0,1"`)

### Reward Wrappers

| File | Used by | Description |
|------|---------|-------------|
| `reward_wrapper.py` | `ppo_outer_loop.py` | Fixed YAML-based template with tunable weights per component |
| `reward_wrapper_dynamic.py` | `ppo_outer_loop_full.py` | Extends the above with `set_custom_function()` for LLM-generated Python code |

Both use **weighted additive components** (e.g. `w_reach * reach_reward + w_grasp * grasp_reward + ...`) with `_norm_scale()` for magnitude stability. Environment `reward_mode="none"` is required.

### Supported Tasks

| Task | Components |
|------|-----------|
| PushCube-v1 | w_reach, w_push, w_z_keep, w_success |
| PickCube-v1 | w_reach, w_grasp, w_place, w_static, w_success |
| OpenCabinetDoor-v1 | w_reach, w_open, w_static, w_success |
| OpenCabinetDrawer-v1 | w_reach, w_open, w_static, w_success |
| PegInsertionSide-v1 | w_reach, w_grasp, w_pre_insertion, w_insertion, w_success |
| PushT-v1 | w_rotation, w_position, w_tcp_guide, w_success |
| AnymalC-Reach-v1 | w_reach, w_vel_z_penalty, w_ang_vel_penalty, w_contact_penalty, w_qpos_penalty |
| UnitreeG1PlaceAppleInBowl-v1 | w_reach, w_grasp, w_place, w_above_bowl, w_release, w_success |
| PickCubePandaAllegro-v2 | w_reach, w_grasp, w_place, w_static, w_success |

## Quick Start

Requires `OPENAI_API_KEY` (or compatible endpoint) for LLM/VLM.

```bash
# Params-only (weight tuning, LLM-only or VLM+LLM):
bash outer_loop_params.sh eureka   # LLM-only
bash outer_loop_params.sh vlm      # VLM+LLM

# Full replacement (reward function generation, LLM-only or VLM+LLM):
bash outer_loop_full.sh eureka     # LLM-only
bash outer_loop_full.sh vlm        # VLM+LLM
```

Manual single-task example:

```bash
export OPENAI_API_KEY=sk-...

# Params-only mode:
python ppo_outer_loop.py \
  --env_id="PickCube-v1" --num_outer_iters=5

# Eureka-full mode:
python ppo_outer_loop_full.py \
  --env_id="PickCube-v1" --eureka_mode \
  --num_outer_iters=5 --num_reward_candidates=4

# Debug (no LLM/VLM):
python ppo_outer_loop.py \
  --env_id="PickCube-v1" --skip_vlm_llm \
  --num_outer_iters=2 --total_timesteps_per_iter=50000
```

## PickCubePandaAllegro (Dexterous Hand)

Training PPO for the Panda + Allegro dexterous hand on PickCube. `coupled_allegro_wrapper.py` maps the 22D action space to 8D (6 arm EE delta + 2 hand scalars).

```bash
# Quick debug:
bash allegro_debug.sh

# Full training:
python ppo.py \
  --env_id="PickCubePandaAllegro-v1" \
  --control_mode="pd_joint_delta_pos_coupled" \
  --num_envs=4096 --num_steps=100 --num_eval_steps=100 \
  --update_epochs=8 --num_minibatches=32 \
  --gamma=0.95 --gae_lambda=0.95 --ent_coef=0.01 \
  --total_timesteps=50_000_000 \
  --finite_horizon_gae --partial_reset
```

## Iterative VLM/LLM Reward Tuning

`ppo_iterative.py` splits a single training run into segments and adjusts reward weights between segments (without restarting from scratch). Lighter-weight than the outer loop but less thorough.

```bash
export OPENAI_API_KEY=sk-... && python ppo_iterative.py \
  --env_id="PickCube-v1" \
  --total_timesteps=10_000_000 --num_segments=10
```

---

## File Overview

### ManiSkill Upstream

| File | Description |
|------|-------------|
| `ppo.py` | Standard PPO baseline (CleanRL-based, state observations) |
| `ppo_fast.py` | PPO with CUDA Graphs (requires torchrl/tensordict) |
| `ppo_rgb.py` | PPO with RGB visual observations |
| `baselines.sh` | Standard PPO baseline commands |
| `examples.sh` | Full list of tested PPO commands for many tasks |
| `README.md` | This file (upstream sections above + custom extension below) |

### Outer Loop System

| File | Description |
|------|-------------|
| `ppo_outer_loop.py` | Params-only outer loop (LLM tunes weight dicts) |
| `ppo_outer_loop_full.py` | Eureka-full / VLM-full outer loop (LLM generates reward functions) |
| `ppo_common.py` | Shared utilities: Agent network, Logger, video/VLM helpers, random weight generation |
| `task_descriptions.py` | Per-task LLM prompt descriptions and state access documentation |
| `reward_wrapper.py` | Fixed-template reward wrapper with tunable weights |
| `reward_wrapper_dynamic.py` | Dynamic reward wrapper supporting LLM-generated Python code |
| `env_contracts.py` | Environment setup validation (action dims, control modes) |
| `coupled_allegro_wrapper.py` | Maps 22D PandaAllegro action space to 8D coupled control |
| `ppo_iterative.py` | Iterative reward tuning (inner loop, no restart) |
| `generate_rollout_videos.py` | Generate rollout videos from saved checkpoints |

### Shell Scripts

| File | Description |
|------|-------------|
| `outer_loop_params.sh` | Params-only mode (takes `vlm` or `eureka` argument) |
| `outer_loop_full.sh` | Full replacement mode (takes `vlm` or `eureka` argument) |
| `allegro_debug.sh` | PandaAllegro single-run debug script |
| `iterative.sh` | Iterative reward tuning (in-place weight adjustment, no restart) |

### Plotting (`plotting/`)

| File | Description |
|------|-------------|
| `plot_all_tasks.py` | All-tasks summary: training curves + success vs iteration (`--mode` selects outer-loop/eureka/full) |
| `plot_single_run.py` | Single-run progress monitor for full (multi-candidate) modes, works mid-run |
| `plot_outer_loop_iterations.py` | Per-iteration overlay with arbitrary `--metric` (e.g. reward weights) |
| `plot_method_comparison.py` | Compare outer-loop vs eureka across tasks |

## Some Notes

- Evaluation with GPU simulation (especially with randomized objects) is a bit tricky. We recommend reading through [our docs](https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/baselines.html#evaluation) on online RL evaluation in order to understand how to fairly evaluate policies with GPU simulation.
- Many tasks support visual observations, however we have not carefully verified yet if the camera poses for the tasks are setup in a way that makes it possible to solve some tasks from visual observations.

## Citation

If you use this baseline please cite the following
```
@article{DBLP:journals/corr/SchulmanWDRK17,
  author       = {John Schulman and
                  Filip Wolski and
                  Prafulla Dhariwal and
                  Alec Radford and
                  Oleg Klimov},
  title        = {Proximal Policy Optimization Algorithms},
  journal      = {CoRR},
  volume       = {abs/1707.06347},
  year         = {2017},
  url          = {http://arxiv.org/abs/1707.06347},
  eprinttype    = {arXiv},
  eprint       = {1707.06347},
  timestamp    = {Mon, 13 Aug 2018 16:47:34 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/SchulmanWDRK17.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
#!/bin/bash
# PPO single-run debug for Panda+Allegro+TouchLab (PickCube v2, coupled fingers)
#
# Uses ppo.py directly (no outer loop / no Eureka).
# CoupledAllegroActionWrapper reduces 22D -> 8D action space:
#   [0:6] arm EE delta pose (pd_ee_delta_pose)
#   [6]   finger group scalar [-1=open, 1=closed]
#   [7]   thumb scalar [-1=open, 1=closed]
#
# Usage:
#   bash allegro_debug.sh

seed=9351
ENV="PickCubePandaAllegro-v2"

# --- Parallelism ---
NUM_ENVS=4096        # TouchLab sensors need more GPU memory per env
NUM_EVAL_ENVS=16

# --- Rollout ---
NUM_STEPS=100          # = max_episode_steps (full-episode rollouts)
NUM_EVAL_STEPS=100
TOTAL=50_000_000        # short run for debug; increase for real training

# --- PPO ---
UPDATE_EPOCHS=8        # more gradient steps per rollout
NUM_MINIBATCHES=32     # minibatch = 256*100/32 = 800
GAMMA=0.95             # longer horizon for grasp→lift→place chain
GAE_LAMBDA=0.95
ENT_COEF=0.01          # entropy bonus for exploration
LR=3e-4
REWARD_SCALE=1.0

CUDA_VISIBLE_DEVICES=0 python ppo.py \
  --env_id="${ENV}" \
  --seed=${seed} \
  --num_envs=${NUM_ENVS} \
  --num_steps=${NUM_STEPS} \
  --num_eval_steps=${NUM_EVAL_STEPS} \
  --update_epochs=${UPDATE_EPOCHS} \
  --num_minibatches=${NUM_MINIBATCHES} \
  --total_timesteps=${TOTAL} \
  --num_eval_envs=${NUM_EVAL_ENVS} \
  --gamma=${GAMMA} \
  --gae_lambda=${GAE_LAMBDA} \
  --ent_coef=${ENT_COEF} \
  --learning_rate=${LR} \
  --reward_scale=${REWARD_SCALE} \
  --finite_horizon_gae \
  --partial_reset \
  --track \
  --wandb-project-name="maniskill-allegro" \
  --exp-name="ppo-allegro-debug-${ENV}-${seed}-$(date +%Y%m%d_%H%M%S)"

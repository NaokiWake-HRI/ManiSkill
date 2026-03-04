#!/bin/bash
# PPO single-run debug for Panda+Allegro+TouchLab (PickCube v2)
#
# Uses ppo.py directly (no outer loop / no Eureka).
# SimToolReal-style control: arm joint delta (7D) + hand absolute pos (16D) = 23D
#   control_mode: pd_joint_target_delta_pos_arm_abs_hand
#   sim_freq=120, control_freq=60 (matching SimToolReal's 60Hz control + 2 substeps)
#
# Usage:
#   bash allegro_debug.sh

seed=9351
ENV="PickCubePandaAllegro-v2"

# --- Parallelism ---
NUM_ENVS=4096        # TouchLab sensors need more GPU memory per env
NUM_EVAL_ENVS=16

# --- Rollout ---
NUM_STEPS=300          # = max_episode_steps (60Hz * 5s = 300 steps)
NUM_EVAL_STEPS=300
TOTAL=200_000_000      # increased for higher control freq + larger action space

# --- PPO ---
UPDATE_EPOCHS=8        # more gradient steps per rollout
NUM_MINIBATCHES=32
GAMMA=0.995            # match SimToolReal (longer horizon at 60Hz)
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

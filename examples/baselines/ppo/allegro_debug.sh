#!/bin/bash
# PPO single-run debug for Panda+Allegro+TouchLab (PickCube v2)
#
# Uses ppo.py directly (no outer loop / no Eureka).
# SimToolReal-style control: arm joint delta (7D) + hand absolute pos (16D) = 23D
#   control_mode: pd_joint_target_delta_pos_arm_abs_hand
#   sim_freq=120, control_freq=60 (matching SimToolReal's 60Hz control + 2 substeps)
#
# Hyperparameters aligned with SimToolReal (rl_games PPO):
#   num_envs=8192, horizon=16, gamma=0.99, lr=1e-4, ent=0.0, clip=0.1
#   reward_scale=0.01, vf_coef=4.0, grad_norm=1.0
#   network=[1024,1024,512,512] ELU
#
# Usage:
#   bash allegro_debug.sh

seed=9351
ENV="PickCubePandaAllegro-v2"

# --- Parallelism (SimToolReal: 8192) ---
NUM_ENVS=8192
NUM_EVAL_ENVS=16

# --- Rollout (SimToolReal: horizon=16, episode=600) ---
NUM_STEPS=16               # short rollout, frequent PPO updates (SimToolReal: 16)
NUM_EVAL_STEPS=600          # full episode for evaluation (60Hz * 10s = 600 steps)
TOTAL=200_000_000

# --- PPO (aligned with SimToolReal) ---
UPDATE_EPOCHS=4             # SimToolReal: mini_epochs=4
NUM_MINIBATCHES=4           # batch=8192*16=131072, minibatch=32768 (SimToolReal: 32768)
GAMMA=0.99                  # SimToolReal: 0.99
GAE_LAMBDA=0.95             # SimToolReal: 0.95
ENT_COEF=0.0                # SimToolReal: 0.0
LR=1e-4                     # SimToolReal: 1e-4
CLIP_COEF=0.1               # SimToolReal: e_clip=0.1
VF_COEF=4.0                 # SimToolReal: critic_coef=4.0
MAX_GRAD_NORM=1.0           # SimToolReal: grad_norm=1.0
REWARD_SCALE=0.01           # SimToolReal: reward_shaper scale=0.01

# --- Network (SimToolReal: [1024,1024,512,512] ELU) ---
ACTIVATION="elu"

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
  --clip_coef=${CLIP_COEF} \
  --vf_coef=${VF_COEF} \
  --max_grad_norm=${MAX_GRAD_NORM} \
  --reward_scale=${REWARD_SCALE} \
  --hidden_sizes 1024 1024 512 512 \
  --activation=${ACTIVATION} \
  --finite_horizon_gae \
  --partial_reset \
  --track \
  --wandb-project-name="maniskill-allegro" \
  --exp-name="ppo-allegro-debug-${ENV}-${seed}-$(date +%Y%m%d_%H%M%S)"

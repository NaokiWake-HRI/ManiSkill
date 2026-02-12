#!/bin/bash
# PPO training for Panda+Allegro+FSR (PickCube, coupled fingers)
#
# Tuned for GPU-heavy training with sufficient resources.
#
# Usage:
#   bash allegro_debug.sh

seed=9351
ENV="PickCubePandaAllegroTouch-v1"
CONTROL_MODE="pd_joint_delta_pos_coupled"

# --- Parallelism ---
NUM_ENVS=512           # FSR touch sensors need more GPU memory per env
NUM_EVAL_ENVS=16

# --- Rollout ---
NUM_STEPS=100          # = max_episode_steps (full-episode rollouts)
NUM_EVAL_STEPS=100
TOTAL=50_000_000       # dexterous tasks need more samples

# --- PPO ---
UPDATE_EPOCHS=8        # more gradient steps per rollout
NUM_MINIBATCHES=32     # minibatch = 512*100/32 = 1600
GAMMA=0.95             # longer horizon for grasp→lift→place chain
GAE_LAMBDA=0.95
ENT_COEF=0.01          # entropy bonus for exploration
LR=3e-4
REWARD_SCALE=1.0

python ppo.py \
  --env_id="${ENV}" \
  --seed=${seed} \
  --num_envs=${NUM_ENVS} \
  --num_steps=${NUM_STEPS} \
  --num_eval_steps=${NUM_EVAL_STEPS} \
  --update_epochs=${UPDATE_EPOCHS} \
  --num_minibatches=${NUM_MINIBATCHES} \
  --total_timesteps=${TOTAL} \
  --num_eval_envs=${NUM_EVAL_ENVS} \
  --control_mode="${CONTROL_MODE}" \
  --gamma=${GAMMA} \
  --gae_lambda=${GAE_LAMBDA} \
  --ent_coef=${ENT_COEF} \
  --learning_rate=${LR} \
  --reward_scale=${REWARD_SCALE} \
  --output-dir=debug \
  --finite_horizon_gae \
  --partial_reset \
  --exp-name="ppo-${ENV}-coupled-${seed}" \
  --track

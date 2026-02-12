#!/bin/bash
# Debug script for PPO with Panda+Allegro (coupled fingers)
# Quick test run with reduced timesteps to verify the environment works
#
# Usage:
#   bash allegro_debug.sh

seed=9351
ENV="PickCubePandaAllegro-v1"
CONTROL_MODE="pd_joint_delta_pos_coupled"

# Keep these modest for a first look; bump TOTAL for longer runs.
TOTAL=5_000_000
NUM_ENVS=512
NUM_STEPS=50
UPDATE_EPOCHS=4
NUM_MINIBATCHES=32
NUM_EVAL_ENVS=8

python ppo.py \
  --env_id="${ENV}" \
  --seed=${seed} \
  --num_envs=${NUM_ENVS} \
  --num_steps=${NUM_STEPS} \
  --update_epochs=${UPDATE_EPOCHS} \
  --num_minibatches=${NUM_MINIBATCHES} \
  --total_timesteps=${TOTAL} \
  --num_eval_envs=${NUM_EVAL_ENVS} \
  --control_mode="${CONTROL_MODE}" \
  --exp-name="debug-ppo-${ENV}-coupled-${seed}" \
  --track

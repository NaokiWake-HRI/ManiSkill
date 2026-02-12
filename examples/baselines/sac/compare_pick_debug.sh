#!/bin/bash
# Quick comparison: PickCube SAC, seed=9351
#   1) noVLM (skip_vlm_llm)
#   2) withVLM (include reward plot, recompute)
#
# Usage:
#   export OPENAI_API_KEY=sk-... && bash compare_pick_debug.sh
#
# Fast mode (2x speed, same learning dynamics):
#   export OPENAI_API_KEY=sk-... && FAST=1 bash compare_pick_debug.sh

seed=9351
ENV="PickCube-v1"
TOTAL=250_000
SEGMENTS=10

if [ "${FAST:-0}" = "1" ]; then
    echo "Running in FAST mode (num_envs=32, training_freq=128)"
    COMMON="--env_id=${ENV} --seed=${seed} \
  --num_envs=32 --training_freq=128 --num_eval_envs=16 \
  --total_timesteps=${TOTAL} --num_segments=${SEGMENTS} --track"
else
    COMMON="--env_id=${ENV} --seed=${seed} \
  --num_envs=16 --num-steps=50 --num_eval_envs=16 \
  --total_timesteps=${TOTAL} --num_segments=${SEGMENTS} --track"
fi

### 1) noVLM ###
echo "=== [1/2] noVLM ==="
python sac_iterative_debug.py ${COMMON} \
  --skip_vlm_llm \
  --exp-name="sac-debug-noVLM-${ENV}-${seed}"

### 2) withVLM (recompute + critic warmup) ###
if [ -z "${OPENAI_API_KEY}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set. Skipping withVLM run."
    exit 1
fi
echo "=== [2/2] withVLM (recompute + critic warmup) ==="
python sac_iterative_debug.py ${COMMON} \
  --vlm_reward_plot \
  --critic_warmup_steps=5000 \
  --exp-name="sac-debug-withVLM-recompute-${ENV}-${seed}"

#!/bin/bash
# Iterative VLM/LLM PPO Baselines
# Compare with standard baselines.sh results on wandb
#
# Usage:
#   export OPENAI_API_KEY=sk-... && bash baselines_iterative.sh  # Full run with VLM/LLM
#   SKIP_VLM=1 bash baselines_iterative.sh                       # Without VLM/LLM (reward wrapper only)

seeds=(9351 4796 1788)

SKIP_VLM_FLAG=""
if [ "${SKIP_VLM:-0}" = "1" ]; then
    SKIP_VLM_FLAG="--skip_vlm_llm"
    echo "Running WITHOUT VLM/LLM (reward wrapper only)"
else
    if [ -z "${OPENAI_API_KEY}" ]; then
        echo "ERROR: OPENAI_API_KEY is not set. Either set it or use SKIP_VLM=1."
        echo "  export OPENAI_API_KEY=sk-..."
        exit 1
    fi
    echo "Running WITH VLM/LLM (OPENAI_API_KEY is set)"
fi

### PickCube Panda+Allegro ###
for seed in ${seeds[@]}
do
  python ppo_iterative.py --env_id="PickCubePandaAllegro-v1" --seed=${seed} \
    --num_envs=256 --num-steps=100 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=10_000_000 \
    --num_eval_envs=8 \
    --num_segments=10 \
    --control_mode="pd_joint_delta_pos_coupled" \
    --exp-name="ppo-iterative-PickCubePandaAllegro-v1-state-${seed}" \
    --track \
    ${SKIP_VLM_FLAG}
done

#!/bin/bash
# Iterative VLM/LLM SAC Baselines
# Compare with standard sac.py results on wandb
#
# Usage:
#   export OPENAI_API_KEY=sk-... && bash baselines_iterative.sh  # Full run with VLM/LLM
#   SKIP_VLM=1 bash baselines_iterative.sh                       # Without VLM/LLM (reward wrapper only)

seeds=(9351) # 4796 1788

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

### PickCube ###
for seed in ${seeds[@]}
do
  python sac_iterative.py --env_id="PickCube-v1" --seed=${seed} \
    --num_envs=16 --num-steps=50 --num_eval_envs=16 \
    --total_timesteps=500_000 \
    --num_segments=20 \
    --vlm_reward_plot \
    --exp-name="sac-iterative-PickCube-v1-state-${seed}" \
    --track \
    ${SKIP_VLM_FLAG}
done

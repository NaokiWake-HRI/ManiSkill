#!/bin/bash
# PPO Outer Loop: VLM/LLM-guided Reward Weight Optimization
#
# Starts from random weights, runs full PPO training per iteration,
# then uses VLM/LLM to suggest improved weights for the next iteration.
# Iteration 1 (random weights) serves as the baseline.
#
# Usage:
#   export OPENAI_API_KEY=sk-... && bash outer_loop_run.sh

seed=9351
OUTER_ITERS=5
WSEED=42

if [ -z "${OPENAI_API_KEY}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "  export OPENAI_API_KEY=sk-..."
    exit 1
fi

echo "=== PPO Outer Loop ==="
echo "Outer iterations: ${OUTER_ITERS}"
echo "Weight seed: ${WSEED}"
echo ""

for ENV in "PushCube-v1" "PickCube-v1" "OpenCabinetDoor-v1" "OpenCabinetDrawer-v1" "PushChair-v1" "PegInsertionSide-v1" "PushT-v1" "UnitreeG1PlaceAppleInBowl-v1" "AnymalC-Reach-v1"
do
    # Set total timesteps based on task
    if [ "${ENV}" == "PegInsertionSide-v1" ]; then
        TOTAL=4_500_000
    else
        TOTAL=3_000_000
    fi

    echo "=== ${ENV} ==="
    python ppo_outer_loop.py \
      --env_id="${ENV}" \
      --seed=${seed} \
      --num_envs=512 \
      --num_steps=50 \
      --update_epochs=4 \
      --num_minibatches=32 \
      --num_eval_envs=8 \
      --num_outer_iters=${OUTER_ITERS} \
      --total_timesteps_per_iter=${TOTAL} \
      --weight_seed=${WSEED} \
      --vlm_reward_plot \
      --track \
      --exp-name="ppo-outer-loop-${ENV}-${seed}"
    echo ""
done

echo "=== Outer loop experiments complete ==="
echo "Check wandb for results (group: PPO-OuterLoop)"

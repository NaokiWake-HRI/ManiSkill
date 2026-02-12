#!/bin/bash
# PPO Outer Loop: Pure Eureka Mode (LLM-only, no VLM)
#
# This script runs the outer loop optimization using only LLM (without VLM).
# The LLM uses learning curve data and performance metrics to optimize weights,
# similar to the original Eureka paper approach.
#
# Comparison:
#   - outer_loop_run.sh: VLM + LLM (video analysis + reward optimization)
#   - outer_loop_eureka.sh: LLM only (pure Eureka mode)
#
# Usage:
#   export OPENAI_API_KEY=sk-... && bash outer_loop_eureka.sh

seed=9351
OUTER_ITERS=5
WSEED=42

if [ -z "${OPENAI_API_KEY}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "  export OPENAI_API_KEY=sk-..."
    exit 1
fi

echo "=== PPO Outer Loop: Pure Eureka Mode (LLM-only) ==="
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
      --eureka_mode \
      --track \
      --exp-name="ppo-eureka-${ENV}-${seed}"
    echo ""
done

echo "=== Eureka mode experiments complete ==="
echo "Check wandb for results (group: PPO-OuterLoop, tags: eureka)"

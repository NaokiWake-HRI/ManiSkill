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

for ENV in "PushCube-v1" "PickCube-v1" "OpenCabinetDoor-v1" "OpenCabinetDrawer-v1" "PegInsertionSide-v1" "PushT-v1" "UnitreeG1PlaceAppleInBowl-v1" "AnymalC-Reach-v1"
do
    # Set hyperparameters based on task (following baselines.sh exactly, except total_timesteps)
    if [ "${ENV}" == "PushCube-v1" ] || [ "${ENV}" == "PickCube-v1" ]; then
        TOTAL=3_000_000
        EVAL_STEPS=50
        NUM_STEPS=4
        UPDATE_EPOCHS=8
        GAMMA_ARG=""
        GAE_LAMBDA_ARG=""
    elif [ "${ENV}" == "PegInsertionSide-v1" ]; then
        TOTAL=4_500_000  # Baseline: 75M, scaled down proportionally
        EVAL_STEPS=100
        NUM_STEPS=16
        UPDATE_EPOCHS=8
        GAMMA_ARG="--gamma=0.97"
        GAE_LAMBDA_ARG="--gae_lambda=0.95"
    elif [ "${ENV}" == "OpenCabinetDoor-v1" ] || [ "${ENV}" == "OpenCabinetDrawer-v1" ]; then
        TOTAL=3_000_000
        EVAL_STEPS=100
        NUM_STEPS=16
        UPDATE_EPOCHS=8
        GAMMA_ARG=""
        GAE_LAMBDA_ARG=""
    elif [ "${ENV}" == "PushT-v1" ]; then
        TOTAL=3_000_000
        EVAL_STEPS=100
        NUM_STEPS=16
        UPDATE_EPOCHS=8
        GAMMA_ARG="--gamma=0.99"
        GAE_LAMBDA_ARG=""
    elif [ "${ENV}" == "UnitreeG1PlaceAppleInBowl-v1" ]; then
        TOTAL=3_000_000
        EVAL_STEPS=100
        NUM_STEPS=32
        UPDATE_EPOCHS=8
        GAMMA_ARG=""
        GAE_LAMBDA_ARG=""
    elif [ "${ENV}" == "AnymalC-Reach-v1" ]; then
        TOTAL=3_000_000
        EVAL_STEPS=200
        NUM_STEPS=16
        UPDATE_EPOCHS=8
        GAMMA_ARG="--gamma=0.99"
        GAE_LAMBDA_ARG="--gae_lambda=0.95"
    else
        TOTAL=3_000_000
        EVAL_STEPS=50
        NUM_STEPS=50
        UPDATE_EPOCHS=8
        GAMMA_ARG=""
        GAE_LAMBDA_ARG=""
    fi

    echo "=== ${ENV} ==="
    python ppo_outer_loop.py \
      --env_id="${ENV}" \
      --seed=${seed} \
      --num_envs=512 \
      --num_steps=${NUM_STEPS} \
      --update_epochs=${UPDATE_EPOCHS} \
      --num_minibatches=32 \
      --num_eval_envs=16 \
      --num_eval_steps=${EVAL_STEPS} \
      --num_outer_iters=${OUTER_ITERS} \
      --total_timesteps_per_iter=${TOTAL} \
      --weight_seed=${WSEED} \
      ${GAMMA_ARG} \
      ${GAE_LAMBDA_ARG} \
      --eureka_mode \
      --track \
      --exp-name="ppo-eureka-${ENV}-${seed}"
    echo ""
done

echo "=== Eureka mode experiments complete ==="
echo "Check wandb for results (group: PPO-OuterLoop, tags: eureka)"

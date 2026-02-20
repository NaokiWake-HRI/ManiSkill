#!/bin/bash
# PPO Outer Loop: Pure Eureka Mode (LLM-only, no VLM)
#
# This script runs the outer loop optimization using only LLM (without VLM).
# The LLM uses learning curve data and performance metrics to optimize weights,
# similar to the original Eureka paper approach.
#
# Comparison:
#   - outer_loop_vlm_params.sh: VLM + LLM (video analysis + reward optimization)
#   - outer_loop_eureka.sh: LLM only (pure Eureka mode)
#
# Usage:
#   export OPENAI_API_KEY=sk-... && bash outer_loop_eureka.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

seeds=(9351 4796 1788)
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
echo "Seeds: ${seeds[@]}"
echo ""

for seed in ${seeds[@]}
do
  echo "========================================="
  echo "=== Running with seed: ${seed} ==="
  echo "========================================="

  for ENV in "UnitreeG1PlaceAppleInBowl-v1" "AnymalC-Reach-v1" #"PegInsertionSide-v1" "PushT-v1" "PushCube-v1" "PickCube-v1" "OpenCabinetDoor-v1" "OpenCabinetDrawer-v1" # 
  do
    # Hyperparameters per task (same as outer_loop_vlm_params.sh)
    # Outer loop uses longer rollouts and increased parallel environments (1024 vs 2048-4096 baseline).
    # Total timesteps: 15M/iter for complex tasks vs 50-75M baseline.
    # Batch size ratios between tasks are preserved from baselines.sh (2x for PegInsertion/PushT).
    #   Baseline reference batch sizes (num_envs * num_steps):
    #     PushCube/PickCube  = 4096*4   = 16,384  (1x)
    #     PegInsertion       = 2048*16  = 32,768  (2x)
    #     OpenCabinet        = 1024*16  = 16,384  (1x)
    #     PushT              = 4096*16  = 65,536  (4x)
    #     UnitreeG1          = 1024*32  = 32,768  (2x)
    #     AnymalC            = 4096*16  = 65,536  (4x)
    if [ "${ENV}" == "PushCube-v1" ] || [ "${ENV}" == "PickCube-v1" ]; then
        TOTAL=3_000_000          # Baseline: 50M
        EVAL_STEPS=50
        NUM_ENVS=256             # Baseline: 4096
        NUM_STEPS=100            # Baseline: 4  (batch: 256*100=25,600  1x)
        UPDATE_EPOCHS=8
        GAMMA_ARG=""
        GAE_LAMBDA_ARG=""
    elif [ "${ENV}" == "PegInsertionSide-v1" ]; then
        TOTAL=24_000_000          # Baseline: 75M
        EVAL_STEPS=100
        NUM_ENVS=1024            # Baseline: 2048
        NUM_STEPS=64             # Baseline: 16 (batch: 1024*64=65,536  2x)
        UPDATE_EPOCHS=8
        GAMMA_ARG="--gamma=0.97"
        GAE_LAMBDA_ARG="--gae_lambda=0.95"
    elif [ "${ENV}" == "OpenCabinetDoor-v1" ] || [ "${ENV}" == "OpenCabinetDrawer-v1" ]; then
        TOTAL=6_000_000          # Baseline: 50M
        EVAL_STEPS=100
        NUM_ENVS=256             # Baseline: 1024
        NUM_STEPS=100            # Baseline: 16 (batch: 256*100=25,600  1x)
        UPDATE_EPOCHS=8
        GAMMA_ARG=""
        GAE_LAMBDA_ARG=""
    elif [ "${ENV}" == "PushT-v1" ]; then
        TOTAL=24_000_000          # Baseline: 50M
        EVAL_STEPS=100
        NUM_ENVS=1024            # Baseline: 4096
        NUM_STEPS=128            # Baseline: 16 (batch: 1024*128=131,072  2x)
        UPDATE_EPOCHS=8
        GAMMA_ARG="--gamma=0.99"
        GAE_LAMBDA_ARG=""
    elif [ "${ENV}" == "UnitreeG1PlaceAppleInBowl-v1" ]; then
        TOTAL=12_000_000          # Baseline: 50M
        EVAL_STEPS=100
        NUM_ENVS=512             # Baseline: 1024
        NUM_STEPS=100            # Baseline: 32 (batch: 512*100=51,200  2x)
        UPDATE_EPOCHS=8
        GAMMA_ARG=""
        GAE_LAMBDA_ARG=""
    elif [ "${ENV}" == "AnymalC-Reach-v1" ]; then
        TOTAL=6_000_000          # Baseline: 50M
        EVAL_STEPS=200
        NUM_ENVS=512             # Baseline: 4096
        NUM_STEPS=200            # Baseline: 16 (batch: 512*200=102,400 4x)
        UPDATE_EPOCHS=8
        GAMMA_ARG="--gamma=0.99"
        GAE_LAMBDA_ARG="--gae_lambda=0.95"
    else
        TOTAL=3_000_000
        EVAL_STEPS=50
        NUM_ENVS=256
        NUM_STEPS=100
        UPDATE_EPOCHS=8
        GAMMA_ARG=""
        GAE_LAMBDA_ARG=""
    fi

    echo "=== ${ENV} ==="
    python ppo_outer_loop.py \
      --env_id="${ENV}" \
      --seed=${seed} \
      --num_envs=${NUM_ENVS} \
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
      --rl_project_path="/home/robotics/naoki_workspace/codes/robotics_rl" \
      --track \
      --exp-name="ppo-eureka-${ENV}-${seed}"
    echo ""
  done

  echo "=== Completed seed ${seed} ==="
  echo ""
done

echo "========================================="
echo "=== All Eureka mode experiments complete ==="
echo "Seeds run: ${seeds[@]}"
echo "Check wandb for results (group: PPO-OuterLoop, tags: eureka)"
echo "========================================="

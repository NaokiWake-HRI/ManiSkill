#!/bin/bash
# PPO Outer Loop: Unified script for VLM+LLM and Eureka (LLM-only) params-only modes
#
# Modes:
#   vlm    - VLM+LLM: VLM analyzes robot behavior from videos,
#            LLM optimizes reward weights using VLM feedback.
#   eureka - Eureka (LLM-only): No VLM video analysis.
#            Ablation counterpart of vlm mode for fair comparison.
#
# Common behavior:
# - LLM tunes reward weight dicts (params-only, not full function replacement)
# - Single candidate per iteration (no K-candidate parallelism)
# - Seeds and envs are processed sequentially on a single GPU
#
# Usage:
#   export OPENAI_API_KEY=sk-... && bash outer_loop_params.sh vlm
#   export OPENAI_API_KEY=sk-... && bash outer_loop_params.sh eureka

# --- Mode selection ---
MODE="${1:-vlm}"
if [ "${MODE}" != "vlm" ] && [ "${MODE}" != "eureka" ]; then
    echo "Usage: bash outer_loop_params.sh [vlm|eureka]"
    echo "  vlm    : VLM+LLM mode (default)"
    echo "  eureka : LLM-only mode (no VLM)"
    exit 1
fi

# --- Configuration ---
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

seeds=(9351 4796 1788)
OUTER_ITERS=5
WSEED=42

if [ "${MODE}" == "eureka" ]; then
    EUREKA_ARG="--eureka_mode"
    EXP_PREFIX="ppo-eureka"
    MODE_LABEL="Pure Eureka (LLM-only)"
    WANDB_TAG="eureka"
else
    EUREKA_ARG="--vlm_reward_plot"
    EXP_PREFIX="ppo-outer-loop"
    MODE_LABEL="VLM+LLM Params"
    WANDB_TAG="vlm-params"
fi

if [ -z "${OPENAI_API_KEY}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "  export OPENAI_API_KEY=sk-..."
    exit 1
fi

echo "=== PPO Outer Loop: ${MODE_LABEL} ==="
echo "Outer iterations: ${OUTER_ITERS}"
echo "Weight seed: ${WSEED}"
echo "Seeds: ${seeds[@]}"
echo ""

any_failed=0
for seed in "${seeds[@]}"
do
  echo "========================================="
  echo "=== Running with seed: ${seed} ==="
  echo "========================================="

  for ENV in "PushCube-v1" "PickCube-v1" "OpenCabinetDoor-v1" "OpenCabinetDrawer-v1" "PegInsertionSide-v1" "PushT-v1" "UnitreeG1PlaceAppleInBowl-v1" "AnymalC-Reach-v1"
  do
    # Hyperparameters per task.
    # Outer loop uses longer rollouts and increased parallel environments.
    # Total timesteps per iter kept shorter than baseline for quick outer-loop iteration.
    # Batch size ratios between tasks are preserved from baselines.sh.
    #   Baseline reference batch sizes (num_envs * num_steps):
    #     PushCube/PickCube  = 4096*4   = 16,384  (1x)
    #     PegInsertion       = 2048*16  = 32,768  (2x)
    #     OpenCabinet        = 1024*16  = 16,384  (1x)
    #     PushT              = 4096*16  = 65,536  (4x)
    #     UnitreeG1          = 1024*32  = 32,768  (2x)
    #     AnymalC            = 4096*16  = 65,536  (4x)
    GAMMA_ARG=""
    GAE_LAMBDA_ARG=""
    if [ "${ENV}" == "PushCube-v1" ] || [ "${ENV}" == "PickCube-v1" ]; then
        TOTAL=3_000_000          # Baseline: 50M
        EVAL_STEPS=50
        NUM_ENVS=256             # Baseline: 4096
        NUM_STEPS=100            # Baseline: 4  (batch: 256*100=25,600  1x)
        UPDATE_EPOCHS=8
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
    elif [ "${ENV}" == "PushT-v1" ]; then
        TOTAL=24_000_000          # Baseline: 50M
        EVAL_STEPS=100
        NUM_ENVS=1024            # Baseline: 4096
        NUM_STEPS=128            # Baseline: 16 (batch: 1024*128=131,072  2x)
        UPDATE_EPOCHS=8
        GAMMA_ARG="--gamma=0.99"
    elif [ "${ENV}" == "UnitreeG1PlaceAppleInBowl-v1" ]; then
        TOTAL=12_000_000          # Baseline: 50M
        EVAL_STEPS=100
        NUM_ENVS=512             # Baseline: 1024
        NUM_STEPS=100            # Baseline: 32 (batch: 512*100=51,200  2x)
        UPDATE_EPOCHS=8
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
      ${EUREKA_ARG} \
      --rl_project_path="/home/robotics/naoki_workspace/codes/robotics_rl" \
      --track \
      --exp-name="${EXP_PREFIX}-${ENV}-${seed}"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "ERROR: ${ENV} seed=${seed} failed with exit code ${rc}"
        any_failed=1
    fi
    echo ""
  done

  echo "=== Completed seed ${seed} ==="
  echo ""
done

echo "========================================="
if [ $any_failed -ne 0 ]; then
    echo "=== WARNING: Some experiments FAILED ==="
else
    echo "=== All ${MODE_LABEL} experiments complete ==="
fi
echo "Seeds run: ${seeds[@]}"
echo "Check wandb for results (group: PPO-OuterLoop, tags: ${WANDB_TAG})"
echo "========================================="
exit $any_failed

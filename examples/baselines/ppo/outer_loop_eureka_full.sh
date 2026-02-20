#!/bin/bash
# PPO Outer Loop: Eureka Full Replacement Mode (LLM generates complete reward functions)
#
# This script implements the complete Eureka algorithm from the paper:
# - Generates K=4 reward function candidates per iteration
# - Trains each candidate and evaluates fitness (success_rate)
# - Selects best candidate and performs Reward Reflection
# - LLM rewrites entire reward function (not just weights)
#
# Comparison:
#   - outer_loop_vlm_params.sh: VLM + LLM (video analysis + weight optimization)
#   - outer_loop_eureka.sh: LLM only (weight optimization, Eureka-style)
#   - outer_loop_eureka_full.sh: LLM only (full reward function replacement, complete Eureka)
#
# GPU Setup:
#   GPU 0: NVIDIA RTX PRO 6000 Blackwell (96GB)
#   GPU 1: NVIDIA GeForce RTX 5090 (32GB)
#   Seeds are distributed across GPUs for parallel execution.
#
# Usage:
#   export OPENAI_API_KEY=sk-... && bash outer_loop_eureka_full.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

seeds=(9351 4796 1788)
OUTER_ITERS=5
WSEED=42
GPUS=(0 1)
LOG_DIR="logs/eureka_full_$(date +%Y%m%d_%H%M%S)"

if [ -z "${OPENAI_API_KEY}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "  export OPENAI_API_KEY=sk-..."
    exit 1
fi

mkdir -p "${LOG_DIR}"

echo "=== PPO Outer Loop: Eureka Full Replacement Mode ==="
echo "Outer iterations: ${OUTER_ITERS}"
echo "Weight seed: ${WSEED}"
echo "Seeds: ${seeds[@]}"
echo "GPUs: ${GPUS[@]}"
echo "Logs: ${LOG_DIR}"
echo ""

run_seed() {
    local seed=$1
    local gpu=$2
    export CUDA_VISIBLE_DEVICES=$gpu

    echo "[GPU ${gpu}] Starting seed ${seed}"

    for ENV in "PushCube-v1" "PickCube-v1" "OpenCabinetDoor-v1" "OpenCabinetDrawer-v1" "UnitreeG1PlaceAppleInBowl-v1" "AnymalC-Reach-v1" #"PegInsertionSide-v1" "PushT-v1"
    do
        # Hyperparameters per task
        # NUM_ENVS scaled up for RTX PRO 6000 (96GB) / RTX 5090 (32GB).
        # TOTAL timesteps kept the same as before for quick outer-loop iteration.
        # Batch size ratios between tasks are preserved from baselines.sh.
        #   Baseline reference batch sizes (num_envs * num_steps):
        #     PushCube/PickCube  = 4096*4   = 16,384  (1x)
        #     PegInsertion       = 2048*16  = 32,768  (2x)
        #     OpenCabinet        = 1024*16  = 16,384  (1x)
        #     PushT              = 4096*16  = 65,536  (4x)
        #     UnitreeG1          = 1024*32  = 32,768  (2x)
        #     AnymalC            = 4096*16  = 65,536  (4x)
        local TOTAL EVAL_STEPS NUM_ENVS NUM_STEPS UPDATE_EPOCHS GAMMA_ARG GAE_LAMBDA_ARG
        if [ "${ENV}" == "PushCube-v1" ] || [ "${ENV}" == "PickCube-v1" ]; then
            TOTAL=3_000_000          # Baseline: 50M
            EVAL_STEPS=50
            NUM_ENVS=2048            # Baseline: 4096  (was: 256)
            NUM_STEPS=100            # Baseline: 4  (batch: 2048*100=204,800  1x)
            UPDATE_EPOCHS=8
            GAMMA_ARG=""
            GAE_LAMBDA_ARG=""
        elif [ "${ENV}" == "PegInsertionSide-v1" ]; then
            TOTAL=24_000_000          # Baseline: 75M
            EVAL_STEPS=100
            NUM_ENVS=2048            # Baseline: 2048  (was: 1024)
            NUM_STEPS=64             # Baseline: 16 (batch: 2048*64=131,072  2x)
            UPDATE_EPOCHS=8
            GAMMA_ARG="--gamma=0.97"
            GAE_LAMBDA_ARG="--gae_lambda=0.95"
        elif [ "${ENV}" == "OpenCabinetDoor-v1" ] || [ "${ENV}" == "OpenCabinetDrawer-v1" ]; then
            TOTAL=6_000_000          # Baseline: 50M
            EVAL_STEPS=100
            NUM_ENVS=1024            # Baseline: 1024  (was: 256)
            NUM_STEPS=100            # Baseline: 16 (batch: 1024*100=102,400  1x)
            UPDATE_EPOCHS=8
            GAMMA_ARG=""
            GAE_LAMBDA_ARG=""
        elif [ "${ENV}" == "PushT-v1" ]; then
            TOTAL=24_000_000          # Baseline: 50M
            EVAL_STEPS=100
            NUM_ENVS=2048            # Baseline: 4096  (was: 1024)
            NUM_STEPS=128            # Baseline: 16 (batch: 2048*128=262,144  2x)
            UPDATE_EPOCHS=8
            GAMMA_ARG="--gamma=0.99"
            GAE_LAMBDA_ARG=""
        elif [ "${ENV}" == "UnitreeG1PlaceAppleInBowl-v1" ]; then
            TOTAL=12_000_000          # Baseline: 50M
            EVAL_STEPS=100
            NUM_ENVS=1024            # Baseline: 1024  (was: 512)
            NUM_STEPS=100            # Baseline: 32 (batch: 1024*100=102,400  2x)
            UPDATE_EPOCHS=8
            GAMMA_ARG=""
            GAE_LAMBDA_ARG=""
        elif [ "${ENV}" == "AnymalC-Reach-v1" ]; then
            TOTAL=6_000_000          # Baseline: 50M
            EVAL_STEPS=200
            NUM_ENVS=2048            # Baseline: 4096  (was: 512)
            NUM_STEPS=200            # Baseline: 16 (batch: 2048*200=409,600 4x)
            UPDATE_EPOCHS=8
            GAMMA_ARG="--gamma=0.99"
            GAE_LAMBDA_ARG="--gae_lambda=0.95"
        else
            TOTAL=3_000_000
            EVAL_STEPS=50
            NUM_ENVS=2048
            NUM_STEPS=100
            UPDATE_EPOCHS=8
            GAMMA_ARG=""
            GAE_LAMBDA_ARG=""
        fi

        echo "[GPU ${gpu}] === ${ENV} seed=${seed} NUM_ENVS=${NUM_ENVS} ==="
        python ppo_outer_loop_full.py \
          --env_id="${ENV}" \
          --seed=${seed} \
          --num_envs=${NUM_ENVS} \
          --num_steps=${NUM_STEPS} \
          --update_epochs=${UPDATE_EPOCHS} \
          --num_minibatches=32 \
          --num_eval_envs=64 \
          --num_eval_steps=${EVAL_STEPS} \
          --num_outer_iters=${OUTER_ITERS} \
          --total_timesteps_per_iter=${TOTAL} \
          --weight_seed=${WSEED} \
          --num_reward_candidates=4 \
          --enable_reward_reflection \
          ${GAMMA_ARG} \
          ${GAE_LAMBDA_ARG} \
          --eureka_mode \
          --rl_project_path="/home/robotics/naoki_workspace/codes/robotics_rl" \
          --track \
          --exp-name="ppo-eureka-full-${ENV}-${seed}"
        echo ""
    done

    echo "[GPU ${gpu}] === Completed seed ${seed} ==="
}

# Distribute seeds across GPUs in parallel
idx=0
while [ $idx -lt ${#seeds[@]} ]; do
    pids=()
    for gpu_idx in ${!GPUS[@]}; do
        seed_idx=$((idx + gpu_idx))
        if [ $seed_idx -lt ${#seeds[@]} ]; then
            seed=${seeds[$seed_idx]}
            gpu=${GPUS[$gpu_idx]}
            log_file="${LOG_DIR}/seed_${seed}_gpu${gpu}.log"
            run_seed ${seed} ${gpu} > "${log_file}" 2>&1 &
            pids+=($!)
            echo "Started seed ${seed} on GPU ${gpu} (PID: $!) → ${log_file}"
        fi
    done
    echo "Waiting for ${#pids[@]} parallel jobs..."
    for pid in ${pids[@]}; do
        wait $pid
        echo "  PID $pid finished (exit code: $?)"
    done
    idx=$((idx + ${#GPUS[@]}))
done

echo "========================================="
echo "=== All Eureka full replacement experiments complete ==="
echo "Seeds run: ${seeds[@]}"
echo "Logs: ${LOG_DIR}"
echo "Check wandb for results (group: PPO-OuterLoop, tags: eureka-full)"
echo "========================================="

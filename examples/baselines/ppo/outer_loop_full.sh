#!/bin/bash
# PPO Outer Loop: Unified script for VLM+LLM and Eureka (LLM-only) modes
#
# Modes:
#   vlm                  - VLM+LLM Full Replacement: default VLM mode.
#   vlm_failureselection - VLM+LLM Full Replacement with failure-focused episode selection.
#   eureka               - Eureka Full Replacement: LLM-only, no VLM video analysis.
#                          Ablation counterpart of vlm mode for fair comparison.
#
# Common behavior:
# - Generates K=4 reward function candidates per iteration
# - Trains K candidates in parallel across GPUs (--gpus flag)
# - Performs Reward Reflection for next iteration
# - LLM rewrites entire reward function (not just weights)
#
# Cross-experiment resume (--resume_from_counterpart):
#   When CROSS_RESUME=1, automatically finds the counterpart experiment's
#   latest run for each env and resumes from its iter 0.
#   e.g., eureka mode finds the latest vlm run, vlm mode finds the latest eureka run.
#
# Parallelization: K candidates are trained in parallel across GPUs.
#   With K=4 and 2 GPUs, each outer iteration runs 2 batches of 2.
#   Seeds and envs are processed sequentially.
#
# GPU Setup:
#   GPU 0: NVIDIA RTX PRO 6000 Blackwell (96GB)
#   GPU 1: NVIDIA GeForce RTX 5090 (32GB)
#
# Usage:
#   export OPENAI_API_KEY=sk-... && bash outer_loop_full.sh vlm
#   export OPENAI_API_KEY=sk-... && bash outer_loop_full.sh vlm_failureselection
#   export OPENAI_API_KEY=sk-... && bash outer_loop_full.sh eureka

# --- Mode selection ---
MODE="${1:-vlm_failureselection}"
if [ "${MODE}" != "vlm_failureselection" ] && [ "${MODE}" != "eureka" ]; then
    echo "Usage: bash outer_loop_full.sh [vlm_failureselection|eureka]"
    echo "  vlm_failureselection : VLM+LLM mode with failure episode selection (default)"
    echo "  eureka               : LLM-only mode (no VLM)"
    exit 1
fi

# --- Configuration ---
seeds=(9351) # 4796 1788
OUTER_ITERS=${OUTER_ITERS_OVERRIDE:-5}
WSEED=42
NUM_CANDIDATES=${NUM_CANDIDATES_OVERRIDE:-16}
PROCS_PER_GPU=${PROCS_PER_GPU_OVERRIDE:-8}  # Number of concurrent candidates per GPU
# Auto-detect free GPUs (memory usage < 1GB = idle), repeat each GPU PROCS_PER_GPU times
detect_free_gpus() {
    local raw_gpus=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' '$2 < 1024 {printf sep $1; sep=","}')
    if [ -z "${raw_gpus}" ]; then
        echo "ERROR: No free GPUs detected (all have >1GB memory usage)."
        return 1
    fi
    # Expand: e.g. "0,1,2" with PROCS_PER_GPU=2 -> "0,0,1,1,2,2"
    GPUS=$(echo "${raw_gpus}" | tr ',' '\n' | awk -v n="${PROCS_PER_GPU}" '{for(i=0;i<n;i++) print}' | paste -sd,)
    echo "[GPU] Free GPUs: ${raw_gpus} (x${PROCS_PER_GPU} = ${GPUS})"
}
# If GPUS_OVERRIDE is set, use it directly; otherwise auto-detect
if [ -n "${GPUS_OVERRIDE}" ]; then
    GPUS="${GPUS_OVERRIDE}"
    echo "[GPU] Using override: ${GPUS}"
else
    detect_free_gpus || exit 1
fi
CROSS_RESUME=0  # Set to 1 to auto-resume from counterpart's iter 0
EARLY_STOP_SUCCESS=1 # Set to 1 to stop when success_rate >= 1.0
VLM_REWARD_PLOT=0     # Set to 1 to send per-step reward plot to VLM

if [ "${MODE}" == "eureka" ]; then
    EUREKA_ARG="--eureka_mode"
    EXP_PREFIX="ppo-eureka-full"
    MODE_LABEL="Eureka Full Replacement (NO VLM)"
    WANDB_TAG="eureka-full"
    VLM_SELECTION_ARG=""
    VLM_CATEGORY_ARG=""
else
    EUREKA_ARG=""
    EXP_PREFIX="ppo-vlm-full-failureselection"
    MODE_LABEL="VLM+LLM Full Replacement (failure selection)"
    WANDB_TAG="vlm-full-failureselection"
    VLM_SELECTION_ARG="--vlm_episode_selection=categorized"
    VLM_CATEGORY_ARG="--vlm_category_focus=failure"
fi

LOG_DIR="logs/${MODE}_full_$(date +%Y%m%d_%H%M%S)"

if [ -z "${OPENAI_API_KEY}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "  export OPENAI_API_KEY=sk-..."
    exit 1
fi

mkdir -p "${LOG_DIR}"

echo "=== PPO Outer Loop: ${MODE_LABEL} ==="
echo "Parallelization: K-candidate parallel (--gpus=${GPUS})"
echo "Outer iterations: ${OUTER_ITERS}"
echo "Weight seed: ${WSEED}"
echo "Seeds: ${seeds[@]}"
echo "GPUs: ${GPUS}"
echo "Cross-resume: ${CROSS_RESUME}"
if [ -n "${VLM_SELECTION_ARG}" ]; then
    echo "VLM episode selection: categorized"
    echo "VLM category focus: failure"
fi
echo "Logs: ${LOG_DIR}"
echo ""

any_failed=0
for seed in "${seeds[@]}"; do
    # ENV list: use ENVS_OVERRIDE if set, otherwise default list
    DEFAULT_ENVS=("PegInsertionSide-v1" "PushT-v1" "UnitreeG1PlaceAppleInBowl-v1" "OpenCabinetDoor-v1" "RotateValveLevel0-v1" "UnitreeG1TransportBox-v1")
    if [ -n "${ENVS_OVERRIDE}" ]; then
        IFS=',' read -ra ENV_LIST <<< "${ENVS_OVERRIDE}"
    else
        ENV_LIST=("${DEFAULT_ENVS[@]}")
    fi
    for ENV in "${ENV_LIST[@]}"
    do
        # Re-detect free GPUs before each task (skip if GPUS_OVERRIDE is set)
        if [ -z "${GPUS_OVERRIDE}" ]; then
            detect_free_gpus || { echo "SKIP ${ENV}: no free GPUs"; any_failed=1; continue; }
        fi

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
        TOTAL=""
        EVAL_STEPS=""
        NUM_ENVS=""
        NUM_STEPS=""
        UPDATE_EPOCHS=""
        GAMMA_ARG=""
        GAE_LAMBDA_ARG=""
        if [ "${ENV}" == "PushCube-v1" ] || [ "${ENV}" == "PickCube-v1" ]; then
            TOTAL=50_000_000      # Baseline: 50M
            EVAL_STEPS=50
            NUM_ENVS=4096            # Baseline: 4096
            NUM_STEPS=4              # Baseline: 4  (batch: 4096*4=16,384)
            UPDATE_EPOCHS=8
        elif [ "${ENV}" == "PegInsertionSide-v1" ]; then
            TOTAL=75_000_000          # Baseline: 75M
            EVAL_STEPS=100
            NUM_ENVS=2048            # Baseline: 2048
            NUM_STEPS=16             # Baseline: 16 (batch: 2048*16=32,768)
            UPDATE_EPOCHS=8
            GAMMA_ARG="--gamma=0.97"
            GAE_LAMBDA_ARG="--gae_lambda=0.95"
        elif [ "${ENV}" == "OpenCabinetDoor-v1" ] || [ "${ENV}" == "OpenCabinetDrawer-v1" ]; then
            TOTAL=50_000_000          # Baseline: 50M
            EVAL_STEPS=100
            NUM_ENVS=1024            # Baseline: 1024
            NUM_STEPS=16             # Baseline: 16 (batch: 1024*16=16,384)
            UPDATE_EPOCHS=8
        elif [ "${ENV}" == "PushT-v1" ]; then
            TOTAL=50_000_000          # Baseline: 50M
            EVAL_STEPS=100
            NUM_ENVS=4096            # Baseline: 4096
            NUM_STEPS=16             # Baseline: 16 (batch: 4096*16=65,536)
            UPDATE_EPOCHS=8
            GAMMA_ARG="--gamma=0.99"
        elif [ "${ENV}" == "UnitreeG1PlaceAppleInBowl-v1" ]; then
            TOTAL=50_000_000          # Baseline: 50M
            EVAL_STEPS=100
            NUM_ENVS=1024            # Baseline: 1024
            NUM_STEPS=32             # Baseline: 32 (batch: 1024*32=32,768)
            UPDATE_EPOCHS=8
        elif [ "${ENV}" == "AnymalC-Reach-v1" ]; then
            TOTAL=50_000_000         # Baseline: 50M
            EVAL_STEPS=200
            NUM_ENVS=4096            # Baseline: 4096
            NUM_STEPS=16             # Baseline: 16 (batch: 4096*16=65,536)
            UPDATE_EPOCHS=8
            GAMMA_ARG="--gamma=0.99"
            GAE_LAMBDA_ARG="--gae_lambda=0.95"
        elif [ "${ENV}" == "RotateValveLevel0-v1" ]; then
            TOTAL=50_000_000         # Ref: TriFingerRotateCube 50M
            EVAL_STEPS=80            # max_episode_steps=80
            NUM_ENVS=128             # Dexterity tasks: smaller envs
            NUM_STEPS=250            # Ref: TriFingerRotateCube 250
            UPDATE_EPOCHS=8
        elif [ "${ENV}" == "UnitreeG1TransportBox-v1" ]; then
            TOTAL=100_000_000        # Baseline: 100M
            EVAL_STEPS=100           # max_episode_steps=100
            NUM_ENVS=1024            # Baseline: 1024
            NUM_STEPS=32             # Baseline: 32 (batch: 1024*32=32,768)
            UPDATE_EPOCHS=8
        else
            TOTAL=3_000_000
            EVAL_STEPS=50
            NUM_ENVS=256
            NUM_STEPS=100
            UPDATE_EPOCHS=8
        fi

        # Allow overriding TOTAL timesteps (e.g. for smoke tests)
        if [ -n "${TOTAL_OVERRIDE}" ]; then
            TOTAL=${TOTAL_OVERRIDE}
        fi

        log_file="${LOG_DIR}/seed_${seed}_${ENV}.log"
        CROSS_RESUME_ARG=""
        if [ "${CROSS_RESUME}" -eq 1 ]; then
            CROSS_RESUME_ARG="--resume_from_counterpart"
        fi
        EARLY_STOP_ARG=""
        if [ "${EARLY_STOP_SUCCESS}" -eq 1 ]; then
            EARLY_STOP_ARG="--early_stop_success"
        fi
        VLM_REWARD_PLOT_ARG=""
        if [ "${VLM_REWARD_PLOT}" -eq 1 ]; then
            VLM_REWARD_PLOT_ARG="--vlm_reward_plot"
        fi

        echo "=== ${ENV} seed=${seed} NUM_ENVS=${NUM_ENVS} ==="
        python -u ppo_outer_loop_full.py \
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
          --num_reward_candidates=${NUM_CANDIDATES} \
          --enable_reward_reflection \
          --gpus="${GPUS}" \
          ${GAMMA_ARG} \
          ${GAE_LAMBDA_ARG} \
          ${EUREKA_ARG} \
          ${VLM_SELECTION_ARG} \
          ${VLM_CATEGORY_ARG} \
          --rl_project_path="/home/nwake/codes/RL_project" \
          --track \
          --exp-name="${EXP_PREFIX}-${ENV}-${seed}" \
          ${CROSS_RESUME_ARG} \
          ${EARLY_STOP_ARG} \
          ${VLM_REWARD_PLOT_ARG} \
          2>&1 | tee "${log_file}"
        rc=${PIPESTATUS[0]}
        if [ $rc -ne 0 ]; then
            echo "ERROR: ${ENV} seed=${seed} failed with exit code ${rc}"
            any_failed=1
        fi
    done
done

echo "========================================="
if [ $any_failed -ne 0 ]; then
    echo "=== WARNING: Some experiments FAILED ==="
else
    echo "=== All ${MODE_LABEL} experiments complete ==="
fi
echo "Seeds run: ${seeds[@]}"
echo "Logs: ${LOG_DIR}"
echo "Check wandb for results (group: PPO-OuterLoop, tags: ${WANDB_TAG})"
echo "========================================="
exit $any_failed

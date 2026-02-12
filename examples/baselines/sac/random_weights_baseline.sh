#!/bin/bash
# Random Weights Baseline + VLM/LLM Comparison
# Purpose: Verify that VLM/LLM tuning outperforms random weight initialization
#
# Runs 2 experiments per task:
#   1) Random weights + no VLM/LLM (pure random baseline)
#   2) Random weights + VLM/LLM tuning (tests if LLM can improve from random start)
#
# Note: Always runs in FAST mode (num_envs=32, training_freq=128) for speed
#
# Usage:
#   export OPENAI_API_KEY=sk-... && bash random_weights_baseline.sh

seed=9351
SEGMENTS=10
WSEED=42  # Single random seed for reproducibility

echo "=== Random Weights Experiments ==="
echo "Weight seed: ${WSEED}"
echo "Comparing: (1) random+noVLM vs (2) random+withVLM"
echo ""

for ENV in "PushCube-v1" "PickCube-v1"
do
    # Set total timesteps based on task
    if [ "${ENV}" == "PickCube-v1" ]; then
        TOTAL=250_000
    else
        TOTAL=200_000
    fi

    echo "=== ${ENV} with random weights (total_timesteps=${TOTAL}) ==="

    # Generate random weights with VERY wide range (extreme baseline)
    # Component weights: [0.01, 10.0]  (100x range!)
    # Success weight: [0.1, 20.0]     (200x range!)
    WEIGHTS=$(python3 -c "
import random
random.seed(${WSEED})

if '${ENV}' == 'PushCube-v1':
    w_reach = random.uniform(0.01, 10.0)
    w_push = random.uniform(0.01, 10.0)
    w_z_keep = random.uniform(0.01, 10.0)
    w_success = random.uniform(0.1, 20.0)
    print(f'{w_reach:.3f} {w_push:.3f} {w_z_keep:.3f} {w_success:.3f}')
elif '${ENV}' == 'PickCube-v1':
    w_reach = random.uniform(0.01, 10.0)
    w_grasp = random.uniform(0.01, 10.0)
    w_place = random.uniform(0.01, 10.0)
    w_static = random.uniform(0.01, 10.0)
    w_success = random.uniform(0.1, 20.0)
    print(f'{w_reach:.3f} {w_grasp:.3f} {w_place:.3f} {w_static:.3f} {w_success:.3f}')
")

    # Parse weights into array
    read -ra W_ARRAY <<< "${WEIGHTS}"

    if [ "${ENV}" == "PushCube-v1" ]; then
        W_REACH=${W_ARRAY[0]}
        W_PUSH=${W_ARRAY[1]}
        W_Z_KEEP=${W_ARRAY[2]}
        W_SUCCESS=${W_ARRAY[3]}
        echo "  w_reach=${W_REACH}, w_push=${W_PUSH}, w_z_keep=${W_Z_KEEP}, w_success=${W_SUCCESS}"
    else
        W_REACH=${W_ARRAY[0]}
        W_GRASP=${W_ARRAY[1]}
        W_PLACE=${W_ARRAY[2]}
        W_STATIC=${W_ARRAY[3]}
        W_SUCCESS=${W_ARRAY[4]}
        echo "  w_reach=${W_REACH}, w_grasp=${W_GRASP}, w_place=${W_PLACE}, w_static=${W_STATIC}, w_success=${W_SUCCESS}"
    fi

    COMMON="--env_id=${ENV} --seed=${seed} \
      --num_envs=32 --training_freq=128 --num_eval_envs=16 \
      --total_timesteps=${TOTAL} --num_segments=${SEGMENTS} --track"

    # Note: RewardWrapper __init__ accepts 'weights' dict parameter
    # We need to pass it via command line, but sac_iterative_debug.py doesn't support it yet
    # Workaround: Create a temporary modified version or add CLI support
    # For now, create a wrapper script that modifies TASK_DEFAULTS

    # Create temp config file (shared by both noVLM and withVLM runs)
    TMP_CONFIG="/tmp/random_weights_${ENV}_${WSEED}.json"
    if [ "${ENV}" == "PushCube-v1" ]; then
        cat > ${TMP_CONFIG} << EOF
{
  "w_reach": ${W_REACH},
  "w_push": ${W_PUSH},
  "w_z_keep": ${W_Z_KEEP},
  "w_success": ${W_SUCCESS}
}
EOF
    else
        cat > ${TMP_CONFIG} << EOF
{
  "w_reach": ${W_REACH},
  "w_grasp": ${W_GRASP},
  "w_place": ${W_PLACE},
  "w_static": ${W_STATIC},
  "w_success": ${W_SUCCESS}
}
EOF
    fi

    # 1) Random weights WITHOUT VLM/LLM
    echo "  [1/2] Running without VLM/LLM..."
    python sac_iterative_debug.py ${COMMON} \
      --skip_vlm_llm \
      --initial_weights_file=${TMP_CONFIG} \
      --exp-name="sac-random-noVLM-ws${WSEED}-${ENV}-${seed}"

    # 2) Random weights WITH VLM/LLM (same starting point)
    if [ -z "${OPENAI_API_KEY}" ]; then
        echo "  [2/2] SKIPPED: OPENAI_API_KEY not set"
    else
        echo "  [2/2] Running with VLM/LLM (same initial weights)..."
        python sac_iterative_debug.py ${COMMON} \
          --vlm_reward_plot \
          --critic_warmup_steps=5000 \
          --initial_weights_file=${TMP_CONFIG} \
          --exp-name="sac-random-withVLM-ws${WSEED}-${ENV}-${seed}"
    fi

    rm ${TMP_CONFIG}
    echo ""
done

echo "=== Random weights experiments complete ==="
echo ""
echo "Compare wandb results:"
echo "  PushCube:"
echo "    - sac-random-noVLM-ws${WSEED}-PushCube-v1-${seed}    (random weights, no tuning)"
echo "    - sac-random-withVLM-ws${WSEED}-PushCube-v1-${seed}  (random weights, with VLM/LLM)"
echo "  PickCube:"
echo "    - sac-random-noVLM-ws${WSEED}-PickCube-v1-${seed}    (random weights, no tuning)"
echo "    - sac-random-withVLM-ws${WSEED}-PickCube-v1-${seed}  (random weights, with VLM/LLM)"
echo ""
echo "Also compare with default weights experiments:"
echo "  - sac-debug-noVLM-*-${seed}           (default weights, no tuning)"
echo "  - sac-debug-withVLM-recompute-*-${seed} (default weights, with VLM/LLM)"

#!/bin/bash
# Debug: test reward recomputation vs buffer clearing.
#
# Modes:
#   MODE=recompute (default) - VLM/LLM active, buffer rewards recomputed on weight change (no clear)
#   MODE=clear               - same as sac_iterative.py (VLM/LLM active, buffer cleared on weight change)
#   MODE=clear_only          - no VLM/LLM, buffer cleared at segment boundaries (isolate clear impact)
#
# Usage:
#   export OPENAI_API_KEY=sk-... && bash baselines_iterative_debug.sh
#   MODE=clear_only bash baselines_iterative_debug.sh

seeds=(9351) # 4796 1788
MODE="${MODE:-recompute}"

case "$MODE" in
  recompute)
    echo "Running WITH VLM/LLM, reward recomputation (no buffer clear)"
    if [ -z "${OPENAI_API_KEY}" ]; then
        echo "ERROR: OPENAI_API_KEY is not set."
        exit 1
    fi
    EXTRA_FLAGS="--vlm_reward_plot"
    EXP_SUFFIX="recompute"
    ;;
  clear)
    echo "Running WITH VLM/LLM, buffer clear (original behavior via sac_iterative.py)"
    echo "Use baselines_iterative.sh instead for this mode."
    exit 1
    ;;
  clear_only)
    echo "Running WITHOUT VLM/LLM, buffer cleared at segment boundaries"
    EXTRA_FLAGS="--skip_vlm_llm --clear_buffer_at_segment"
    EXP_SUFFIX="clearonly"
    ;;
  *)
    echo "Unknown MODE=$MODE. Use: recompute, clear, clear_only"
    exit 1
    ;;
esac

### PickCube ###
for seed in ${seeds[@]}
do
  python sac_iterative_debug.py --env_id="PickCube-v1" --seed=${seed} \
    --num_envs=16 --num-steps=50 --num_eval_envs=16 \
    --total_timesteps=500_000 \
    --num_segments=20 \
    --exp-name="sac-iterative-debug-${EXP_SUFFIX}-PickCube-v1-state-${seed}" \
    --track \
    ${EXTRA_FLAGS}
done

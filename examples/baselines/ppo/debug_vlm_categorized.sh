#!/bin/bash
# Debug script: test VLM categorized episode selection on easy tasks.
#
# Minimal config: K=2 candidates, N=2 outer iters, tiny training budget.
# Purpose: verify that env_last_outcomes, vlm_final video dir, and
#          categorized frames (success/near_miss/failure panels) work correctly.
#
# Check outputs:
#   runs/outer-loop_full/<env>/*/debug_html/iter_*_vlm.html  <- VLM frames & response
#   runs/outer-loop_full/<env>/*/videos/iter_*_vlm_final/    <- dedicated VLM video
#
# Usage:
#   export OPENAI_API_KEY=sk-... && bash debug_vlm_categorized.sh
#   export OPENAI_API_KEY=sk-... && bash debug_vlm_categorized.sh random   # compare with random mode

set -euo pipefail

VLM_MODE="${1:-categorized}"  # "categorized" or "random"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "  export OPENAI_API_KEY=sk-..."
    exit 1
fi

echo "=== Debug VLM Categorized Episode Selection ==="
echo "VLM episode selection: ${VLM_MODE}"
echo ""

for ENV in "PushCube-v1" "PickCube-v1"; do
    echo "----------------------------------------"
    echo "[${ENV}] Starting (K=2, N=2, 5M steps)"
    echo "----------------------------------------"

    python -u ppo_outer_loop_full.py \
      --env_id="${ENV}" \
      --seed=42 \
      --num_envs=1024 \
      --num_steps=4 \
      --update_epochs=4 \
      --num_minibatches=8 \
      --num_eval_envs=16 \
      --num_eval_steps=50 \
      --num_outer_iters=2 \
      --total_timesteps_per_iter=5_000_000 \
      --num_reward_candidates=2 \
      --enable_reward_reflection \
      --vlm_episode_selection="${VLM_MODE}" \
      --rl_project_path="/home/robotics/naoki_workspace/codes/robotics_rl" \
      --exp-name="debug-vlm-${VLM_MODE}-${ENV}" \
      2>&1 | tee "logs/debug_vlm_${VLM_MODE}_${ENV}_$(date +%Y%m%d_%H%M%S).log"

    echo "[${ENV}] Done."
    echo ""
done

echo "=== Debug complete ==="
echo "Check debug_html/ for VLM frames and videos/iter_*_vlm_final/ for dedicated VLM videos."

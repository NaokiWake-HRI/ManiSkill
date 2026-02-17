#!/bin/bash
# Quick test: K=2 candidates, N=2 iterations, PushCube, LLM-only

if [ -z "${OPENAI_API_KEY}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "  export OPENAI_API_KEY=sk-..."
    exit 1
fi

echo "=== Eureka Full Replacement Test (K=2, N=2) ==="
echo "Task: PushCube-v1"
echo "Seed: 9351"
echo ""

python ppo_outer_loop_full.py \
  --env_id="PushCube-v1" \
  --seed=9351 \
  --num_envs=256 \
  --num_steps=100 \
  --update_epochs=8 \
  --num_minibatches=32 \
  --num_eval_envs=16 \
  --num_eval_steps=50 \
  --num_outer_iters=2 \
  --total_timesteps_per_iter=300_000 \
  --weight_seed=42 \
  --num_reward_candidates=2 \
  --enable_reward_reflection \
  --eureka_mode \
  --track \
  --exp-name="test-eureka-full-PushCube-K2N2"

echo ""
echo "=== Test complete ==="
echo "Check: runs/eureka_full/PushCube-v1/*/outer_loop_history.json"

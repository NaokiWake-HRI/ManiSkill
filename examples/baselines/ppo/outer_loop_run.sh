#!/bin/bash
# PPO Outer Loop: VLM/LLM-guided Reward Weight Optimization
#
# Starts from random weights, runs full PPO training per iteration,
# then uses VLM/LLM to suggest improved weights for the next iteration.
# Iteration 1 (random weights) serves as the baseline.
#
# Usage:
#   export OPENAI_API_KEY=sk-... && bash outer_loop_run.sh

seed=9351
OUTER_ITERS=5
WSEED=42

if [ -z "${OPENAI_API_KEY}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "  export OPENAI_API_KEY=sk-..."
    exit 1
fi

echo "=== PPO Outer Loop ==="
echo "Outer iterations: ${OUTER_ITERS}"
echo "Weight seed: ${WSEED}"
echo ""

for ENV in "PushCube-v1" "PickCube-v1" "OpenCabinetDoor-v1" "OpenCabinetDrawer-v1" "PegInsertionSide-v1" "PushT-v1" "UnitreeG1PlaceAppleInBowl-v1" "AnymalC-Reach-v1"
do
    # Hyperparameters per task.
    # Outer loop uses longer rollouts (num_steps >= episode length) to compensate
    # for fewer total timesteps (3M/iter vs 50M baseline). Batch size ratios
    # between tasks are preserved from baselines.sh.
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
        TOTAL=9_000_000          # Baseline: 75M
        EVAL_STEPS=100
        NUM_ENVS=512             # Baseline: 2048
        NUM_STEPS=100            # Baseline: 16 (batch: 512*100=51,200  2x)
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
        TOTAL=6_000_000          # Baseline: 50M
        EVAL_STEPS=100
        NUM_ENVS=512             # Baseline: 4096
        NUM_STEPS=200            # Baseline: 16 (batch: 512*200=102,400 4x)
        UPDATE_EPOCHS=8
        GAMMA_ARG="--gamma=0.99"
        GAE_LAMBDA_ARG=""
    elif [ "${ENV}" == "UnitreeG1PlaceAppleInBowl-v1" ]; then
        TOTAL=6_000_000          # Baseline: 50M
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
    elif [ "${ENV}" == "OpenCabinetDrawer-v1" ]; then
        TOTAL=6_000_000          # Baseline: 50M
        EVAL_STEPS=100
        NUM_ENVS=256             # Baseline: 1024
        NUM_STEPS=100            # Baseline: 16 (batch: 256*100=25,600  1x)
        UPDATE_EPOCHS=8
        GAMMA_ARG=""
        GAE_LAMBDA_ARG=""
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
      --vlm_reward_plot \
      --track \
      --exp-name="ppo-outer-loop-${ENV}-${seed}"
    echo ""
done

echo "=== Outer loop experiments complete ==="
echo "Check wandb for results (group: PPO-OuterLoop)"

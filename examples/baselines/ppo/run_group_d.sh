#!/bin/bash
# Group D: GPU 5 — RotateValve smoke test (video export check)
export GPUS_OVERRIDE="5,5"
export ENVS_OVERRIDE="RotateValveLevel0-v1"
export NUM_CANDIDATES_OVERRIDE=2
export OUTER_ITERS_OVERRIDE=2
export TOTAL_OVERRIDE=500_000
exec bash outer_loop_full.sh "$@"

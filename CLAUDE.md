# ManiSkill Project

## Python Environment
- Use the project-local venv: `/home/nwake/codes/ManiSkill/.venv/bin/python`
- Activate with: `source /home/nwake/codes/ManiSkill/.venv/bin/activate`

## Outer-Loop (VLM/LLM-guided Reward Weight Optimization)

### Adding a New Task to the Outer-Loop

When adding a new task to the outer-loop system, the following files **must all be updated**:

1. **`examples/baselines/ppo/reward_wrapper.py`**
   - Add default weights to `TASK_DEFAULTS` dict (key = task ID, e.g. `"MyTask"`)
   - Add task ID mapping in `_resolve_task_id()` if the env_id pattern doesn't match an existing key
   - Implement `_compute_my_task(self, info)` method in `RewardWrapper`
   - Register the compute function in `__init__`'s `self._compute_fn` dict
   - Include all reward components in `self._last_breakdown` (this feeds into LLM's component analysis)

2. **`examples/baselines/ppo/ppo_outer_loop.py`**
   - Add task description to `_llm_task_descs` dict (~L990). This is critical — without it, the LLM gets a wrong generic fallback description ("robot hand must achieve stable fingertip contact with a cube")
   - The `reward_fn_source` is auto-extracted via `inspect.getsource()`, no manual action needed

3. **`examples/baselines/ppo/outer_loop_run.sh`**
   - Add task-specific hyperparameters (TOTAL, NUM_ENVS, NUM_STEPS, etc.) in the if/elif chain
   - Add the env name to the `for ENV in ...` loop

4. **`examples/baselines/ppo/plot_outer_loop_summary.py`**
   - Add the task env_id (e.g. `"MyTask-v1"`) to the `TASKS` list

### Key Design Principles
- All reward shaping is done through **weighted additive components** (`w_xxx * component`). The LLM tunes these weights.
- Avoid conditional overrides (e.g. `if condition: reward = X`) that bypass the weight system. Instead, convert them into binary/continuous components with their own weight (e.g. `w_above_bowl * (condition).float()`).
- The LLM sees: task description, reward function source code, current/initial weights, training curves, per-component reward breakdowns, and VLM video analysis comments.
- Reward normalization (`_norm_scale()`) keeps reward magnitude stable when weights change.

### LLM Modes
- **params-only (default)**: `--enable_function_code=False`. LLM can only adjust existing weight values. Current experimental focus.
- **function_code (Eureka-style)**: `--enable_function_code=True`. LLM can generate custom reward code that is added on top of the weighted components.
  - **NOT YET IMPLEMENTED in ppo_outer_loop.py**: The suggestion handler (L1173) only processes `type == "params"`. `function_code` responses are logged but ignored. Before enabling this mode, implement the custom code application logic.

### Known Pitfalls (past bugs)
- `_llm_task_descs` must cover ALL tasks. Missing entries fall back to a wrong generic description from `episode_collector.py` ("robot hand must achieve stable fingertip contact with a cube").
- `reward_fn_source` is extracted via `inspect.getsource(getattr(RewardWrapper, method_name))`. Do NOT use instance variables from `run_ppo_training()` (e.g. `reward_wrapper_train`) — they are local to that function and not accessible from the main block.
- `_reward_method_map` in `ppo_outer_loop.py` must be kept in sync with the `_compute_fn` dict in `reward_wrapper.py`. When adding a new task, update both.

# ManiSkill Project

## Python Environment
- Use the project-local venv: `/home/nwake/codes/ManiSkill/.venv/bin/python`
- Activate with: `source /home/nwake/codes/ManiSkill/.venv/bin/activate`
- Bash でスクリプト実行時は常に `.venv` の python を使うこと: `/home/nwake/codes/ManiSkill/.venv/bin/python <script>`

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

---

## Eureka Full Replacement Mode (2026-02-17)

Eureka paper (Ma et al., 2023) のフルアルゴリズム実装。LLMが報酬関数の**重みだけでなく関数自体を生成・置換**する。

### 新規ファイル（既存コードへの影響ゼロ）

| ファイル | 役割 |
|---------|------|
| `examples/baselines/ppo/reward_wrapper_dynamic.py` | 動的報酬関数の差し替えインフラ。`set_custom_function(code=...)` でLLM生成コードをコンパイル・実行。ランタイムエラー時はYAMLフォールバック |
| `examples/baselines/ppo/ppo_outer_loop_full.py` | Eurekaアルゴリズム本体。K候補生成→並列訓練→Fitness評価→Reward Reflection→次イテレーション |
| `examples/baselines/ppo/outer_loop_eureka_full.sh` | 本実験用スクリプト。ログ先: `runs/eureka_full/` |
| `examples/baselines/ppo/test_eureka_full.sh` | 簡易テスト用 (K=2, N=2, PushCube) |

### RL_Project側の変更
- `experiments/callbacks/episode_collector.py`: `eureka_full_replacement=True` 時のLLMプロンプト追加（条件分岐で保護、既存モードに影響なし）

### アルゴリズムの流れ

```
Iteration 0:
  参照関数 = スパースリワード (success bonus only)
  K=4 候補を LLM で生成 (seed固定で再現性)
  各候補で PPO 訓練 → fitness = success_rate で評価
  最良候補を選択

Iteration 1+:
  前回ベスト候補のコード + 統計情報 + Reward Reflection を LLM に渡す
  K=4 候補を生成（seedなし）
  VLM 分析（有効時のみ）
  訓練 → 評価 → 選択 → Reflection
```

### STATE_ACCESS_DOCS の管理

`ppo_outer_loop_full.py` の `_state_access_docs` dict に各タスクの利用可能な属性を手動で記述。LLMはこれを参照してコードを生成する。

**環境実装と不一致するとLLM生成コードが100%失敗する。新タスク追加・環境変更時は必ず実装と照合すること。**

各タスクの正しい属性名:

| タスク | オブジェクト | TCP | ゴール |
|--------|-------------|-----|--------|
| PushCube | `base.obj.pose.p` | `base.agent.tcp.pose.p` | `base.goal_region.pose.p` |
| PickCube | `base.cube.pose.p` | `base.agent.tcp_pose.p` | `base.goal_site.pose.p` |
| OpenCabinetDoor/Drawer | - | `base.agent.tcp.pose.p` | `info["handle_link_pos"]` |
| PegInsertionSide | `base.peg.pose.p` | `base.agent.tcp.pose.p` | `base.box_hole_pose.p` |
| PushT | `base.tee.pose.p` | `base.agent.tcp.pose.p` | `base.goal_tee.pose.p` |
| AnymalC | - | `base.agent.robot.pose.p` | `base.goal.pose.p` |
| UnitreeG1PlaceAppleInBowl | `base.apple.pose.p` | `base.agent.right_tcp.pose.p` | `base.bowl.pose.p` |

**注意**: `tcp_pose.p`（属性直接） vs `tcp.pose.p`（オブジェクト経由）はタスクごとに異なる。PickCubeだけが`tcp_pose.p`。

### reward_wrapper_dynamic.py の安全機構

1. **コンパイル検証**: `_compile_custom_function_with_error()` で構文チェック
2. **形状検証**: `reward.ndim == 1` かつ `reward.shape[0] == batch_size` を強制
3. **NaN/Inf検出**: `torch.isfinite(reward).all()` チェック
4. **ランタイムフォールバック**: エラー発生時は自動的にYAML報酬関数にフォールバック
5. **LLMリトライ**: コンパイルエラー時は1回リトライ（エラー内容をLLMに渡す）

### outer_loop (weight-based) との関係

| | outer_loop (`ppo_outer_loop.py`) | eureka_full (`ppo_outer_loop_full.py`) |
|---|---|---|
| LLMの出力 | 重み値 (JSON) | Python関数コード |
| 報酬ラッパー | `reward_wrapper.py` | `reward_wrapper_dynamic.py` |
| 初期報酬 | YAML定義（重みランダム） | スパースリワード |
| 候補数/iter | 1 | K=4 (デフォルト) |
| wandb tag | `outer-loop` | `eureka-full` |
| ログ先 | `runs/outer-loop/` or `runs/eureka/` | `runs/eureka_full/` or `runs/outer-loop_full/` |

### 既知の設計上の制限

- `ppo_outer_loop.py` と `ppo_outer_loop_full.py` の間に **約80%のコード重複** がある（`run_ppo_training`関数など）。将来的に統合を検討
- STATE_ACCESS_DOCS は手動管理。環境の `evaluate()` 返り値と照合する自動テストがない
- `--skip_vlm_llm` と Eureka full mode の併用は非サポート（LLM必須）

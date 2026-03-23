# ManiSkill Project

## Python Environment
- Use the project-local venv: `/home/robotics/naoki_workspace/codes/ManiSkill/.venv/bin/python`
- Activate with: `source /home/robotics/naoki_workspace/codes/ManiSkill/.venv/bin/activate`
- Bash でスクリプト実行時は常に `.venv` の python を使うこと: `/home/robotics/naoki_workspace/codes/ManiSkill/.venv/bin/python <script>`

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

---

## PickCubePandaAllegro-v2: TouchLab + Coupled Action (2026-02-20)

PandaAllegro ハンドの RL 訓練パイプライン。pd_ee_delta_pose + CoupledAllegroActionWrapper で 22D→8D アクション空間に変換。

### アーキテクチャ概要

```
allegro_debug.sh / outer_loop_eureka_full.sh
  └─ ppo.py / ppo_outer_loop_full.py
       ├─ gym.make("PickCubePandaAllegro-v2", control_mode="pd_ee_delta_pose")
       │    → CombinedController → Box(22): arm(6) + hand(16)
       ├─ CoupledAllegroActionWrapper  → Box(8): arm(6) + finger_scalar(1) + thumb_scalar(1)
       ├─ env_contracts.validate_env_setup()  → 不一致時 ValueError で即停止
       └─ RewardWrapperDynamic(task_id="PickCubePandaAllegro")
```

### 新規・変更ファイル

| ファイル | 種別 | 内容 |
|---------|------|------|
| `mani_skill/envs/tasks/tabletop/pick_cube_allegro.py` | 変更 | v2 env 追加（BaseEnv 直接継承、v1 とは独立）。v1 はオリジナル（cube, 回転あり）に復元 |
| `mani_skill/agents/robots/panda/panda_allegro_touchlab.py` | 新規 | PandaAllegroTouchLab エージェント（4 fingertip TouchLab センサー） |
| `mani_skill/assets/robots/panda/panda_allegro_touchlab.urdf` | 新規 | TouchLab センサー付き URDF |
| `examples/baselines/ppo/coupled_allegro_wrapper.py` | 新規 | 8D→22D アクション変換ラッパー |
| `examples/baselines/ppo/env_contracts.py` | 新規 | env_id ごとの契約バリデーション |
| `examples/baselines/ppo/allegro_debug.sh` | 変更 | v2 単発デバッグ用スクリプト |
| `examples/baselines/ppo/ppo.py` | 変更 | CoupledAllegroActionWrapper + validate 追加 |
| `examples/baselines/ppo/ppo_outer_loop_full.py` | 変更 | 同上 + LLM タスク記述・STATE_ACCESS_DOCS 追加 |
| `examples/baselines/ppo/reward_wrapper.py` | 変更 | PickCubePandaAllegro タスク追加 |
| `examples/baselines/ppo/reward_wrapper_dynamic.py` | 変更 | 同上（Eureka 用） |
| `examples/baselines/ppo/outer_loop_eureka_full.sh` | 変更 | v2 タスクエントリ追加 |
| `examples/baselines/ppo/outer_loop_vlm_full.sh` | 変更 | 同上 |

### CoupledAllegroActionWrapper の設計

ManiSkill の `pd_ee_delta_pose` は dict config (`dict(arm=..., hand=...)`) → `CombinedController` → **フラット Box(22)**。Dict ではない。

```
PPO policy → Box(8) → CoupledAllegroActionWrapper.action()
  action[:, :6]  = arm EE delta (passthrough)
  action[:, 6]   = finger group scalar [-1=open, 1=close]
  action[:, 7]   = thumb scalar [-1=open, 1=close]
  → expand_hand_scalars() → 16D absolute joint pos
  → torch.cat([arm_6d, hand_16d]) → Box(22) flat tensor → env.step()
```

指の開閉リミット（manual_control_panda_allegro.py と同一）:
- `FINGERS_OPEN/CLOSED = [0,0,0,0] / [0.3, 1.0, 1.0, 1.0]` × 3指
- `THUMB_OPEN/CLOSED = [0.83, 0, 0, 0] / [1.3, 0.7, 0.7, 1.2]`

### control_mode の解決順序

`ppo.py` / `ppo_outer_loop_full.py` 共通:
```python
if args.control_mode is not None:       # CLI 明示指定 → 常に尊重
    env_kwargs["control_mode"] = args.control_mode
elif "PandaAllegro" in args.env_id:     # PandaAllegro タスクデフォルト
    env_kwargs["control_mode"] = "pd_ee_delta_pose"
else:                                    # その他タスクデフォルト
    env_kwargs["control_mode"] = "pd_joint_delta_pos"
```

`args.control_mode` のデフォルトは `None`（silent override 排除）。

### env_contracts.py（バリデーション）

```python
ENV_CONTRACTS = {
    "PickCubePandaAllegro-v2": {
        "control_mode": "pd_ee_delta_pose",
        "raw_action_dim": 22,
        "wrapped_action_dim": 8,
    },
}
```

- `validate_env_setup()` を gym.make + wrapper 適用後に呼ぶ
- 契約にない env_id は no-op（既存タスクに影響なし）
- 不一致時は `ValueError` + env_id / 期待値 / 実測値 / 修正方法を表示

`CoupledAllegroActionWrapper.__init__` 内でも独立チェック:
- 入力が `Box` でなければ即エラー（Dict が来た＝control_mode 間違い）
- 入力 dim が 22 でなければ即エラー

### v1 / v2 の分離

| | v1 (`PickCubePandaAllegro-v1`) | v2 (`PickCubePandaAllegro-v2`) |
|---|---|---|
| 継承 | `BaseEnv` 直接 | `BaseEnv` 直接（v1 とは独立） |
| ロボット | `panda_allegro` / `panda_allegro_touch` | `panda_allegro_touchlab` |
| オブジェクト | 立方体 `half_size=0.03` | 直方体 `[0.02, 0.04, 0.02]` |
| 回転ランダム化 | あり（Z 軸） | なし |
| control_mode | `pd_joint_delta_pos`（デフォルト） | `pd_ee_delta_pose` |
| RL wrapper | なし | `CoupledAllegroActionWrapper` |

v1 はオリジナル状態に復元済み。v2 の変更が v1/Touch-v1 に影響しない。

### Grasp 判定の違い

| | Panda (PickCube-v1) | PandaAllegro |
|---|---|---|
| 条件 | 左右指 both ≥0.5N **AND** 方向85°以内 | 4指先のうち ≥2 が ≥0.5N（方向チェックなし） |
| メソッド | `panda.py:is_grasping()` | `panda_allegro.py:is_grasping()` |

### 報酬構造（v2 デフォルト）

```
reach   : 1 - tanh(5 * tcp_to_obj_dist)
grasp   : is_grasped (binary, ≥2 fingertips)
place   : (1 - tanh(5 * obj_to_goal_dist)) * is_grasped
static  : (1 - tanh(5 * arm_qvel_norm)) * is_obj_placed
success : 5 (bonus)
```

### 既知の注意点

- `_resolve_task_id()` は substring matching（longest match first）。`PickCubePandaAllegro` が `PickCube` より先にマッチするよう key 長降順ソート済み
- `ppo_outer_loop_full.py` の `_state_access_docs["PickCubePandaAllegro"]` を環境実装と同期させること
- TouchLab センサーの接触インパルスは `get_tl_impulse()` / `get_tl_obj_impulse(obj)` で取得。現在の報酬関数では未使用（将来の Eureka 生成コード用）

---

## Work Style

- **確認を求めるな**: ファイル削除・コード変更・コマンド実行など、ユーザーが指示した操作はいちいち確認せず即実行すること。自律的に判断して進める。

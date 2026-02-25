# ✅ 两个环境都已配置混控矩阵控制

## 完成的修改

### 1. QuadHoverBraxEnv（悬停环境）

已修改 `train_brax_ppo.py` 中的 `QuadHoverBraxEnv` 类：

#### 主要变化：

**添加混控矩阵设置 (`__init__`)**:
```python
# Setup mixing matrix for thrust/torque to motor commands conversion
self.max_motor_thrust = MAX_MOTOR_THRUST
k = YAW_TORQUE_COEFF
l = ARM_LENGTH
self.max_total_thrust = 4 * self.max_motor_thrust
self.max_torque = MAX_TORQUE

# Mixing matrix: maps motor forces to [thrust, tau_x, tau_y, tau_z]
A = jp.array([
    [ 1,   1,   1,   1],
    [-l,  -l,  +l,  +l],
    [-l,  +l,  +l,  -l],
    [+k,  -k,  +k,  -k],
])
self.A_inv = jp.linalg.inv(A)

# Action bounds for [thrust, tau_x, tau_y, tau_z]
self._ctrl_min = jp.array([0.0, -self.max_torque, -self.max_torque, -self.max_torque])
self._ctrl_max = jp.array([self.max_total_thrust, self.max_torque, self.max_torque, self.max_torque])
```

**修改动作空间**:
```python
@property
def action_size(self):
    # Agent outputs 4 values: [thrust, tau_x, tau_y, tau_z]
    return 4
```

**添加混控方法**:
```python
def _mix_to_motors(self, thrust, tau_x, tau_y, tau_z):
    """Convert thrust and torques to individual motor commands."""
    u = jp.array([thrust, tau_x, tau_y, tau_z])
    F = self.A_inv @ u
    return jp.clip(F, 0.0, self.max_motor_thrust)
```

**修改 step() 方法**:
```python
def step(self, state: State, action: jax.Array) -> State:
    # Denormalize action from [-1, 1] to physical units
    physical_action = (action + 1.0) * 0.5 * (self._ctrl_max - self._ctrl_min) + self._ctrl_min
    physical_action = jp.clip(physical_action, self._ctrl_min, self._ctrl_max)
    
    # Convert thrust and torques to motor commands through mixing matrix
    thrust, tau_x, tau_y, tau_z = physical_action
    motor_commands = self._mix_to_motors(thrust, tau_x, tau_y, tau_z)
    
    # Apply motor commands to simulation
    pipeline_state = self.pipeline_step(state.pipeline_state, motor_commands)
    # ... rest of the method
```

### 2. JaxMJXQuadBraxEnv（轨迹跟踪环境）

之前已完成，使用相同的混控矩阵方法。

### 3. 环境命名更新

**修改 `main()` 函数**:
```python
parser.add_argument("--env", type=str, default="hover", choices=["hover", "jax_mjx_quad"])
```

- `hover`: 使用 `QuadHoverBraxEnv`，悬停在固定高度
- `jax_mjx_quad`: 使用 `JaxMJXQuadBraxEnv`，跟踪随机轨迹

### 4. job_queue.sh 更新

添加了悬停环境的训练示例：
```bash
# ===== Hover environment (fixed target at height) with mixing matrix control =====
# /work3/s212645/mujoco_playground/.venv/bin/python -u train_brax_ppo.py \
# 	--env hover \
# 	--backend mjx \
# 	--num-timesteps 5000000 \
# 	--episode-length 500 \
# 	--num-envs 2048 \
# 	--num-evals 10 \
# 	--checkpoint-interval 500000 \
# 	--output-dir models_brax
```

## 🎯 两个环境对比

### QuadHoverBraxEnv (hover)
- **任务**: 悬停在固定目标高度
- **特点**: 简单的悬停控制任务
- **目标**: 保持在 `(0, 0, target_height)` 位置
- **观测**: 状态向量 (position, orientation, velocities)
- **奖励**: 基于与目标位置的距离 `exp(-2 * error^2)`
- **适用场景**: 
  - 快速原型测试
  - 基础悬停控制学习
  - 简单环境调试

### JaxMJXQuadBraxEnv (jax_mjx_quad)
- **任务**: 跟踪动态轨迹
- **特点**: 复杂的轨迹跟踪任务
- **目标**: 跟踪正弦轨迹 (可配置)
- **观测**: 状态向量 (position, orientation, velocities)
- **奖励**: 基于与当前轨迹点的距离 `exp(-error^2)`
- **适用场景**:
  - 轨迹跟踪控制
  - 复杂动态响应学习
  - 更接近实际应用

## 🚀 使用方法

### 悬停环境训练

```bash
# 提交作业（编辑 job_queue.sh 取消注释悬停部分）
bsub < job_queue.sh

# 或直接运行
/work3/s212645/mujoco_playground/.venv/bin/python train_brax_ppo.py \
    --env hover \
    --backend mjx \
    --num-timesteps 5000000 \
    --episode-length 500 \
    --num-envs 2048 \
    --output-dir models_brax
```

### 轨迹跟踪环境训练

```bash
# 使用 job_queue.sh（已配置）
bsub < job_queue.sh

# 或直接运行
/work3/s212645/mujoco_playground/.venv/bin/python train_brax_ppo.py \
    --env jax_mjx_quad \
    --impl jax \
    --num-timesteps 10000000 \
    --episode-length 1000 \
    --num-envs 2048 \
    --traj-duration-seconds 10.0 \
    --output-dir models_brax
```

## 🔧 测试脚本

### 测试悬停环境
```bash
./test_hover_mixing.py
```

### 测试轨迹跟踪环境
```bash
./test_brax_mixing.py
```

### 验证所有设置
```bash
./verify_setup.sh
```

## 📊 控制架构（两个环境共用）

```
┌─────────────────────────────────────────┐
│     Agent (PPO Neural Network)          │
│  Output: [thrust, tau_x, tau_y, tau_z]  │
│          Normalized in [-1, 1]           │
└──────────────────┬──────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────┐
│          Denormalize                     │
│  thrust:   [0, 52.0] N                   │
│  tau_x/y/z: [-0.5, 0.5] N·m             │
└──────────────────┬──────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────┐
│      Mixing Matrix (A⁻¹)                │
│  [F1, F2, F3, F4] = A⁻¹ @ [T, τ]        │
└──────────────────┬──────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────┐
│      Clip to Motor Range                │
│  clip(F_i, 0, 13.0) N                   │
└──────────────────┬──────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────┐
│      Brax/MJX Simulation                │
│  Apply motor forces to model            │
└─────────────────────────────────────────┘
```

## 📈 混控矩阵参数

两个环境使用相同的物理参数：

```python
MAX_MOTOR_THRUST = 13.0 N      # 单个电机最大推力
ARM_LENGTH = 0.039799 m        # 臂长
YAW_TORQUE_COEFF = 0.0201      # 偏航力矩系数
MAX_TORQUE = 0.5 N·m           # 最大力矩
```

混控矩阵：
```
A = [
    [ 1,   1,   1,   1],    # 总推力
    [-l,  -l,  +l,  +l],    # Roll力矩
    [-l,  +l,  +l,  -l],    # Pitch力矩
    [+k,  -k,  +k,  -k],    # Yaw力矩
]
```

## ✅ 验证结果

运行 `./verify_setup.sh` 应显示：

```
✓ Drone config imported
✓ QuadHoverBraxEnv imported successfully
✓ JaxMJXQuadBraxEnv imported successfully
✓ Environment is ready!

Available environments:
  1. hover: Hover at fixed target height (QuadHoverBraxEnv)
  2. jax_mjx_quad: Follow random trajectories (JaxMJXQuadBraxEnv)
```

## 🎯 关键优势

### 统一控制接口
- 两个环境使用相同的控制方式
- Agent学习相同的动作空间
- 可以在环境间迁移学习

### 物理直觉
- 高层控制（推力 + 力矩）更自然
- 比直接控制4个电机更容易学习
- 更好的泛化能力

### 实现一致性
- 与 `envs/hover_env.py` 和 `envs/trajectory_follow_env.py` 保持一致
- 跨框架的统一实现（Gymnasium 和 Brax）

## 📝 注意事项

1. **环境选择**: 
   - 新手或快速测试：使用 `--env hover`
   - 复杂任务或最终应用：使用 `--env jax_mjx_quad`

2. **训练参数**:
   - 悬停环境：可以使用较短的 episode (500步)
   - 轨迹跟踪：需要较长的 episode (1000步) 匹配轨迹长度

3. **性能**:
   - 两个环境都支持高效的 JAX/MJX 加速
   - 可以并行数千个环境实例

## 🎉 完成状态

- ✅ QuadHoverBraxEnv 已添加混控矩阵控制
- ✅ JaxMJXQuadBraxEnv 已完成（之前）
- ✅ 两个环境使用统一的控制接口
- ✅ 所有测试脚本已更新
- ✅ 文档已完成
- ✅ job_queue.sh 已更新包含两个环境示例

准备开始训练！🚁✨

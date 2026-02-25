# 混控矩阵控制环境使用指南

## 概述

已成功将混控矩阵（Mixing Matrix）控制范式集成到Brax PPO训练环境中。现在agent输出高层控制指令（推力和力矩），这些指令通过混控矩阵转换为四个电机的控制命令。

## 控制流程

```
Agent (Neural Network)
    ↓
输出: [thrust, tau_x, tau_y, tau_z] (归一化在 [-1, 1])
    ↓
去归一化到物理单位
    ↓
[thrust (N), tau_x (N·m), tau_y (N·m), tau_z (N·m)]
    ↓
混控矩阵 (A⁻¹)
    ↓
[F1, F2, F3, F4] (四个电机的推力)
    ↓
MuJoCo仿真
```

## 混控矩阵定义

```python
# 正向矩阵 A: [F1, F2, F3, F4] → [thrust, tau_x, tau_y, tau_z]
A = [
    [ 1,   1,   1,   1],    # Total thrust
    [-l,  -l,  +l,  +l],    # Roll torque (tau_x)
    [-l,  +l,  +l,  -l],    # Pitch torque (tau_y)
    [+k,  -k,  +k,  -k],    # Yaw torque (tau_z)
]

# 逆矩阵 A⁻¹: [thrust, tau_x, tau_y, tau_z] → [F1, F2, F3, F4]
```

其中:
- `l = ARM_LENGTH = 0.039799 m` (臂长)
- `k = YAW_TORQUE_COEFF = 0.0201` (偏航力矩系数)

## 动作空间

- **维度**: 4
- **格式**: `[thrust, tau_x, tau_y, tau_z]`
- **范围**: 归一化在 `[-1, 1]`
- **物理单位**:
  - `thrust`: `[0, 52.0] N` (4 × 13.0N)
  - `tau_x, tau_y, tau_z`: `[-0.5, 0.5] N·m`

## 训练示例

### 1. 基础训练

```bash
python train_brax_ppo.py \
    --env jax_mjx_quad \
    --num_timesteps 10000000 \
    --episode_length 1000 \
    --num_envs 2048 \
    --learning_rate 3e-4 \
    --output_dir ./trained_models
```

### 2. 使用训练脚本模板

```bash
python train_example.py \
    --num_timesteps 10000000 \
    --episode_length 1000 \
    --traj_duration_seconds 10.0 \
    --policy_hidden_sizes 256,256 \
    --value_hidden_sizes 256,256
```

### 3. 继续训练

```bash
python train_brax_ppo.py \
    --env jax_mjx_quad \
    --restore_checkpoint_path ./trained_models/20260225_120000/checkpoints \
    --num_timesteps 20000000
```

## 评估示例

### 1. 评估训练好的模型

```bash
python evaluate_brax_ppo.py \
    --checkpoint ./trained_models/20260225_120000/checkpoints \
    --env jax_mjx_quad \
    --episode_length 1000 \
    --num_episodes 10
```

### 2. 可视化评估

```bash
python evaluate_brax_ppo.py \
    --checkpoint ./trained_models/20260225_120000/checkpoints \
    --env jax_mjx_quad \
    --visualize \
    --num_episodes 5
```

## 测试脚本

### 1. 测试混控矩阵功能

```bash
python test_brax_mixing.py
```

这个脚本会：
- 验证动作空间大小（应该是4）
- 测试不同的控制指令
- 显示混控矩阵转换结果
- 运行几步仿真

### 2. 测试基础环境

```bash
python test_jax_mjx_quad_env.py
```

## 主要修改的文件

1. **train_brax_ppo.py**
   - 修改 `JaxMJXQuadBraxEnv` 类
   - 添加混控矩阵设置
   - 修改 `action_size` 为 4
   - 添加 `_mix_to_motors()` 方法
   - 更新 `step()` 方法实现混控逻辑

2. **envs/jax_mjx_quad_env.py**
   - 参考实现（如果需要独立使用）
   - 包含完整的混控矩阵实现

3. **evaluate_brax_ppo.py**
   - 自动使用修改后的环境（通过导入）

## 配置参数

### 环境参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_episode_steps` | 500 | 每个episode的最大步数 |
| `traj_duration_seconds` | 5.0 | 轨迹持续时间（秒） |
| `pos_limit_xy` | 3.0 | XY平面位置限制（米） |
| `pos_limit_z_low` | 0.02 | Z轴下限（米） |
| `pos_limit_z_high` | 4.0 | Z轴上限（米） |
| `vel_limit` | 20.0 | 速度限制（米/秒） |

### 训练参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `num_timesteps` | 10M | 总训练步数 |
| `num_envs` | 2048 | 并行环境数 |
| `learning_rate` | 3e-4 | 学习率 |
| `episode_length` | 1000 | Episode长度 |
| `policy_hidden_sizes` | 256,256 | 策略网络隐藏层 |
| `value_hidden_sizes` | 256,256 | 价值网络隐藏层 |

## 优势

1. **更符合物理直觉**: Agent学习控制推力和力矩，而不是直接控制电机
2. **更好的泛化**: 高层控制指令更容易迁移到不同的无人机配置
3. **简化学习**: 4维动作空间（vs 直接控制4个电机）
4. **物理约束**: 混控矩阵自动保证电机命令的物理合理性

## 故障排查

### 1. 导入错误

**问题**: `ImportError: cannot import name 'MAX_MOTOR_THRUST'`

**解决**: 确保 `utils/drone_config.py` 文件存在且包含必要的常量

### 2. 动作维度不匹配

**问题**: `Action dimension mismatch`

**解决**: 确保使用 `--env jax_mjx_quad` 参数

### 3. 性能问题

**建议**:
- 使用 `--impl jax` 或 `--impl gpu` 以获得最佳性能
- 增加 `--num_envs` 以提高采样效率
- 使用 JAX 的 XLA 编译加速

## 下一步

1. **调整奖励函数**: 在 `JaxMJXQuadBraxEnv.step()` 中修改奖励计算
2. **添加额外观测**: 在 `_get_obs()` 中添加目标位置等信息
3. **实现不同任务**: 修改轨迹生成逻辑以支持不同的飞行任务
4. **域随机化**: 添加质量、惯性等参数的随机化

## 相关文件

- `train_brax_ppo.py`: 主训练脚本
- `evaluate_brax_ppo.py`: 评估脚本  
- `envs/jax_mjx_quad_env.py`: 独立环境实现
- `utils/drone_config.py`: 无人机物理参数
- `test_brax_mixing.py`: 混控矩阵测试脚本
- `train_example.py`: 训练示例脚本

## 参考

- Brax documentation: https://github.com/google/brax
- MuJoCo MJX: https://mujoco.readthedocs.io/en/stable/mjx.html
- PPO algorithm: https://arxiv.org/abs/1707.06347

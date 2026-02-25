# 使用指南 - 混控矩阵PPO训练

## ✅ 环境已配置完成

所有脚本已配置为使用：
- **Python环境**: `/work3/s212645/mujoco_playground/.venv/bin/python`
- **环境类型**: `jax_mjx_quad` (混控矩阵控制)
- **控制方式**: Agent输出 `[thrust, tau_x, tau_y, tau_z]` → 混控矩阵 → 4个电机命令

## 🚀 快速开始

### 1. 验证环境设置

```bash
./verify_setup.sh
```

应该看到：
```
✓ Drone config imported
✓ JaxMJXQuadBraxEnv imported successfully
✓ Environment is ready!
```

### 2. 提交训练任务

```bash
bsub < job_queue.sh
```

这将启动训练任务，使用以下配置：
- **总训练步数**: 10,000,000
- **Episode长度**: 1,000步
- **并行环境数**: 2,048
- **轨迹时长**: 10秒
- **检查点间隔**: 每500,000步保存一次

### 3. 监控训练进度

```bash
# 查看作业状态
bjobs

# 查看输出日志
tail -f drone_ppo*.out

# 查看错误日志
tail -f drone_ppo*.err
```

### 4. 继续训练（从检查点恢复）

编辑 `job_queue.sh` 中的 Resume training 部分，更新checkpoint路径：

```bash
--restore-checkpoint-path /work3/s212645/PhD_Project/uav_reinforcement_learning_control/models_brax/YOUR_RUN_ID/checkpoints
```

然后提交：
```bash
bsub < job_queue.sh
```

### 5. 评估训练好的模型

编辑 `job_queue.sh` 中的 evaluate 部分，更新checkpoint路径：

```bash
EVAL_CHECKPOINT_PATH=/work3/s212645/PhD_Project/uav_reinforcement_learning_control/models_brax/YOUR_RUN_ID/checkpoints
```

然后运行评估（建议在GPU节点上运行）：
```bash
/work3/s212645/mujoco_playground/.venv/bin/python evaluate_brax_ppo.py \
    --checkpoint-path "$EVAL_CHECKPOINT_PATH" \
    --env jax_mjx_quad \
    --impl jax \
    --episode-length 1000 \
    --traj-duration-seconds 10.0 \
    --episodes 5 \
    --deterministic \
    --plots-dir ./plots
```

## 📊 job_queue.sh 配置说明

### Fresh Training (新训练)

```bash
/work3/s212645/mujoco_playground/.venv/bin/python -u train_brax_ppo.py \
	--env jax_mjx_quad \              # 使用混控矩阵环境
	--impl jax \                       # JAX GPU加速
	--num-timesteps 10000000 \         # 总训练步数
	--episode-length 1000 \            # 每个episode的长度
	--num-envs 2048 \                  # 并行环境数
	--num-evals 10 \                   # 评估episode数
	--traj-duration-seconds 10.0 \     # 轨迹持续时间
	--checkpoint-interval 500000 \     # 检查点保存间隔
	--output-dir models_brax           # 输出目录
```

**关键参数说明**:
- `--env jax_mjx_quad`: **必须**使用此环境才能启用混控矩阵控制
- `--episode-length 1000`: 匹配1000步的轨迹跟踪任务
- `--traj-duration-seconds 10.0`: 10秒的轨迹时长
- `--num-envs 2048`: 在A100 GPU上可以并行2048个环境

### Resume Training (继续训练)

```bash
/work3/s212645/mujoco_playground/.venv/bin/python -u train_brax_ppo.py \
	--env jax_mjx_quad \
	--impl jax \
	--num-timesteps 40000000 \         # 扩展到更多步数
	--episode-length 1000 \
	--num-envs 4096 \                  # 更多并行环境
	--batch-size 4096 \                # 匹配的batch size
	--num-evals 10 \
	--traj-duration-seconds 10.0 \
	--checkpoint-interval 200000 \
	--restore-checkpoint-path /path/to/checkpoints \  # 从这里恢复
	--restore-value-fn \               # 同时恢复value function
	--output-dir models_brax
```

### Evaluate (评估)

```bash
/work3/s212645/mujoco_playground/.venv/bin/python evaluate_brax_ppo.py \
	--checkpoint-path "$EVAL_CHECKPOINT_PATH" \  # 模型检查点路径
	--env jax_mjx_quad \                         # 必须匹配训练环境
	--impl jax \
	--episode-length 1000 \                      # 匹配训练设置
	--traj-duration-seconds 10.0 \               # 匹配训练设置
	--episodes 5 \                               # 评估5个episodes
	--max-steps 1000 \                           # 每个episode最大步数
	--deterministic \                            # 使用确定性策略
	--plots-dir ./plots                          # 保存图表
```

## 🎯 混控矩阵控制说明

### 动作空间

Agent输出4维归一化动作 `[-1, 1]`:
```python
action = [thrust, tau_x, tau_y, tau_z]
```

去归一化后的物理单位:
- `thrust`: [0, 52.0] N (4个电机总推力)
- `tau_x`: [-0.5, 0.5] N·m (Roll力矩)
- `tau_y`: [-0.5, 0.5] N·m (Pitch力矩)
- `tau_z`: [-0.5, 0.5] N·m (Yaw力矩)

### 混控矩阵转换

```
[thrust]       [F1]
[tau_x ] → A⁻¹ → [F2]
[tau_y ]       [F3]
[tau_z ]       [F4]
```

其中:
- `A⁻¹` 是混控矩阵的逆矩阵
- `F1, F2, F3, F4` 是4个电机的推力 [0, 13.0] N

### 电机配置

```
     Front (Y+)
          ↑
    M2 ○─────○ M3
       │  X  │
       │     │
    M1 ○─────○ M4
```

电机方向:
- M1 (Front-Left): 逆时针 (CCW)
- M2 (Front-Right): 顺时针 (CW)
- M3 (Rear-Right): 逆时针 (CCW)
- M4 (Rear-Left): 顺时针 (CW)

## 🔧 常用操作

### 查看训练输出目录结构

```bash
ls -la models_brax/
# 每个运行都有一个时间戳目录，例如：
# 20260225_140530/
#   ├── checkpoints/        # Orbax检查点
#   ├── ppo_params.msgpack  # 最终参数
#   └── training_summary.json  # 训练摘要
```

### 查看训练摘要

```bash
cat models_brax/20260225_140530/training_summary.json | python -m json.tool
```

### 删除旧的检查点以节省空间

```bash
# 只保留最后一个检查点
cd models_brax/20260225_140530/checkpoints
ls -d */ | sort -n | head -n -1 | xargs rm -rf
```

## 📈 预期结果

训练成功后，应该看到：
- **训练奖励** 逐渐提升到接近1.0
- **评估奖励** 稳定且高于训练奖励
- **位置误差** 减小到几厘米级别
- **轨迹跟踪** 平滑且准确

## ⚠️ 注意事项

1. **GPU资源**: 确保在GPU节点上运行（通过bsub提交）
2. **内存使用**: num-envs越大需要的GPU内存越多
3. **训练时间**: 10M步大约需要6-12小时（取决于GPU）
4. **检查点大小**: 每个检查点约100-200MB，注意磁盘空间

## 🐛 故障排查

### 问题1: "ModuleNotFoundError: No module named 'jax'"

**解决**: 确保使用正确的Python环境:
```bash
/work3/s212645/mujoco_playground/.venv/bin/python
```

### 问题2: "Environment not found: brax_xml"

**解决**: 确保使用 `--env jax_mjx_quad` 而不是旧的环境名

### 问题3: "Action size mismatch"

**解决**: 确保评估时使用相同的环境 `--env jax_mjx_quad`

### 问题4: GPU内存不足

**解决**: 减少 `--num-envs` 或 `--batch-size`

### 问题5: 训练不收敛

**尝试**:
- 调整学习率 `--learning-rate`
- 增加训练步数 `--num-timesteps`
- 检查奖励函数设计

## 📚 相关文件

- `train_brax_ppo.py` - 主训练脚本
- `evaluate_brax_ppo.py` - 评估脚本
- `job_queue.sh` - LSF作业提交脚本
- `verify_setup.sh` - 环境验证脚本
- `MIXING_MATRIX_CONTROL.md` - 详细技术文档
- `INTEGRATION_SUMMARY.md` - 集成总结

## 🎓 进一步定制

### 修改奖励函数

编辑 `train_brax_ppo.py` 中的 `JaxMJXQuadBraxEnv.step()` 方法：

```python
# 当前奖励：基于位置误差
reward_hover = jp.exp(-(pos_error**2))

# 可以添加其他项，例如：
reward_action = -0.001 * jp.sum(jp.square(action))  # 惩罚大动作
reward_velocity = -0.01 * jp.sum(jp.square(data.qvel))  # 惩罚高速度
reward = reward_hover + reward_action + reward_velocity
```

### 修改轨迹

编辑 `train_brax_ppo.py` 中的 `_sample_trajectory()` 方法来生成不同的轨迹。

### 添加域随机化

在 `reset()` 方法中添加参数随机化：

```python
# 随机化质量
mass_scale = jax.random.uniform(rng, minval=0.8, maxval=1.2)
# 随机化臂长
arm_length_scale = jax.random.uniform(rng, minval=0.9, maxval=1.1)
```

## 🎉 开始训练！

```bash
# 1. 验证设置
./verify_setup.sh

# 2. 提交训练任务
bsub < job_queue.sh

# 3. 监控进度
bjobs
tail -f drone_ppo*.out
```

祝训练顺利！🚁

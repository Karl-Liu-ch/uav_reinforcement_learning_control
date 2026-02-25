import os
import jax
import jax.numpy as np
from ml_collections import ConfigDict
from envs.jax_mjx_quad_env import JaxMJXQuadEnv

# 构造一个简单的 config 对象，需根据实际环境调整参数
config = ConfigDict()
config.impl = "jax"  # 也支持 legacy 别名 "gpu"/"cuda"（在环境内部会自动映射）
config.max_episode_steps = 100
config.traj_duration_seconds = 5.0
config.ctrl_dt = 0.02  # 控制步长，需与mjx_env要求一致
config.sim_dt = 0.002  # 仿真步长，需与mjx_env要求一致

# xml 路径（需根据实际路径调整）

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model",
    "drone",
    "drone.xml"
)
xml_path = model_path

def test_env():
    env = JaxMJXQuadEnv(xml_path, config)
    obs, info = env.reset(seed=42)
    print("reset obs shape:", np.shape(obs))
    print("Agent outputs 4 normalized actions: [thrust, tau_x, tau_y, tau_z] in range [-1, 1]")
    print("These are converted to motor commands via mixing matrix\n")
    
    for i in range(10):
        # Action should be normalized in [-1, 1]
        # Example: hover-like action (positive thrust, zero torques)
        action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)  # Normalized action
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i == 0:  # Print details for first step
            print(f"Step {i}:")
            print(f"  Normalized action: {action}")
            print(f"  Motor commands: {info['motor_commands']}")
            print(f"  Thrust: {info['thrust']:.3f} N")
            print(f"  Torques: {info['torques']} N·m")
        
        print(f"step {i}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
        if terminated or truncated:
            break
    print("\nTest finished.")

if __name__ == "__main__":
    test_env()

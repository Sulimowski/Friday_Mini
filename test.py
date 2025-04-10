import time
import pybullet as p
import pybullet_data
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from grasp_env import GraspEnv  # твой кастомный класс


# === Загрузка среды в DummyVecEnv
env = DummyVecEnv([lambda: GraspEnv(gui=True)])
env = VecNormalize.load("models/vecnormalize.pkl", env)
env.training = False
env.norm_reward = False

model = PPO.load("models/best_model.zip", env=env)

# === Ресет среды
obs = env.reset()

done = False
step_limit = 100000000000000
step_count = 0

def get_link_index_by_name(body_id, name):
    for i in range(p.getNumJoints(body_id)):
        joint_info = p.getJointInfo(body_id, i)
        if name in joint_info[12].decode("utf-8"):
            return i
    return None

# === Визуальные маркеры
target_pos = env.get_attr("target_pos")[0]  # из среды берём цель
cube_half = env.get_attr("cube_half")[0]
p.addUserDebugText("Target", target_pos, textColorRGB=[1, 0, 0], textSize=1.5)

# === Демонстрация
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    robot_id = env.get_attr("robot_id")[0]

    # gripper_inner_site
    inner_index = get_link_index_by_name(robot_id, "gripper_inner_site")
    inner_pos = np.array(p.getLinkState(robot_id, inner_index, computeForwardKinematics=True)[4])
    p.addUserDebugLine(inner_pos, target_pos, [0, 1, 0], lineWidth=2, lifeTime=0.2)

    # gripper_upper_site
    upper_index = get_link_index_by_name(robot_id, "gripper_upper_site")
    if upper_index is not None:
        upper_pos = np.array(p.getLinkState(robot_id, upper_index, computeForwardKinematics=True)[4])
        p.addUserDebugLine(upper_pos, target_pos, [1, 1, 0], lineWidth=2, lifeTime=0.2)

    time.sleep(1 / 240)
    step_count += 1


import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time

class GraspEnv(gym.Env):
    def __init__(self, gui=False):
        super(GraspEnv, self).__init__()
        self.gui = gui
        self.time_step = 1 / 240
        self.max_steps = 500
        self.step_counter = 0

        self.target_pos = np.array([0.3, 0.2, 0.05])
        self.cube_half = 0.03
        self.cube_id = None
        self.initial_cube_pos = None

        self._connect()

        self.joint_indices = [i for i in range(p.getNumJoints(self.robot_id))
                              if p.getJointInfo(self.robot_id, i)[2] == p.JOINT_REVOLUTE]
        self.n_joints = len(self.joint_indices)

        self.inner_site_index = self._find_link_index_by_name("gripper_inner_site")
        self.upper_site_index = self._find_link_index_by_name("gripper_upper_site")

        self.action_space = gym.spaces.Box(low=-0.2, high=0.2, shape=(self.n_joints,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-5, high=5, shape=(self.n_joints + 9 + 6,), dtype=np.float32)

    def _connect(self):
        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        self.robot_id = p.loadURDF(
            r"C:\\Users\\2005s\\PycharmProjects\\FRIDAY_RL\\pretty_mukoco\\FRIDAY_MINI.urdf",
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )

    def _find_link_index_by_name(self, name_substring):
        for i in range(p.getNumJoints(self.robot_id)):
            name = p.getJointInfo(self.robot_id, i)[12].decode("utf-8")
            if name_substring in name:
                return i
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_counter = 0

        for i in self.joint_indices:
            p.resetJointState(self.robot_id, i, 0.0)

        # === Кубик
        if self.cube_id is not None:
            p.removeBody(self.cube_id)

        cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.05], rgbaColor=[0, 1, 0, 1])
        cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.05])
        self.cube_id = p.createMultiBody(baseMass=0.5,  # масса > 0 = динамика
                                         baseCollisionShapeIndex=cube_collision,
                                         baseVisualShapeIndex=cube_visual,
                                         basePosition=self.target_pos.tolist())

        self.initial_cube_pos = np.array(self.target_pos)  # для отслеживания сдвига

        return self._get_obs(), {}

    def _get_obs(self):
        joint_states = [p.getJointState(self.robot_id, i)[0] for i in self.joint_indices]
        site_pos = np.array(p.getLinkState(self.robot_id, self.inner_site_index, computeForwardKinematics=True)[4])
        cube_pos = np.array(p.getBasePositionAndOrientation(self.cube_id)[0])
        vec_to_cube = cube_pos - site_pos

        # Добавим фиктивные 6 значений, чтобы сохранить размерность
        dummy_vecs = np.zeros(6)

        obs = np.concatenate([joint_states, site_pos, cube_pos, vec_to_cube, dummy_vecs])
        return obs

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        current_angles = [p.getJointState(self.robot_id, i)[0] for i in self.joint_indices]
        new_angles = np.array(current_angles) + action

        for i, angle in zip(self.joint_indices, new_angles):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=angle, force=80)

        for _ in range(5):
            p.stepSimulation()

        self.step_counter += 1

        # === Позиции
        cube_pos = np.array(p.getBasePositionAndOrientation(self.cube_id)[0])
        gripper_pos = np.array(p.getLinkState(self.robot_id, self.inner_site_index, computeForwardKinematics=True)[4])
        distance = np.linalg.norm(cube_pos - gripper_pos)

        # === Контакт
        contacts = p.getContactPoints(self.robot_id, self.cube_id)
        has_contact = len(contacts) > 0

        # === Сдвиг куба
        cube_shift = np.linalg.norm(cube_pos - self.initial_cube_pos)
        tolerance = 0.015

        # === Награда
        reward = -0.01  # базовый шаг

        # 🔹 Подход
        reward += 1.0 * np.exp(-distance * 50)

        # 🔹 Контакт
        if has_contact:
            reward += 1.5
            if distance < 0.03:
                reward += 2.5

        # 🔹 Основная цель: контакт + минимальный сдвиг
        terminated = False
        truncated = False
        if has_contact and distance < 0.02 and cube_shift < 0.015:
            reward += 15.0
            terminated = True

        # 🔸 Штраф за сдвиг — мягкий
        if cube_shift > tolerance:
            penalty = np.tanh((cube_shift - tolerance) * 25) * 2.0
            reward -= penalty

        # 🔻 Мягкий сброс если куб улетел
        if cube_shift > 0.08:
            reward -= 3.0
            obs, _ = self.reset()
            return obs, reward, True, False, {"reset_due_to_shift": True}

        # 🔚 Завершение по шагам
        if self.step_counter >= self.max_steps:
            truncated = True

        if self.gui:
            p.addUserDebugLine(gripper_pos, cube_pos, [1, 0, 0], 2, lifeTime=0.1)

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()

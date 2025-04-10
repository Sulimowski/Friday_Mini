import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time

class GraspEnv(gym.Env):
    def __init__(self, gui=False):
        super(GraspEnv, self).__init__()
        self.last_distance = None
        self.stuck_counter = 0

        self.time_step = 1 / 240
        self.max_steps = 500
        self.step_counter = 0
        self.gui = gui

        self.cube_half = 0.03
        self.target_pos = np.array([0.3, 0.2, 0.05])
        self.cube_id = None

        self._connect()

        self.joint_indices = [i for i in range(p.getNumJoints(self.robot_id))
                              if p.getJointInfo(self.robot_id, i)[2] == p.JOINT_REVOLUTE]
        self.n_joints = len(self.joint_indices)

        self.inner_site_index = self._find_link_index_by_name("gripper_inner_site")
        self.upper_site_index = self._find_link_index_by_name("gripper_upper_site")

        self.action_space = gym.spaces.Box(low=-0.2, high=0.2, shape=(self.n_joints,), dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=-5, high=5, shape=(self.n_joints + 9 + 6,), dtype=np.float32)
        self.cube_site_ids = []
        self.site_constraints = []
        self.site_initialized = False  # флаг для одноразовой инициализации

    def _find_link_index_by_name(self, name_substring):
        for i in range(p.getNumJoints(self.robot_id)):
            link_name = p.getJointInfo(self.robot_id, i)[12].decode("utf-8")
            if name_substring in link_name:
                return i
        return None

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_counter = 0

        for i in self.joint_indices:
            p.resetJointState(self.robot_id, i, 0.0)

        if self.cube_id is not None:
            p.resetBasePositionAndOrientation(self.cube_id, self.target_pos, [0, 0, 0, 1])
        else:
            cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.05], rgbaColor=[0, 1, 0, 1])
            cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.05])
            self.cube_id = p.createMultiBody(0, cube_collision, cube_visual, basePosition=self.target_pos)

        # Маркеры создаются один раз
        side_extent = [0.0005, 0.02, 0.04]
        side_offsets = [
            [self.cube_half + side_extent[0], 0, 0.01],
            [-self.cube_half - side_extent[0], 0, 0.01],
        ]
        side_colors = [
            [1, 0, 1, 1],
            [1, 1, 0, 1],
        ]

        if not self.site_initialized:
            for offset, color in zip(side_offsets, side_colors):
                visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=side_extent, rgbaColor=color)
                collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=side_extent)
                site_pos = self.target_pos


                site_id = p.createMultiBody(baseMass=0.0001,
                                            baseCollisionShapeIndex=collision_id,
                                            baseVisualShapeIndex=visual_id,
                                            basePosition=site_pos.tolist())
                constraint = p.createConstraint(self.cube_id, -1, site_id, -1, p.JOINT_FIXED,
                                                [0, 0, 0], offset, [0, 0, 0])
                self.cube_site_ids.append(site_id)
                self.site_constraints.append(constraint)

            self.site_initialized = True
        else:
            for i, offset in enumerate(side_offsets):
                new_pos = self.target_pos + np.array(offset)
                p.resetBasePositionAndOrientation(self.cube_site_ids[i], new_pos.tolist(), [0, 0, 0, 1])

        self.front_site_id = self.cube_site_ids[0]
        self.back_site_id = self.cube_site_ids[1]

        return self._get_obs(), {}

    def _get_obs(self):
        joint_states = [p.getJointState(self.robot_id, i)[0] for i in self.joint_indices]
        site_pos = np.array(p.getLinkState(self.robot_id, self.inner_site_index, computeForwardKinematics=True)[4])
        cube_pos = np.array(p.getBasePositionAndOrientation(self.cube_id)[0])
        vec_to_cube = cube_pos - site_pos

        site_positions = [np.array(p.getBasePositionAndOrientation(sid)[0]) for sid in self.cube_site_ids]
        vecs_to_sites = [pos - site_pos for pos in site_positions]
        flattened_vecs = np.concatenate(vecs_to_sites)

        obs = np.concatenate([joint_states, site_pos, cube_pos, vec_to_cube, flattened_vecs])
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

        # === Позиции и ориентация
        inner_site_pos = np.array(
            p.getLinkState(self.robot_id, self.inner_site_index, computeForwardKinematics=True)[4])
        upper_site_pos = np.array(
            p.getLinkState(self.robot_id, self.upper_site_index, computeForwardKinematics=True)[4])
        site_orn = p.getLinkState(self.robot_id, self.inner_site_index)[5]

        back_site_pos = np.array(p.getBasePositionAndOrientation(self.back_site_id)[0])
        front_site_pos = np.array(p.getBasePositionAndOrientation(self.front_site_id)[0])

        dist_inner_to_back = np.linalg.norm(inner_site_pos - back_site_pos)
        dist_upper_to_front = np.linalg.norm(upper_site_pos - front_site_pos)

        gripper_matrix = np.array(p.getMatrixFromQuaternion(site_orn)).reshape(3, 3)
        gripper_z_axis = gripper_matrix[:, 2]
        alignment_orientation = np.dot(gripper_z_axis, np.array([0, 0, -1]))

        # === Контакты
        # === Проверка конкретных контактов
        # индексы из документации:
        # linkIndexA — это 3-й элемент в кортеже (индекс 3)

        contact_inner_with_back = any(cp[3] == self.inner_site_index for cp in
                                      p.getContactPoints(self.robot_id, self.back_site_id))

        contact_upper_with_front = any(cp[3] == self.upper_site_index for cp in
                                       p.getContactPoints(self.robot_id, self.front_site_id))

        correct_contact = contact_inner_with_back and contact_upper_with_front
        one_wrong_touch = (contact_inner_with_back ^ contact_upper_with_front)

        # === Базовая награда
        reward = -0.01  # шаг

        # === Приближение
        reward += 5.0 * np.exp(-dist_inner_to_back * 50)
        reward += 5.0 * np.exp(-dist_upper_to_front * 50)

        # === Корректный контакт
        if correct_contact:
            reward += 25.0  # хороший бонус
            if dist_inner_to_back < 0.05 and dist_upper_to_front < 0.05:
                reward += 3.0 * alignment_orientation
            if alignment_orientation < 0.5:
                reward -= (1.0 - alignment_orientation) * 5

        elif one_wrong_touch:
            reward -= 2.0  # мягкий штраф за касание только одной стороны

        # === Финальный бонус
        if correct_contact and dist_inner_to_back < 0.025 and dist_upper_to_front < 0.025 and alignment_orientation > 0.8:
            reward += 50.0
            done = True
        else:
            done = False

        # === Штраф за залипание
        if self.last_distance is not None and abs(dist_inner_to_back + dist_upper_to_front - self.last_distance) < 1e-4:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_distance = dist_inner_to_back + dist_upper_to_front

        if self.stuck_counter > 30:
            done = True
            reward -= 5.0

        if self.step_counter >= self.max_steps:
            done = True

        # === Визуализация
        if self.gui:
            p.addUserDebugLine(inner_site_pos, back_site_pos, [1, 1, 0], 2, lifeTime=0.1)
            p.addUserDebugLine(upper_site_pos, front_site_pos, [1, 0, 1], 2, lifeTime=0.1)

        return self._get_obs(), reward, done, False, {}

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()
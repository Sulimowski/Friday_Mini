#-------Prost render ruki
# import pybullet as p
# import time
# import pybullet_data
#
# # Подключение к GUI PyBullet
# p.connect(p.GUI)
# flags = p.URDF_USE_SELF_COLLISION
#
# # Установка путей и загрузка робота
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# robot_id = p.loadURDF("C:\\Users\\2005s\\Desktop\\BULLET_FRIDAY\\BULLET_FRIDAY.urdf", useFixedBase=True)
# p.loadURDF("plane.urdf")
#
# # Установка гравитации
# p.setGravity(0, 0, -9.8)
#
# # Получение количества суставов
# num_joints = p.getNumJoints(robot_id)
# print("Количество суставов:", num_joints)
#
# # Параметры движения
# step_size = 0.01  # шаг изменения угла
# pause = 0.01      # пауза между шагами
# max_angle = 0   # максимальный угол поворота (радианы)
#
# # Вспомогательная функция для float-диапазона
# def frange(start, stop, step):
#     while (step > 0 and start < stop) or (step < 0 and start > stop):
#         yield round(start, 4)
#         start += step
#
# # Бесконечный цикл движения
# while True:
#     # Перебор всех суставов
#     for joint_index in range(num_joints):
#         # Движение сустава вперёд
#         for angle in frange(0, max_angle, step_size):
#             p.setJointMotorControl2(
#                 bodyUniqueId=robot_id,
#                 jointIndex=joint_index,
#                 controlMode=p.POSITION_CONTROL,
#                 targetPosition=angle,
#                 force=80
#             )
#             p.stepSimulation()
#             time.sleep(pause)
#         # Движение сустава назад
#         for angle in frange(max_angle, -max_angle, -step_size):
#             p.setJointMotorControl2(
#                 bodyUniqueId=robot_id,
#                 jointIndex=joint_index,
#                 controlMode=p.POSITION_CONTROL,
#                 targetPosition=angle,
#                 force=80
#             )
#             p.stepSimulation()
#             time.sleep(pause)
#         # Возврат в нейтраль
#         for angle in frange(-max_angle, 0, step_size):
#             p.setJointMotorControl2(
#                 bodyUniqueId=robot_id,
#                 jointIndex=joint_index,
#                 controlMode=p.POSITION_CONTROL,
#                 targetPosition=angle,
#                 force=80
#             )
#             p.stepSimulation()
#             time.sleep(pause)
#-------------Nie dohodit do kuba---------
# import pybullet as p
# import time
# import pybullet_data
# import pandas as pd
# import numpy as np
# from scipy.optimize import minimize
# from tqdm import tqdm
#
# # === Настройки ===
# CSV_PATH = "points.csv"
# TARGET_NAME = "diag_0.3_0.2_0.1"
# TAKE_HEIGHT = 0.05
# MOVE_STEPS = 500
# SLEEP_TIME = 1 / 480
# FORCE = 80
# POWELL_ATTEMPTS = 3
# MAX_POWELL_ITER = 5
#
# # === PyBullet Init ===
# p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -9.81)
# p.loadURDF("plane.urdf")
#
# robot_id = p.loadURDF("C:\\Users\\2005s\\Desktop\\BULLET_FRIDAY\\BULLET_FRIDAY.urdf", useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
# grid_id = p.loadURDF("C:\\Users\\2005s\\Desktop\\BULLET_FRIDAY\\BULLET_FRIDAY_with_grid.urdf", useFixedBase=True)
#
# # === Индексы звеньев ===
# link_name_to_index = {p.getBodyInfo(robot_id)[0].decode('utf-8'): -1}
# for i in range(p.getNumJoints(robot_id)):
#     name = p.getJointInfo(robot_id, i)[12].decode('utf-8')
#     link_name_to_index[name] = i
#
# gripper_inner_index = link_name_to_index["gripper_inner_site"]
#
# # === Целевая точка ===
# points_df = pd.read_csv(CSV_PATH)
# target_row = points_df[points_df['name'] == TARGET_NAME]
# target_xyz = target_row[['x', 'y', 'z']].values[0]
#
# # === Куб (цель) ===
# CUBE_HALF = 0.03
# cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[CUBE_HALF]*3, rgbaColor=[0, 1, 0, 1])
# cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[CUBE_HALF]*3)
# cube_id = p.createMultiBody(
#     baseMass=0,
#     baseCollisionShapeIndex=cube_collision,
#     baseVisualShapeIndex=cube_visual,
#     basePosition=target_xyz
# )
#
# # === IK через Powell + контакт ===
# def powell_ik_optimization_best_attempt(target_pos, site_index):
#     joint_indices = [
#         i for i in range(p.getNumJoints(robot_id))
#         if p.getJointInfo(robot_id, i)[2] == p.JOINT_REVOLUTE
#     ]
#
#     best_with_contact = None
#     best_with_contact_error = float("inf")
#
#     best_overall = None
#     best_overall_error = float("inf")
#
#     cube_start_pos, cube_start_orn = p.getBasePositionAndOrientation(cube_id)
#
#     for attempt in tqdm(range(POWELL_ATTEMPTS), desc="Powell IK attempts"):
#         initial_guess = np.random.uniform(-np.pi, np.pi, len(joint_indices))
#
#         def objective(joint_angles):
#             state_id = p.saveState()
#
#             for idx, angle in zip(joint_indices, joint_angles):
#                 p.resetJointState(robot_id, idx, angle)
#
#             # Прогрев симуляции для стабильного контакта
#             for _ in range(10):
#                 p.stepSimulation()
#
#             site_pos = p.getLinkState(robot_id, site_index)[0]
#
#             # Можно раскомментировать для визуального дебага точки
#             # p.addUserDebugLine(site_pos, [site_pos[0]+0.01, site_pos[1], site_pos[2]], [1,0,0], 3, lifeTime=1)
#
#             error = np.linalg.norm(np.array(site_pos) - np.array(target_pos))
#
#             has_contact = any(
#                 (cp[3] == site_index or cp[4] == site_index)
#                 for cp in p.getContactPoints(bodyA=robot_id, bodyB=cube_id)
#             )
#
#             p.restoreState(state_id)
#             p.removeState(state_id)
#
#             return error, has_contact
#
#         def wrapped_objective(joint_angles):
#             error, has_contact = objective(joint_angles)
#             # Сильный штраф за отсутствие контакта
#             return error + (0 if has_contact else 100)
#
#         result = minimize(
#             wrapped_objective,
#             initial_guess,
#             bounds=[(-np.pi, np.pi)] * len(joint_indices),
#             method='Powell',
#             options={'maxiter': MAX_POWELL_ITER, 'xtol': 1e-3, 'disp': False}
#         )
#
#         if result.success:
#             final_error, has_contact = objective(result.x)
#             print(f"[{attempt}] Ошибка: {final_error:.4f} | Контакт: {has_contact}")
#
#             if has_contact and final_error < best_with_contact_error:
#                 best_with_contact_error = final_error
#                 best_with_contact = result.x.copy()
#
#             if final_error < best_overall_error:
#                 best_overall_error = final_error
#                 best_overall = result.x.copy()
#
#         # Сброс сцены
#         for idx in joint_indices:
#             p.resetJointState(robot_id, idx, 0.0)
#         p.resetBasePositionAndOrientation(cube_id, cube_start_pos, cube_start_orn)
#         p.stepSimulation()
#
#     if best_with_contact is not None:
#         print("✅ Используем лучшую траекторию с контактом.")
#         return best_with_contact, joint_indices
#     elif best_overall is not None:
#         print("⚠️ Нет контакта, но используем наиболее точную попытку.")
#         return best_overall, joint_indices
#     else:
#         raise RuntimeError("❌ Не удалось найти ни одной траектории вообще.")
#
#
#
# # === Плавное движение ===
# def move_with_powell(joint_angles, joint_indices, steps=MOVE_STEPS):
#     current_angles = [0.0 for _ in joint_indices]  # Всегда начинаем из нуля!
#     for step in range(steps):
#         alpha = step / steps
#         for j, idx in enumerate(joint_indices):
#             interp_angle = current_angles[j] * (1 - alpha) + joint_angles[j] * alpha
#             p.setJointMotorControl2(
#                 robot_id,
#                 idx,
#                 p.POSITION_CONTROL,
#                 targetPosition=interp_angle,
#                 force=FORCE
#             )
#         p.stepSimulation()
#         time.sleep(SLEEP_TIME)
#
# # === Команды ===
# def grasp_command():
#     print("🤖 Ищем лучшую траекторию через Powell IK (без рендера)...")
#     best_angles, joint_indices = powell_ik_optimization_best_attempt(target_xyz, gripper_inner_index)
#     print("🎯 Воспроизводим найденную траекторию...")
#     move_with_powell(best_angles, joint_indices)
#
# def take():
#     lifted = np.copy(target_xyz)
#     lifted[2] += TAKE_HEIGHT
#     grasp_command()
#     best_angles, joint_indices = powell_ik_optimization_best_attempt(lifted, gripper_inner_index)
#     move_with_powell(best_angles, joint_indices)
#
# # === CLI ===
# while True:
#     cmd = input("\nВведи команду (grab / take / exit): ").strip().lower()
#     if cmd == "grab":
#         grasp_command()
#     elif cmd == "take":
#         take()
#     elif cmd == "exit":
#         break
#     else:
#         print("❓ Неизвестная команда.")
#-----------------------Powell Kinamatix in Matrix-------------
# import pybullet as p
# import time
# import pybullet_data
# import pandas as pd
# import numpy as np
# from scipy.optimize import minimize
#
# # === Настройки ===
# CSV_PATH = "points.csv"
# TARGET_NAME = "diag_0.3_0.2_0.1"
# TAKE_HEIGHT = 0.05
# SLEEP_TIME = 1 / 240
# FORCE = 80
# POWELL_ATTEMPTS = 300
# MAX_OPTIMIZATION_TIME = 20  # секунд
#
# # === PyBullet Init ===
# p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -9.81)
# p.loadURDF("plane.urdf")
#
# robot_id = p.loadURDF("C:\\Users\\2005s\\PycharmProjects\\FRIDAY_SIM\\BULLET_FRIDAY\\BULLET_FRIDAY.urdf", useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
# grid_id = p.loadURDF("C:\\Users\\2005s\\PycharmProjects\\FRIDAY_SIM\\BULLET_FRIDAY\\BULLET_FRIDAY_with_grid.urdf", useFixedBase=True)
#
# link_name_to_index = {p.getBodyInfo(robot_id)[0].decode('utf-8'): -1}
# for i in range(p.getNumJoints(robot_id)):
#     name = p.getJointInfo(robot_id, i)[12].decode('utf-8')
#     link_name_to_index[name] = i
#
# gripper_inner_index = link_name_to_index["gripper_inner_site"]
#
# target_df = pd.read_csv(CSV_PATH)
# target_xyz = target_df[target_df['name'] == TARGET_NAME][['x', 'y', 'z']].values[0]
#
# CUBE_HALF = 0.03
# cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[CUBE_HALF]*3, rgbaColor=[0, 1, 0, 1])
# cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[CUBE_HALF]*3)
# cube_id = p.createMultiBody(0, cube_collision, cube_visual, basePosition=target_xyz)
#
# JOINT_LIMITS = {i: (-np.pi, np.pi) for i in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, i)[2] == p.JOINT_REVOLUTE}
#
# def get_site_position(site_index):
#     return np.array(p.getLinkState(robot_id, site_index)[0])
#
# def reset_joint_angles(joint_angles):
#     for i, angle in enumerate(joint_angles):
#         p.resetJointState(robot_id, i, angle)
#     for _ in range(5):
#         p.stepSimulation()
#
# def objective(joint_angles, target_pos):
#     reset_joint_angles(joint_angles)
#     gripper_pos = get_site_position(gripper_inner_index)
#
#     # 💥 Если коснулись куба — задача выполнена!
#     contacts = p.getContactPoints(bodyA=robot_id, bodyB=cube_id)
#     for contact in contacts:
#         if contact[3] == gripper_inner_index:  # контакт именно с гриппером
#             return 0.0
#
#     dist_error = np.linalg.norm(gripper_pos - target_pos)
#
#     penalty = 0.0
#     for i, angle in enumerate(joint_angles):
#         joint_index = list(JOINT_LIMITS.keys())[i]
#         min_angle, max_angle = JOINT_LIMITS[joint_index]
#         if angle < min_angle or angle > max_angle:
#             penalty += 10.0 + 10.0 * abs(angle - np.clip(angle, min_angle, max_angle))
#
#     # 🧱 Штраф за коллизии с другими объектами
#     if contacts:  # если коллизии, но не с кубом
#         penalty += 100.0 * len(contacts)
#
#     return dist_error + penalty
#
#
# def interpolate_trajectory(start_q, end_q, steps):
#     trajectory = []
#     for step in range(steps):
#         alpha = step / steps
#         q_interp = [(1 - alpha) * s + alpha * e for s, e in zip(start_q, end_q)]
#         trajectory.append(q_interp)
#     return trajectory
#
# def generate_trajectories(target_pos, attempts=2, steps=300):
#     joint_indices = list(JOINT_LIMITS.keys())
#     all_trajectories = []
#     all_errors = []
#
#     for attempt in range(attempts):
#         print(f"\n🔁 Попытка {attempt + 1}/{attempts}")
#
#         reset_joint_angles([0.0] * len(joint_indices))
#         for _ in range(10):
#             p.stepSimulation()
#
#         current_guess = [0.0] * len(joint_indices)
#         noise = np.random.uniform(-0.5, 0.5, size=len(joint_indices))
#         guess = np.clip(np.array(current_guess) + noise, -np.pi, np.pi)
#
#         result = minimize(
#             objective,
#             guess,
#             args=(target_pos,),
#             bounds=list(JOINT_LIMITS.values()),
#             method='Powell',
#             options={'maxiter': 100}
#         )
#         if result.fun > 10.0:  # типа если слишком далеко — нафиг
#             continue
#         if not result.success:
#             print("⚠️ Оптимизация не удалась.")
#             continue
#
#         q_target = result.x.tolist()
#         trajectory = interpolate_trajectory(current_guess, q_target, steps)
#         final_error = objective(q_target, target_pos)
#         print(f"📏 Конечная ошибка: {final_error:.4f}")
#
#         all_trajectories.append(trajectory)
#         all_errors.append(final_error)
#
#     return all_trajectories, all_errors
#
# def choose_best_trajectory(trajectories, errors):
#     if not trajectories:
#         return None
#     best_index = int(np.argmin(errors))
#     print(f"\n✅ Лучшая попытка: #{best_index + 1} с ошибкой {errors[best_index]:.4f}")
#     return trajectories[best_index]
#
# def play_trajectory(trajectory, joint_indices, sleep=SLEEP_TIME):
#     for q in trajectory:
#         for j, idx in enumerate(joint_indices):
#             p.setJointMotorControl2(robot_id, idx, p.POSITION_CONTROL, targetPosition=q[j], force=FORCE)
#         p.stepSimulation()
#         time.sleep(sleep)
#
# def grasp_command(target=None, attempts=5, steps=300):
#     if target is None:
#         target = target_xyz
#
#     print("📡 Запуск серии попыток...")
#     trajectories, errors = generate_trajectories(target, attempts=attempts, steps=steps)
#     best_trajectory = choose_best_trajectory(trajectories, errors)
#
#     if best_trajectory is None:
#         print("❌ Все попытки неудачны.")
#         return
#
#     print("🎬 Воспроизведение лучшей траектории...")
#     joint_indices = list(JOINT_LIMITS.keys())
#     reset_joint_angles([0.0] * len(joint_indices))
#     for _ in range(10):
#         p.stepSimulation()
#     play_trajectory(best_trajectory, joint_indices)
#
# def take():
#     print("🤏 Захват объекта...")
#     grasp_command()
#
# # === CLI ===
# if __name__ == "__main__":
#     while True:
#         cmd = input("Введите команду (grab / take / exit): ").strip().lower()
#         if cmd == "grab":
#             grasp_command()
#         elif cmd == "take":
#             take()
#         elif cmd == "exit":
#             break
#         else:
#             print("❓ Неизвестная команда.")

# import pybullet as p
# import time
# import pybullet_data
# import pandas as pd
# import numpy as np
# from scipy.optimize import minimize
#
# # === Настройки ===
# CSV_PATH = "points.csv"
# TARGET_NAME = "diag_0.3_0.2_0.1"
# CUBE_HALF = 0.03
# SLEEP_TIME = 1 / 240
# FORCE = 80
#
# # === PyBullet Init ===
# p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -9.81)
# p.loadURDF("plane.urdf")
#
# robot_id = p.loadURDF("C:\\Users\\2005s\\PycharmProjects\\FRIDAY_SIM\\BULLET_FRIDAY\\BULLET_FRIDAY.urdf", useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
# grid_id = p.loadURDF("C:\\Users\\2005s\\PycharmProjects\\FRIDAY_SIM\\BULLET_FRIDAY\\BULLET_FRIDAY_with_grid.urdf", useFixedBase=True)
#
# # === Индексы звеньев ===
# link_name_to_index = {p.getBodyInfo(robot_id)[0].decode('utf-8'): -1}
# for i in range(p.getNumJoints(robot_id)):
#     name = p.getJointInfo(robot_id, i)[12].decode('utf-8')
#     link_name_to_index[name] = i
# gripper_inner_index = link_name_to_index["gripper_inner_site"]
#
# # === Цель и куб ===
# target_df = pd.read_csv(CSV_PATH)
# target_xyz = target_df[target_df['name'] == TARGET_NAME][['x', 'y', 'z']].values[0]
#
# cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[CUBE_HALF]*3, rgbaColor=[0, 1, 0, 1])
# cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[CUBE_HALF]*3)
# cube_id = p.createMultiBody(0, cube_collision, cube_visual, basePosition=target_xyz)
#
# # === Ограничения суставов ===
# JOINT_LIMITS = {i: (-np.pi, np.pi) for i in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, i)[2] == p.JOINT_REVOLUTE}
#
# # === Утилиты ===
# def get_site_position(site_index):
#     return np.array(p.getLinkState(robot_id, site_index)[0])
#
# def reset_joint_angles(joint_angles):
#     for i, angle in enumerate(joint_angles):
#         p.resetJointState(robot_id, i, angle)
#     for _ in range(5):
#         p.stepSimulation()
#
# # === Целевая функция
# def objective_phase1(joint_angles, target_pos):
#     reset_joint_angles(joint_angles)
#     for _ in range(3):
#         p.stepSimulation()
#
#     pos_inner = get_site_position(gripper_inner_index)
#     error = np.linalg.norm(pos_inner - target_pos)
#
#     p.addUserDebugLine(pos_inner, target_pos, [0, 1, 0], lineWidth=2, lifeTime=1)
#     p.addUserDebugText(f"{error:.3f}", pos_inner + [0, 0, 0.02], textColorRGB=[1, 0, 0], lifeTime=1)
#
#     # === Контакт обязателен
#     # contacts = p.getContactPoints(bodyA=robot_id, bodyB=cube_id)
#     # touching = any(c[3] == gripper_inner_index or c[4] == gripper_inner_index for c in contacts)
#     #
#     # if not touching:
#     #     return 1e6  # ❌ если не дотронулся — отбраковка
#
#     return error
#
#
# # === Оптимизация
# def evaluate_grasp_pybullet(target_pos, max_attempts=5):
#     joint_indices = list(JOINT_LIMITS.keys())
#     best_qpos = None
#     best_error = float('inf')
#
#     for attempt in range(max_attempts):
#         print(f"⚙️ Попытка {attempt + 1}")
#
#         reset_joint_angles([0.0] * len(joint_indices))
#         for _ in range(3):
#             p.stepSimulation()
#
#         base_guess = np.array([0.0] * len(joint_indices))
#         noise = np.random.uniform(-0.1, 0.1, size=len(joint_indices))
#         initial_guess = np.clip(base_guess + noise,
#                                 [JOINT_LIMITS[i][0] for i in joint_indices],
#                                 [JOINT_LIMITS[i][1] for i in joint_indices])
#
#         result = minimize(
#             objective_phase1,
#             initial_guess,
#             args=(target_pos,),  # 👈 БЕЗ поднятия по Z
#             bounds=[JOINT_LIMITS[i] for i in joint_indices],
#             method='Powell',
#             options={'maxiter': 20}  # ограничим длительность
#         )
#
#         if result.success:
#             error = objective_phase1(result.x, target_pos)
#             print(f"  📏 Ошибка: {error:.4f}")
#             if error < best_error:
#                 best_error = error
#                 best_qpos = result.x
#
#     if best_qpos is None:
#         return None
#
#     return best_qpos.tolist()
#
#
# # === Воспроизведение
# def interpolate_trajectory(start_q, end_q, steps):
#     return [(1 - alpha) * np.array(start_q) + alpha * np.array(end_q) for alpha in np.linspace(0, 1, steps)]
#
# def play_trajectory(trajectory, joint_indices, sleep=SLEEP_TIME):
#     for q in trajectory:
#         for j, idx in enumerate(joint_indices):
#             p.setJointMotorControl2(robot_id, idx, p.POSITION_CONTROL, targetPosition=q[j], force=FORCE)
#         p.stepSimulation()
#         time.sleep(sleep)
#
# def grasp_command(target=None, steps=200):
#     if target is None:
#         target = target_xyz
#
#     print("🚀 Поиск оптимальной траектории...")
#     qpos = evaluate_grasp_pybullet(target)
#
#     if qpos is None:
#         print("❌ Не удалось найти траекторию.")
#         return
#
#     joint_indices = list(JOINT_LIMITS.keys())
#     reset_joint_angles([0.0] * len(joint_indices))
#     for _ in range(5):
#         p.stepSimulation()
#
#     print("🎯 Выполнение траектории...")
#     traj = interpolate_trajectory([0.0]*len(qpos), qpos, steps)
#     play_trajectory(traj, joint_indices)
#
# def take():
#     print("🤏 Захват объекта...")
#     grasp_command()
#
# # === CLI ===
# if __name__ == "__main__":
#     while True:
#         cmd = input("Введите команду (grab / take / exit): ").strip().lower()
#         if cmd == "grab":
#             grasp_command()
#         elif cmd == "take":
#             take()
#         elif cmd == "exit":
#             break
#         else:
#             print("❓ Неизвестная команда.")
#--------------------------Generate The Matrix
# import os
#
# robot_arm_urdf_path = "C:\\Users\\2005s\\Desktop\\BULLET_FRIDAY\\BULLET_FRIDAY.urdf"  # Путь к исходному URDF руки
# output_urdf_path = "C:\\Users\\2005s\\Desktop\\BULLET_FRIDAY\\BULLET_FRIDAY_with_grid.urdf"
#
# # Загрузка URDF робо-руки
# with open(robot_arm_urdf_path, "r", encoding="utf-8") as f:
#     robot_arm_urdf = f.read()
#
# # Параметры сетки
# step = 0.1       # в метрах
# min_val = -0.4
# max_val = 0.4
# point_radius = 0.006  # радиус точки в метрах
#
# # Вспомогательная генерация диапазона с float
# def frange(start, stop, step):
#     while start <= stop:
#         yield round(start, 4)
#         start += step
#
# # Центр XY
# offset_x = -((min_val + max_val) / 2)  # = -0.0
# offset_y = -((min_val + max_val) / 2)  # = -0.0
# offset_z = +abs(min_val)               # = +0.4, чтобы нижняя грань была на z = 0
#
# def generate_visual_point(name, x, y, z, color="1 1 0 1"):
#     link = f"""
#   <link name="{name}">
#     <visual>
#       <origin xyz="{x:.3f} {y:.3f} {z:.3f}" />
#       <geometry>
#         <sphere radius="{point_radius}"/>
#       </geometry>
#       <material name="{name}_mat">
#         <color rgba="{color}" />
#       </material>
#     </visual>
#   </link>
#     """
#     joint = f"""
#   <joint name="joint_{name}" type="fixed">
#     <parent link="base"/>
#     <child link="{name}"/>
#     <origin xyz="{offset_x:.3f} {offset_y:.3f} {offset_z:.3f}"/>
#   </joint>
#     """
#     return link + joint
#
#
# # Генерация всех точек
# point_blocks = ""
#
# # XY
# for x in frange(min_val, max_val, step):
#     for y in frange(min_val, max_val, step):
#         point_blocks += generate_visual_point(f"pt_xy_{x:.2f}_{y:.2f}", x, y, 0)
#
# # XZ
# for x in frange(min_val, max_val, step):
#     for z in frange(min_val, max_val, step):
#         point_blocks += generate_visual_point(f"pt_xz_{x:.2f}_{z:.2f}", x, 0, z)
#
# # YZ
# for y in frange(min_val, max_val, step):
#     for z in frange(min_val, max_val, step):
#         point_blocks += generate_visual_point(f"pt_yz_{y:.2f}_{z:.2f}", 0, y, z)
#
# # Диагонали
# for x in frange(min_val, max_val, step):
#     for y in frange(min_val, max_val, step):
#         for z in frange(min_val, max_val, step):
#             point_blocks += generate_visual_point(f"diag_{x:.2f}_{y:.2f}_{z:.2f}", x, y, z, color="1 0 1 1")
#
# # Финальный URDF
# full_urdf = f"""<?xml version="1.0" ?>
# <robot name="robot_with_grid">
#
# <!-- Робо-рука -->
# {robot_arm_urdf}
#
# <!-- Точки сетки -->
# {point_blocks}
#
# </robot>
# """
#
# # Сохраняем
# with open(output_urdf_path, "w", encoding="utf-8") as f:
#     f.write(full_urdf)
#
# print(f"✅ URDF с визуальной сеткой и фиксированными связями сохранён: {output_urdf_path}")
#
# -------------------------------Extract Points CO
# import csv
#
# # Параметры сетки
# step = 0.1  # в метрах
# min_val = -0.4
# max_val = 0.4
#
# # CSV-выход
# csv_path = "C:\\Users\\2005s\\Desktop\\BULLET_FRIDAY\\points.csv"
#
# # Вспомогательная генерация с шагом float
# def frange(start, stop, step):
#     while start <= stop:
#         yield round(start, 4)
#         start += step
#
# # Собираем строки
# rows = [("name", "x", "y", "z")]
#
# # XY-плоскость
# for x in frange(min_val, max_val, step):
#     for y in frange(min_val, max_val, step):
#         name = f"point_xy_{x}_{y}"
#         rows.append((name, x, y, 0.0))
#
# # XZ-плоскость
# for x in frange(min_val, max_val, step):
#     for z in frange(min_val, max_val, step):
#         name = f"point_xz_{x}_{z}"
#         rows.append((name, x, 0.0, z))
#
# # YZ-плоскость
# for y in frange(min_val, max_val, step):
#     for z in frange(min_val, max_val, step):
#         name = f"point_yz_{y}_{z}"
#         rows.append((name, 0.0, y, z))
#
# # Диагонали
# for x in frange(min_val, max_val, step):
#     for y in frange(min_val, max_val, step):
#         for z in frange(min_val, max_val, step):
#             name = f"diag_{x}_{y}_{z}"
#             rows.append((name, x, y, z))
#
# # Сохраняем в CSV
# with open(csv_path, "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerows(rows)
#
# print(f"✅ CSV с координатами точек сохранён: {csv_path}")
#
#-----------------------------------STEREO 3D MAP
#!/usr/bin/env python3
#
# import depthai as dai
# from time import sleep
# import numpy as np
# import cv2
# import time
# import sys
# try:
#     import open3d as o3d
# except ImportError:
#     sys.exit("Critical dependency missing: Open3D. Please install it using the command: '{} -m pip install open3d' and then rerun the script.".format(sys.executable))
#
# FPS = 30
# class FPSCounter:
#     def __init__(self):
#         self.frameCount = 0
#         self.fps = 0
#         self.startTime = time.time()
#
#     def tick(self):
#         self.frameCount += 1
#         if self.frameCount % 10 == 0:
#             elapsedTime = time.time() - self.startTime
#             self.fps = self.frameCount / elapsedTime
#             self.frameCount = 0
#             self.startTime = time.time()
#         return self.fps
#
# pipeline = dai.Pipeline()
# camRgb = pipeline.create(dai.node.ColorCamera)
# monoLeft = pipeline.create(dai.node.MonoCamera)
# monoRight = pipeline.create(dai.node.MonoCamera)
# depth = pipeline.create(dai.node.StereoDepth)
# pointcloud = pipeline.create(dai.node.PointCloud)
# sync = pipeline.create(dai.node.Sync)
# xOut = pipeline.create(dai.node.XLinkOut)
# xOut.input.setBlocking(False)
#
#
# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
# camRgb.setIspScale(1,3)
# camRgb.setFps(FPS)
#
# monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
# monoLeft.setCamera("left")
# monoLeft.setFps(FPS)
# monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
# monoRight.setCamera("right")
# monoRight.setFps(FPS)
#
# depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
# depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
# depth.setLeftRightCheck(True)
# depth.setExtendedDisparity(False)
# depth.setSubpixel(True)
# depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)
#
# monoLeft.out.link(depth.left)
# monoRight.out.link(depth.right)
# depth.depth.link(pointcloud.inputDepth)
# camRgb.isp.link(sync.inputs["rgb"])
# pointcloud.outputPointCloud.link(sync.inputs["pcl"])
# sync.out.link(xOut.input)
# xOut.setStreamName("out")
#
#
#
#
# with dai.Device(pipeline) as device:
#     isRunning = True
#     def key_callback(vis, action, mods):
#         global isRunning
#         if action == 0:
#             isRunning = False
#
#     q = device.getOutputQueue(name="out", maxSize=4, blocking=False)
#     vis = o3d.visualization.VisualizerWithKeyCallback()
#     vis.create_window()
#     vis.register_key_action_callback(81, key_callback)
#     pcd = o3d.geometry.PointCloud()
#     coordinateFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0,0,0])
#     vis.add_geometry(coordinateFrame)
#
#     first = True
#     fpsCounter = FPSCounter()
#     while isRunning:
#         inMessage = q.get()
#         inColor = inMessage["rgb"]
#         inPointCloud = inMessage["pcl"]
#         cvColorFrame = inColor.getCvFrame()
#         # Convert the frame to RGB
#         cvRGBFrame = cv2.cvtColor(cvColorFrame, cv2.COLOR_BGR2RGB)
#         fps = fpsCounter.tick()
#         # Display the FPS on the frame
#         cv2.putText(cvColorFrame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow("color", cvColorFrame)
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break
#         if inPointCloud:
#             t_before = time.time()
#             points = inPointCloud.getPoints().astype(np.float64)
#             pcd.points = o3d.utility.Vector3dVector(points)
#             colors = (cvRGBFrame.reshape(-1, 3) / 255.0).astype(np.float64)
#             pcd.colors = o3d.utility.Vector3dVector(colors)
#             if first:
#                 vis.add_geometry(pcd)
#                 first = False
#             else:
#                 vis.update_geometry(pcd)
#         vis.poll_events()
#         vis.update_renderer()
#     vis.destroy_window()

# -------------------------------------------------------------PHOTO 3D
#
# import cv2
# import depthai as dai
# import numpy as np
# import argparse
# import time
# import blobconverter
# import open3d as o3d
#
# # --------------- Аргументы ---------------
# parser = argparse.ArgumentParser()
# parser.add_argument("-w", "--width", help="Model input width", default=320, type=int)
# args = parser.parse_args()
#
# # Размер входа модели
# if args.width == 320:
#     NN_WIDTH, NN_HEIGHT = 320, 240
#     NN_PATH = blobconverter.from_zoo(name="depth_estimation_mbnv2_240x320", zoo_type="depthai", shaves=6)
# else:
#     NN_WIDTH, NN_HEIGHT = 640, 480
#     NN_PATH = blobconverter.from_zoo(name="depth_estimation_mbnv2_480x640", zoo_type="depthai", shaves=6)
#
# # --------------- Пайплайн ---------------
# pipeline = dai.Pipeline()
# pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)
#
# cam = pipeline.create(dai.node.ColorCamera)
# cam.setPreviewSize(NN_WIDTH, NN_HEIGHT)
# cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# cam.setInterleaved(False)
# cam.setFps(30)
#
# nn = pipeline.create(dai.node.NeuralNetwork)
# nn.setBlobPath(str(NN_PATH))
# cam.preview.link(nn.input)
#
# xout_cam = pipeline.create(dai.node.XLinkOut)
# xout_cam.setStreamName("cam")
# nn.passthrough.link(xout_cam.input)
#
# xout_nn = pipeline.create(dai.node.XLinkOut)
# xout_nn.setStreamName("nn")
# nn.out.link(xout_nn.input)
#
# # --------------- Внутренние параметры камеры OAK-D Lite ---------------
# # Приблизительные значения (можно уточнить по калибровке)
# fx = 445.0
# fy = 445.0
# cx = NN_WIDTH / 2
# cy = NN_HEIGHT / 2
#
# # --------------- Запуск устройства ---------------
# with dai.Device(pipeline) as device:
#     q_cam = device.getOutputQueue("cam", 4, blocking=False)
#     q_nn = device.getOutputQueue("nn", 4, blocking=False)
#
#     while True:
#         in_cam = q_cam.get()
#         in_nn = q_nn.get()
#
#         frame = in_cam.getCvFrame()
#         rgb_small = cv2.resize(frame, (NN_WIDTH // 2, NN_HEIGHT // 2))
#
#         # Глубина от сети
#         depth_raw = np.array(in_nn.getFirstLayerFp16()).reshape((NN_HEIGHT // 2, NN_WIDTH // 2)).astype(np.float32)
#
#         # Нормализованная карта глубины для отображения
#         depth_vis = ((depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min()) * 255).astype(np.uint8)
#         depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
#         depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
#
#         # Отображение RGB + Depth
#         combined = cv2.hconcat([frame, depth_colored])
#         cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
#         cv2.resizeWindow("Detections", 1280, 480)
#         cv2.imshow("Detections", combined)
#
#         # Генерация облака точек
#         i, j = np.meshgrid(np.arange(NN_WIDTH // 2), np.arange(NN_HEIGHT // 2))
#         z = depth_raw.flatten()
#         x = (i.flatten() - cx / 2) * z / fx
#         y = -(j.flatten() - cy / 2) * z / fy
#         points = np.stack((x, y, z), axis=-1)
#
#         # Цвета из downscaled RGB
#         colors = rgb_small.reshape(-1, 3).astype(np.float32) / 255.0
#
#         # Облако точек через Open3D
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
#         pcd.colors = o3d.utility.Vector3dVector(colors)
#
#         # Визуализация один раз (на каждую Q)
#         o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")
#
#         # Выход
#         if cv2.waitKey(1) == ord('q'):
#             break
#
#     cv2.destroyAllWindows()

# import depthai as dai
# import numpy as np
# import cv2
# from ultralytics import YOLO
# import time
#
# # === Загрузка YOLO модели ===
# model = YOLO("C:/Users/2005s/PycharmProjects/FRIDAY_SIM/yolo11s.pt")
#
# TARGET_CLASS = "cup"
# CONFIDENCE_THRESHOLD = 0.5
# FPS = 30
#
# class FPSCounter:
#     def __init__(self):
#         self.frameCount = 0
#         self.fps = 0
#         self.startTime = time.time()
#
#     def tick(self):
#         self.frameCount += 1
#         if self.frameCount % 10 == 0:
#             elapsedTime = time.time() - self.startTime
#             self.fps = self.frameCount / elapsedTime
#             self.frameCount = 0
#             self.startTime = time.time()
#         return self.fps
#
# # === Pipeline ===
# pipeline = dai.Pipeline()
# camRgb = pipeline.create(dai.node.ColorCamera)
# monoLeft = pipeline.create(dai.node.MonoCamera)
# monoRight = pipeline.create(dai.node.MonoCamera)
# depth = pipeline.create(dai.node.StereoDepth)
# sync = pipeline.create(dai.node.Sync)
# xOut = pipeline.create(dai.node.XLinkOut)
# xOut.setStreamName("out")
# xOut.input.setBlocking(False)
#
# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
# camRgb.setIspScale(1, 3)
# camRgb.setFps(FPS)
#
# monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
# monoLeft.setCamera("left")
# monoLeft.setFps(FPS)
# monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
# monoRight.setCamera("right")
# monoRight.setFps(FPS)
#
# depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
# depth.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
# depth.setLeftRightCheck(True)
# depth.setExtendedDisparity(True)
# depth.setSubpixel(True)
# depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)
#
# monoLeft.out.link(depth.left)
# monoRight.out.link(depth.right)
# camRgb.isp.link(sync.inputs["rgb"])
# depth.depth.link(sync.inputs["depth"])
# sync.out.link(xOut.input)
#
# def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=5):
#     dist = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))
#     for i in range(0, dist, dash_length * 2):
#         start = (
#             int(pt1[0] + (pt2[0] - pt1[0]) * i / dist),
#             int(pt1[1] + (pt2[1] - pt1[1]) * i / dist)
#         )
#         end = (
#             int(pt1[0] + (pt2[0] - pt1[0]) * (i + dash_length) / dist),
#             int(pt1[1] + (pt2[1] - pt1[1]) * (i + dash_length) / dist)
#         )
#         cv2.line(img, start, end, color, thickness)
#
# def draw_dashed_rect(img, pt1, pt2, color, thickness=1):
#     x1, y1 = pt1
#     x2, y2 = pt2
#     draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness)
#     draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness)
#     draw_dashed_line(img, (x2, y2), (x1, y2), color, thickness)
#     draw_dashed_line(img, (x1, y2), (x1, y1), color, thickness)
#
# def draw_3d_box(image, x1, y1, x2, y2, depth_px=10, color=(0,255,0), thickness=2):
#     # Задняя «глубокая» часть бокса
#     offset = depth_px
#     x1_b, y1_b = x1 - offset, y1 - offset
#     x2_b, y2_b = x2 - offset, y2 - offset
#
#     # x1_b, y1_b = x1, y1 - depth_px
#     # x2_b, y2_b = x2, y2 - depth_px
#
#     # Передний прямоугольник
#     cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
#
#     # Задний прямоугольник
#     draw_dashed_rect(image, (x1_b, y1_b), (x2_b, y2_b), color, thickness)
#
#     # Соединительные линии — пунктир
#     draw_dashed_line(image, (x1, y1), (x1_b, y1_b), color, thickness)
#     draw_dashed_line(image, (x2, y1), (x2_b, y1_b), color, thickness)
#     draw_dashed_line(image, (x1, y2), (x1_b, y2_b), color, thickness)
#     draw_dashed_line(image, (x2, y2), (x2_b, y2_b), color, thickness)
#
# # === Основной код ===
# with dai.Device(pipeline) as device:
#     q = device.getOutputQueue(name="out", maxSize=4, blocking=False)
#     fpsCounter = FPSCounter()
#
#     while True:
#         inMessage = q.get()
#         inColor = inMessage["rgb"]
#         inDepth = inMessage["depth"]
#
#         frame = inColor.getCvFrame()
#         depthFrame = inDepth.getFrame()  # depth в миллиметрах
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         fps = fpsCounter.tick()
#
#         # YOLOv8
#         results = model.predict(rgb, verbose=False)[0]
#         cups = [b for b in results.boxes if model.names[int(b.cls[0])] == TARGET_CLASS and b.conf[0] > CONFIDENCE_THRESHOLD]
#
#         for b in cups:
#             x1, y1, x2, y2 = map(int, b.xyxy[0])
#             draw_3d_box(frame, x1, y1, x2, y2, depth_px=30, color=(0,255,0), thickness=2)
#
#
#             # Безопасные границы
#             h, w = depthFrame.shape
#             x1 = max(0, min(x1, w-1))
#             x2 = max(0, min(x2, w-1))
#             y1 = max(0, min(y1, h-1))
#             y2 = max(0, min(y2, h-1))
#
#             roi = depthFrame[y1:y2, x1:x2]
#             if roi.size > 0:
#                 depth_mm = np.median(roi)
#                 depth_m = depth_mm / 1000.0
#                 cv2.putText(frame, f"Depth: {depth_m:.2f}m", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#         cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow("RGB + Depth", frame)
#
#         if cv2.waitKey(1) == ord('q'):
#             break
#
#     cv2.destroyAllWindows()

#------------------------Point cloud with detection and reconstuction of object
import sys
import cv2
import depthai as dai
import numpy as np
import open3d as o3d
from ultralytics import YOLO
from sklearn.decomposition import PCA


yolo = YOLO("C:\\Users\\2005s\\PycharmProjects\\FRIDAY_SIM\\yolo11s.pt")
OBJECT_NAME = "cup"
FPS = 30

# Получение ID объекта по его названию
object_id = next((k for k, v in yolo.names.items() if v == OBJECT_NAME), -1)
if object_id == -1:
    sys.exit(f"Object '{OBJECT_NAME}' not found in YOLO model classes.")


if object_id == -1:
    sys.exit(f"Object '{OBJECT_NAME}' not found in YOLO model classes.")

pipeline = dai.Pipeline()
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
pointcloud = pipeline.create(dai.node.PointCloud)
sync = pipeline.create(dai.node.Sync)
xOut = pipeline.create(dai.node.XLinkOut)
xOut.input.setBlocking(False)

# Конфигурация камеры для близкой дистанции
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setIspScale(1, 3)
camRgb.setFps(FPS)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoLeft.setFps(FPS)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoRight.setFps(FPS)

# Глубина для близкой дистанции
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(True)
depth.setExtendedDisparity(True)
depth.setSubpixel(True)
depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)

monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.depth.link(pointcloud.inputDepth)
camRgb.isp.link(sync.inputs["rgb"])
pointcloud.outputPointCloud.link(sync.inputs["pcl"])
sync.out.link(xOut.input)
xOut.setStreamName("out")

with dai.Device(pipeline) as device:
    q = device.getOutputQueue(name="out", maxSize=4, blocking=False)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()

    first = True

    while True:
        msg = q.get()
        inColor = msg["rgb"]
        inPointCloud = msg["pcl"]

        cvColorFrame = inColor.getCvFrame()
        rgb_frame = cv2.cvtColor(cvColorFrame, cv2.COLOR_BGR2RGB)

        # YOLO Detection
        results = yolo(rgb_frame, verbose=False)[0]

        detected = False
        bbox = None
        for box in results.boxes:
            if int(box.cls.item()) == object_id:
                bbox = box.xyxy[0].int().tolist()
                detected = True
                break  # берём первый найденный объект

        points = inPointCloud.getPoints().astype(np.float64) * 1000  # мм

        # Исправляем переворот по Y
        points[:, 1] = -points[:, 1]

        h, w = rgb_frame.shape[:2]
        colors = rgb_frame.reshape(-1, 3) / 255.0
        if colors.shape[0] != points.shape[0]:
            colors = np.tile([0.5, 0.5, 0.5], (points.shape[0], 1))


        if detected and bbox:
            x1, y1, x2, y2 = bbox
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            mask = mask.flatten()

            colors[mask] = [0, 1, 0]

            obj_points = points[mask]

            if len(obj_points) > 3:
                pca = PCA(n_components=3)
                pca.fit(obj_points)

                main_axis = pca.components_[0]
                proj = np.dot(obj_points, main_axis)
                center_plane = np.mean(proj)

                # Проверяем, где больше точек, впереди или сзади плоскости отражения:
                forward_mask = proj > center_plane
                backward_mask = proj <= center_plane

                # Если впереди точек больше, разворачиваем ось зеркалирования
                if np.sum(forward_mask) > np.sum(backward_mask):
                    main_axis = -main_axis
                    proj = -proj
                    center_plane = -center_plane

                # Теперь зеркалим только те точки, которые сзади (чтобы получить только недостающую половину)
                points_to_mirror = obj_points[backward_mask]
                proj_to_mirror = proj[backward_mask]

                mirrored_points = points_to_mirror - 2 * np.outer((proj_to_mirror - center_plane), main_axis)

                # Добавляем зеркальные точки к сцене
                points = np.vstack([points, mirrored_points])
                colors = np.vstack([colors, np.tile([0, 1, 0], (mirrored_points.shape[0], 1))])

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if first:
            vis.add_geometry(pcd)
            view_ctl = vis.get_view_control()
            view_ctl.set_up([0, -1, 0])
            view_ctl.set_front([0, 0, -1])
            view_ctl.set_lookat([0, 0, 1000])
            view_ctl.set_zoom(0.35)
            first = False
        else:
            vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()

        # Отображение результата YOLO
        cv2.rectangle(cvColorFrame, (x1, y1), (x2, y2), (0, 255, 0), 2) if detected else None
        cv2.imshow("Detection", cvColorFrame)
        if cv2.waitKey(1) == ord('q'):
            break

    vis.destroy_window()
    cv2.destroyAllWindows()

#-------------------try to build in the simulation----ne rabotaet
# import pybullet as p
# import pybullet_data
# import numpy as np
# import cv2
# import depthai as dai
# from ultralytics import YOLO
# from sklearn.decomposition import PCA
#
# # YOLO
# OBJECT_NAME = "cup"
# yolo = YOLO("C:\\Users\\2005s\\PycharmProjects\\FRIDAY_SIM\\yolo11s.pt")
# object_id = next((k for k, v in yolo.names.items() if v == OBJECT_NAME), -1)
#
# pipeline = dai.Pipeline()
# camRgb = pipeline.create(dai.node.ColorCamera)
# monoLeft = pipeline.create(dai.node.MonoCamera)
# monoRight = pipeline.create(dai.node.MonoCamera)
# depth = pipeline.create(dai.node.StereoDepth)
# pointcloud = pipeline.create(dai.node.PointCloud)
# sync = pipeline.create(dai.node.Sync)
# xOut = pipeline.create(dai.node.XLinkOut)
# xOut.input.setBlocking(False)
#
# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
# camRgb.setIspScale(1, 3)
# camRgb.setFps(30)
#
# monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
# monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
# monoLeft.setFps(30)
# monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
# monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
# monoRight.setFps(30)
#
# depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
# depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
# depth.setLeftRightCheck(True)
# depth.setExtendedDisparity(True)
# depth.setSubpixel(True)
# depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)
#
# monoLeft.out.link(depth.left)
# monoRight.out.link(depth.right)
# depth.depth.link(pointcloud.inputDepth)
# camRgb.isp.link(sync.inputs["rgb"])
# pointcloud.outputPointCloud.link(sync.inputs["pcl"])
# sync.out.link(xOut.input)
# xOut.setStreamName("out")
#
# # PyBullet
# physicsClient = p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -9.81)
# p.loadURDF("plane.urdf")
#
# all_green_points = []
#
# with dai.Device(pipeline) as device:
#     q = device.getOutputQueue(name="out", maxSize=4, blocking=False)
#
#     while True:
#         msg = q.get()
#         inColor = msg["rgb"]
#         inPointCloud = msg["pcl"]
#
#         cvColorFrame = inColor.getCvFrame()
#         rgb_frame = cv2.cvtColor(cvColorFrame, cv2.COLOR_BGR2RGB)
#
#         results = yolo(rgb_frame, verbose=False)[0]
#
#         detected = False
#         bbox = None
#         for box in results.boxes:
#             if int(box.cls.item()) == object_id:
#                 bbox = box.xyxy[0].int().tolist()
#                 detected = True
#                 break
#
#         points = inPointCloud.getPoints().astype(np.float64) * 1000
#         points[:, 1] = -points[:, 1]
#
#         h, w = rgb_frame.shape[:2]
#
#         if detected and bbox:
#             x1, y1, x2, y2 = bbox
#             mask = np.zeros((h, w), dtype=bool)
#             mask[y1:y2, x1:x2] = True
#             mask = mask.flatten()
#
#             obj_points = points[mask]
#
#             if len(obj_points) > 3:
#                 pca = PCA(n_components=3)
#                 pca.fit(obj_points)
#
#                 main_axis = pca.components_[0]
#                 proj = np.dot(obj_points, main_axis)
#                 center_plane = np.mean(proj)
#
#                 forward_mask = proj > center_plane
#                 backward_mask = proj <= center_plane
#
#                 if np.sum(forward_mask) > np.sum(backward_mask):
#                     main_axis = -main_axis
#                     proj = -proj
#                     center_plane = -center_plane
#
#                 points_to_mirror = obj_points[backward_mask]
#                 proj_to_mirror = proj[backward_mask]
#
#                 mirrored_points = points_to_mirror - 2 * np.outer((proj_to_mirror - center_plane), main_axis)
#
#                 # Кубики примерно размера объектов на кадре (~3 см)
#                 cube_size = 0.03
#
#                 # Запоминаем все зелёные точки
#                 for point in mirrored_points:
#                     point_meter = point / 1000.0
#                     all_green_points.append(point_meter)
#                     p.loadURDF("cube_small.urdf", basePosition=point_meter,
#                                globalScaling=cube_size, useFixedBase=True)
#
#         # PyBullet simulation step
#         p.stepSimulation()
#
#         # YOLO box
#         if detected:
#             cv2.rectangle(cvColorFrame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#         cv2.imshow("Detection", cvColorFrame)
#
#         if cv2.waitKey(1) == ord('q'):
#             break
#
# cv2.destroyAllWindows()
# p.disconnect()

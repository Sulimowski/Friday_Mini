import numpy as np
import mujoco
import mujoco.viewer
import time
import os
import pandas as pd
from scipy.optimize import minimize


# Пути к файлам
MODEL_PATH = "C:\\Users\\2005s\\PycharmProjects\\FRIDAY_MINI_SOFT\\RoboArm\\FR_MINI_SIM.xml"
NEW_MODEL_PATH = "C:\\Users\\2005s\\PycharmProjects\\FRIDAY_MINI_SOFT\\RoboArm\\modified_robot.xml"
CSV_PATH = "C:\\Users\\2005s\\PycharmProjects\\STEP_URDF_CONVERTOR\\zones.csv"

# === Настройки ===
CUBE_SIZE = (10, 20, 120)
selected_zone_name = 'diag_xyz_-100_100_0'
ALIGNMENT_THRESHOLD = 10.0
GRIPPER_PARALLEL_WEIGHT = 100

# Загружаем зоны из CSV
zones_df = pd.read_csv(CSV_PATH)
if selected_zone_name not in zones_df['name'].values:
    raise ValueError(f"Зона '{selected_zone_name}' не найдена в CSV!")
selected_zone = zones_df[zones_df['name'] == selected_zone_name].iloc[0]
TARGET_POS = np.array([selected_zone['x'], selected_zone['y'], selected_zone['z']])

# Загружаем XML модель
with open(MODEL_PATH, "r", encoding="utf-8") as f:
    xml_content = f.read()

# Добавляем новый объект (кубик)
new_geom = f"""
    <geom name="target_cube" type="box" size="{CUBE_SIZE[0]} {CUBE_SIZE[1]} {CUBE_SIZE[2]}"
          pos="{TARGET_POS[0]} {TARGET_POS[1]} {TARGET_POS[2]}" rgba="0 1 0 1" contype="1" conaffinity="1"/>
"""

# Вставляем в worldbody
index = xml_content.rfind("</worldbody>")
if index != -1:
    xml_content = xml_content[:index] + new_geom + xml_content[index:]

# Сохраняем обновленный XML-файл
with open(NEW_MODEL_PATH, "w", encoding="utf-8") as f:
    f.write(xml_content)

# Загружаем обновленную модель MuJoCo
model = mujoco.MjModel.from_xml_path(NEW_MODEL_PATH)
data = mujoco.MjData(model)

model.opt.gravity = np.array([0, 0, -9.81])  # Включает гравитацию
data.qvel[:] = 2

# Определяем индексы частей хвата
gripper_inner_site_id = model.site("gripper_inner_site").id
gripper_upper_site_id = model.site("gripper_upper_site").id

# Ограничения суставов
JOINT_LIMITS = {f'joint_{i}': (-90, 90) for i in range(5)}
JOINT_LIMITS['joint_5'] = (-15, 90)
JOINT_LIMITS_RAD = {k: (np.radians(v[0]), np.radians(v[1])) for k, v in JOINT_LIMITS.items()}

TOLERANCE = 10.0
MAX_ANGLE_STEP = np.radians(1)
SLEEP_TIME = 0.005


def objective(joint_angles, target_pos, phase=1):
    """ Функция ошибки для разных фаз:
    - phase=1: Движение к первой точке касания (нижняя зона)
    - phase=2: Захват объекта (верхняя зона) """

    for i, key in enumerate(JOINT_LIMITS.keys()):
        data.qpos[i] = np.clip(joint_angles[i], *JOINT_LIMITS_RAD[key])
    mujoco.mj_forward(model, data)

    gripper_inner_pos = data.site_xpos[gripper_inner_site_id]
    gripper_upper_pos = data.site_xpos[gripper_upper_site_id]

    # Ошибка по расстоянию
    error_inner = np.linalg.norm(gripper_inner_pos - target_pos)
    error_upper = np.linalg.norm(gripper_upper_pos - (target_pos + np.array([0, 0, 10])))

    if phase == 1:
        return error_inner  # Минимизируем ошибку касания нижней зоной
    else:
        return error_upper  # Минимизируем ошибку касания верхней зоной


def evaluate_grasp(max_attempts=20):
    """Оптимизация движения в два этапа:
    1. Касание нижней зоны (но теперь выше на 70%)
    2. Закрытие захвата (уменьшение joint_5)
    """

    best_qpos_phase1 = None
    best_error = float('inf')

    for attempt in range(max_attempts):
        print(f"⚙️ Попытка {attempt + 1} (первая фаза)")

        initial_guess = np.random.uniform(
            [v[0] for v in JOINT_LIMITS_RAD.values()],
            [v[1] for v in JOINT_LIMITS_RAD.values()]
        )

        # ⬆️ Поднимаем цель на 70% от высоты объекта
        adjusted_target_pos = TARGET_POS.copy()
        adjusted_target_pos[2] += CUBE_SIZE[2] * 0.7

        result_phase1 = minimize(objective, initial_guess, args=(adjusted_target_pos, 1),
                                 bounds=list(JOINT_LIMITS_RAD.values()), method='Powell')

        if result_phase1.success:
            error_phase1 = objective(result_phase1.x, adjusted_target_pos, 1)
            if error_phase1 < best_error:
                best_qpos_phase1 = result_phase1.x
                best_error = error_phase1

    if best_qpos_phase1 is None:
        return None  # Не удалось найти начальное касание

    # 🔄 Вторая фаза (просто закрываем joint_5)
    best_qpos_phase2 = best_qpos_phase1.copy()
    best_qpos_phase2[-1] = np.radians(-15)  # Закрываем joint_5

    return best_qpos_phase1, best_qpos_phase2


def execute_trajectory(qpos, viewer, steps=100):
    """ Плавно выполняем траекторию """
    current_qpos = np.array(data.qpos[:])
    trajectory = np.linspace(current_qpos, qpos, steps)

    for step in trajectory:
        data.qpos[:] = step  # Обновляем все углы сразу
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        time.sleep(SLEEP_TIME)  # Даем время для плавного движения
        viewer.sync()

# === Основная логика ===
optimal_qpos_phase1, optimal_qpos_phase2 = evaluate_grasp()
if optimal_qpos_phase1 is None:
    raise RuntimeError("Не удалось найти оптимальные позиции!")

print("✅ Первая фаза найдена, запускаем симуляцию!")

initial_qpos = np.array([0, 0, 0, 0, 0, 0])  # Задай нужные углы
data.qpos[:] = initial_qpos

model.opt.solver = mujoco.mjtSolver.mjSOL_PGS  # Улучшает обработку контактов
model.opt.noslip_iterations = 10  # Увеличивает стабильность контактов
TARGET_POS[2] += CUBE_SIZE[2] * 0.7  # Смещаем цель на 70% от высоты объекта

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Настройки камеры
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20
    viewer.cam.distance = 1750.0
    viewer.cam.lookat[:] = TARGET_POS
    viewer.cam.trackbodyid = -1  # Фиксируем камеру
      # Угол обзора
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # Свободное управление камерой



    execute_trajectory(optimal_qpos_phase1, viewer, steps=150)  # Увеличено количество шагов для плавности
    time.sleep(0.5)  # Пауза перед захватом

    # Фаза 2: Захват верхней зоной
    execute_trajectory(optimal_qpos_phase2, viewer, steps=50)
    print("🛑 Завершена оптимальная траектория. Оставляем рендер открытым.")
    while viewer.is_running():  # Удерживаем окно открытым
        viewer.sync()

if os.path.exists(NEW_MODEL_PATH):
    os.remove(NEW_MODEL_PATH)
    print(f"🗑️ Файл {NEW_MODEL_PATH} удален.")

#-------------------------------Active simul NIE RABOTAET
# import numpy as np
# import mujoco
# import mujoco.viewer
# import time
# import os
# import pandas as pd
# from scipy.optimize import minimize
#
#
# # Пути к файлам
# MODEL_PATH = "C:\\Users\\2005s\\Desktop\\FRIDAY_MINI_SIM\\FR_MINI_SIM.xml"
# NEW_MODEL_PATH = "C:\\Users\\2005s\\Desktop\\FRIDAY_MINI_SIM\\modified_robot.xml"
# CSV_PATH = "C:\\Users\\2005s\\PycharmProjects\\STEP_URDF_CONVERTOR\\zones.csv"
#
# # === Настройки ===
# CUBE_SIZE = (10, 20, 120)
# selected_zone_name = 'diag_xyz_-100_100_0'
# ALIGNMENT_THRESHOLD = 10.0
# GRIPPER_PARALLEL_WEIGHT = 100
#
# # Загружаем зоны из CSV
# zones_df = pd.read_csv(CSV_PATH)
# if selected_zone_name not in zones_df['name'].values:
#     raise ValueError(f"Зона '{selected_zone_name}' не найдена в CSV!")
# selected_zone = zones_df[zones_df['name'] == selected_zone_name].iloc[0]
# TARGET_POS = np.array([selected_zone['x'], selected_zone['y'], selected_zone['z']])
#
# # Загружаем XML модель
# with open(MODEL_PATH, "r", encoding="utf-8") as f:
#     xml_content = f.read()
#
# # Добавляем новый объект (кубик)
# new_geom = f"""
#     <geom name="target_cube" type="box" size="{CUBE_SIZE[0]} {CUBE_SIZE[1]} {CUBE_SIZE[2]}"
#           pos="{TARGET_POS[0]} {TARGET_POS[1]} {TARGET_POS[2]}" rgba="0 1 0 1" contype="1" conaffinity="1"/>
# """
#
# # Вставляем в worldbody
# index = xml_content.rfind("</worldbody>")
# if index != -1:
#     xml_content = xml_content[:index] + new_geom + xml_content[index:]
#
# # Сохраняем обновленный XML-файл
# with open(NEW_MODEL_PATH, "w", encoding="utf-8") as f:
#     f.write(xml_content)
#
# # Загружаем обновленную модель MuJoCo
# model = mujoco.MjModel.from_xml_path(NEW_MODEL_PATH)
# data = mujoco.MjData(model)
#
# model.opt.gravity = np.array([0, 0, -9.81])  # Включает гравитацию
# data.qvel[:] = 2
#
# # Определяем индексы частей хвата
# gripper_inner_site_id = model.site("gripper_inner_site").id
# gripper_upper_site_id = model.site("gripper_upper_site").id
#
# # Ограничения суставов
# JOINT_LIMITS = {f'joint_{i}': (-90, 90) for i in range(5)}
# JOINT_LIMITS['joint_5'] = (-15, 90)
# JOINT_LIMITS_RAD = {k: (np.radians(v[0]), np.radians(v[1])) for k, v in JOINT_LIMITS.items()}
#
# TOLERANCE = 10.0
# MAX_ANGLE_STEP = np.radians(1)
# SLEEP_TIME = 0.005
#
#
# def objective(joint_angles, target_pos, phase=1):
#     """ Функция ошибки для разных фаз:
#     - phase=1: Движение к первой точке касания (нижняя зона)
#     - phase=2: Захват объекта (верхняя зона) """
#
#     for i, key in enumerate(JOINT_LIMITS.keys()):
#         data.qpos[i] = np.clip(joint_angles[i], *JOINT_LIMITS_RAD[key])
#     mujoco.mj_forward(model, data)
#
#     gripper_inner_pos = data.site_xpos[gripper_inner_site_id]
#     gripper_upper_pos = data.site_xpos[gripper_upper_site_id]
#
#     # Ошибка по расстоянию
#     error_inner = np.linalg.norm(gripper_inner_pos - target_pos)
#     error_upper = np.linalg.norm(gripper_upper_pos - (target_pos + np.array([0, 0, 10])))
#
#     if phase == 1:
#         return error_inner  # Минимизируем ошибку касания нижней зоной
#     else:
#         return error_upper  # Минимизируем ошибку касания верхней зоной
#
#
# def evaluate_grasp(max_attempts=20):
#     """Оптимизация движения в два этапа:
#     1. Касание нижней зоны (но теперь выше на 70%)
#     2. Закрытие захвата (уменьшение joint_5)
#     """
#
#     best_qpos_phase1 = None
#     best_error = float('inf')
#
#     for attempt in range(max_attempts):
#         print(f"⚙️ Попытка {attempt + 1} (первая фаза)")
#
#         initial_guess = np.random.uniform(
#             [v[0] for v in JOINT_LIMITS_RAD.values()],
#             [v[1] for v in JOINT_LIMITS_RAD.values()]
#         )
#
#         # ⬆️ Поднимаем цель на 70% от высоты объекта
#         adjusted_target_pos = TARGET_POS.copy()
#         adjusted_target_pos[2] += CUBE_SIZE[2] * 0.7
#
#         result_phase1 = minimize(objective, initial_guess, args=(adjusted_target_pos, 1),
#                                  bounds=list(JOINT_LIMITS_RAD.values()), method='Powell')
#
#         if result_phase1.success:
#             error_phase1 = objective(result_phase1.x, adjusted_target_pos, 1)
#             if error_phase1 < best_error:
#                 best_qpos_phase1 = result_phase1.x
#                 best_error = error_phase1
#
#     if best_qpos_phase1 is None:
#         return None  # Не удалось найти начальное касание
#
#     # 🔄 Вторая фаза (просто закрываем joint_5)
#     best_qpos_phase2 = best_qpos_phase1.copy()
#     best_qpos_phase2[-1] = np.radians(-15)  # Закрываем joint_5
#
#     return best_qpos_phase1, best_qpos_phase2
#
#
# def execute_trajectory(qpos, viewer, steps=100):
#     """ Плавно выполняем траекторию """
#     current_qpos = np.array(data.qpos[:])
#     trajectory = np.linspace(current_qpos, qpos, steps)
#
#     for step in trajectory:
#         data.qpos[:] = step  # Обновляем все углы сразу
#         mujoco.mj_forward(model, data)
#         mujoco.mj_step(model, data)
#         time.sleep(SLEEP_TIME)  # Даем время для плавного движения
#         viewer.sync()
#
# # === Основная логика ===
# optimal_qpos_phase1, optimal_qpos_phase2 = evaluate_grasp()
# if optimal_qpos_phase1 is None:
#     raise RuntimeError("Не удалось найти оптимальные позиции!")
#
# print("✅ Первая фаза найдена, запускаем симуляцию!")
#
# initial_qpos = np.array([0, 0, 0, 0, 0, 0])  # Задай нужные углы
# data.qpos[:] = initial_qpos
#
# model.opt.solver = mujoco.mjtSolver.mjSOL_PGS  # Улучшает обработку контактов
# model.opt.noslip_iterations = 10  # Увеличивает стабильность контактов
# TARGET_POS[2] += CUBE_SIZE[2] * 0.7  # Смещаем цель на 70% от высоты объекта
#
# with mujoco.viewer.launch(model, data) as viewer:
#     # Настройки камеры
#     viewer.cam.azimuth = 90
#     viewer.cam.elevation = -20
#     viewer.cam.distance = 1750.0
#     viewer.cam.lookat[:] = TARGET_POS
#     viewer.cam.trackbodyid = -1  # Фиксируем камеру
#     # Угол обзора
#     viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # Свободное управление камерой
#
#     execute_trajectory(optimal_qpos_phase1, viewer, steps=150)  # Увеличено количество шагов для плавности
#     time.sleep(0.5)  # Пауза перед захватом
#
#     # Фаза 2: Захват верхней зоной
#     execute_trajectory(optimal_qpos_phase2, viewer, steps=50)
#     print("🛑 Завершена оптимальная траектория. Оставляем рендер открытым.")
#     while viewer.is_running():  # Удерживаем окно открытым
#         viewer.sync()
#
# if os.path.exists(NEW_MODEL_PATH):
#     os.remove(NEW_MODEL_PATH)
#     print(f"🗑️ Файл {NEW_MODEL_PATH} удален.")
#--------------------------------------------------TRY TO BE BLIND

#
# import numpy as np
# import mujoco
# import mujoco.viewer
# import time
# import os
# import pandas as pd
# from scipy.optimize import minimize
#
# # Пути к файлам
# MODEL_PATH = "C:\\Users\\2005s\\Desktop\\FRIDAY_MINI_SIM\\FR_MINI_SIM.xml"
# NEW_MODEL_PATH = "C:\\Users\\2005s\\Desktop\\FRIDAY_MINI_SIM\\modified_robot.xml"
# CSV_PATH = "C:\\Users\\2005s\\PycharmProjects\\STEP_URDF_CONVERTOR\\zones.csv"
#
# # === Настройки ===
# CUBE_SIZE = (10, 20, 120)
# selected_zone_name = 'diag_xyz_-100_100_0'
# ALIGNMENT_THRESHOLD = 10.0
# GRIPPER_PARALLEL_WEIGHT = 100
#
# # Загружаем зоны из CSV
# zones_df = pd.read_csv(CSV_PATH)
# if selected_zone_name not in zones_df['name'].values:
#     raise ValueError(f"Зона '{selected_zone_name}' не найдена в CSV!")
# selected_zone = zones_df[zones_df['name'] == selected_zone_name].iloc[0]
# TARGET_POS = np.array([selected_zone['x'], selected_zone['y'], selected_zone['z']])
#
# # Загружаем XML модель
# with open(MODEL_PATH, "r", encoding="utf-8") as f:
#     xml_content = f.read()
#
# # Добавляем новый объект (кубик)
# new_geom = f"""
#     <geom name="target_cube" type="box" size="{CUBE_SIZE[0]} {CUBE_SIZE[1]} {CUBE_SIZE[2]}"
#           pos="{TARGET_POS[0]} {TARGET_POS[1]} {TARGET_POS[2]}" rgba="0 1 0 1" contype="1" conaffinity="1"/>
# """
#
# # Вставляем в worldbody
# index = xml_content.rfind("</worldbody>")
# if index != -1:
#     xml_content = xml_content[:index] + new_geom + xml_content[index:]
#
# # Сохраняем обновленный XML-файл
# with open(NEW_MODEL_PATH, "w", encoding="utf-8") as f:
#     f.write(xml_content)
#
# # Загружаем обновленную модель MuJoCo
# model = mujoco.MjModel.from_xml_path(NEW_MODEL_PATH)
# data = mujoco.MjData(model)
#
# model.opt.gravity = np.array([0, 0, -9.81])
# model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
# model.opt.noslip_iterations = 50
# model.opt.timestep = 0.002
# data.qvel[:] = 2
#
# # Определяем индексы частей хвата
# gripper_inner_site_id = model.site("gripper_inner_site").id
# gripper_upper_site_id = model.site("gripper_upper_site").id
#
# # Ограничения суставов
# JOINT_LIMITS = {f'joint_{i}': (-90, 90) for i in range(5)}
# JOINT_LIMITS['joint_5'] = (-15, 90)
# JOINT_LIMITS_RAD = {k: (np.radians(v[0]), np.radians(v[1])) for k, v in JOINT_LIMITS.items()}
#
# TOLERANCE = 10.0
# MAX_ANGLE_STEP = np.radians(0.5)
# SLEEP_TIME = 0.002
#
#
# def objective(joint_angles, target_pos, phase=1):
#     for i, key in enumerate(JOINT_LIMITS.keys()):
#         data.qpos[i] = np.clip(joint_angles[i], *JOINT_LIMITS_RAD[key])
#     mujoco.mj_forward(model, data)
#
#     gripper_inner_pos = data.site_xpos[gripper_inner_site_id]
#     gripper_upper_pos = data.site_xpos[gripper_upper_site_id]
#
#     error_inner = np.linalg.norm(gripper_inner_pos - target_pos)
#     error_upper = np.linalg.norm(gripper_upper_pos - (target_pos + np.array([0, 0, 10])))
#
#     if phase == 0:
#         return abs(gripper_inner_pos[2] - target_pos[2])
#     elif phase == 1:
#         return error_inner
#     else:
#         return error_upper + GRIPPER_PARALLEL_WEIGHT * abs(gripper_upper_pos[2] - gripper_inner_pos[2])
#
#
# def evaluate_grasp(max_attempts=30):
#     best_qpos_phase0 = None
#     best_qpos_phase1 = None
#     best_error = float('inf')
#
#     for attempt in range(max_attempts):
#         initial_guess = np.random.uniform(
#             [v[0] for v in JOINT_LIMITS_RAD.values()],
#             [v[1] for v in JOINT_LIMITS_RAD.values()]
#         )
#
#         adjusted_target_pos = TARGET_POS.copy()
#         adjusted_target_pos[2] += CUBE_SIZE[2] * 0.7
#
#         result_phase0 = minimize(objective, initial_guess, args=(adjusted_target_pos, 0),
#                                  bounds=list(JOINT_LIMITS_RAD.values()), method='Powell')
#
#         if result_phase0.success:
#             best_qpos_phase0 = result_phase0.x
#
#         result_phase1 = minimize(objective, best_qpos_phase0, args=(adjusted_target_pos, 1),
#                                  bounds=list(JOINT_LIMITS_RAD.values()), method='Powell')
#
#         if result_phase1.success:
#             error_phase1 = objective(result_phase1.x, adjusted_target_pos, 1)
#             if error_phase1 < best_error:
#                 best_qpos_phase1 = result_phase1.x
#                 best_error = error_phase1
#
#     if best_qpos_phase1 is None:
#         return None
#
#     best_qpos_phase2 = best_qpos_phase1.copy()
#     best_qpos_phase2[-1] = np.radians(-15)
#
#     return best_qpos_phase0, best_qpos_phase1, best_qpos_phase2
#
#
# def execute_trajectory(qpos, viewer, steps=150):
#     current_qpos = np.array(data.qpos[:])
#     trajectory = np.linspace(current_qpos, qpos, steps)
#
#     for step in trajectory:
#         data.qpos[:] = step
#         mujoco.mj_forward(model, data)
#         mujoco.mj_step(model, data)
#         time.sleep(SLEEP_TIME)
#         viewer.sync()
#
#
# optimal_qpos_phase0, optimal_qpos_phase1, optimal_qpos_phase2 = evaluate_grasp()
# if optimal_qpos_phase0 is None:
#     raise RuntimeError("Не удалось найти оптимальные позиции!")
#
# initial_qpos = np.array([0, 0, 0, 0, 0, 0])
# data.qpos[:] = initial_qpos
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     viewer.cam.azimuth = 90
#     viewer.cam.elevation = -20
#     viewer.cam.distance = 1750.0
#     viewer.cam.lookat[:] = TARGET_POS
#     viewer.cam.trackbodyid = -1
#     viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
#     execute_trajectory(optimal_qpos_phase0, viewer, steps=150)
#     execute_trajectory(optimal_qpos_phase1, viewer, steps=200)
#     time.sleep(0.5)
#     execute_trajectory(optimal_qpos_phase2, viewer, steps=100)
#     while viewer.is_running():
#         viewer.sync()
#
# if os.path.exists(NEW_MODEL_PATH):
#     os.remove(NEW_MODEL_PATH)
#--------------------------------------VAR 2 More stabil
# import numpy as np
# import mujoco
# import mujoco.viewer
# import time
# import os
# import pandas as pd
# from scipy.optimize import minimize
#
# # Пути к файлам
# MODEL_PATH = "C:\\Users\\2005s\\Desktop\\FRIDAY_MINI_SIM\\FR_MINI_SIM.xml"
# NEW_MODEL_PATH = "C:\\Users\\2005s\\Desktop\\FRIDAY_MINI_SIM\\modified_robot.xml"
# CSV_PATH = "C:\\Users\\2005s\\PycharmProjects\\STEP_URDF_CONVERTOR\\zones.csv"
#
# # === Настройки ===
# CUBE_SIZE = (10, 20, 120)
# selected_zone_name = 'diag_xyz_-100_100_0'
# ALIGNMENT_THRESHOLD = 10.0
# GRIPPER_PARALLEL_WEIGHT = 100
#
# # Загружаем зоны из CSV
# zones_df = pd.read_csv(CSV_PATH)
# if selected_zone_name not in zones_df['name'].values:
#     raise ValueError(f"Зона '{selected_zone_name}' не найдена в CSV!")
# selected_zone = zones_df[zones_df['name'] == selected_zone_name].iloc[0]
# TARGET_POS = np.array([selected_zone['x'], selected_zone['y'], selected_zone['z']])
#
# # Загружаем XML модель
# with open(MODEL_PATH, "r", encoding="utf-8") as f:
#     xml_content = f.read()
#
# # Добавляем новый объект (кубик)
# new_geom = f"""
#     <geom name="target_cube" type="box" size="{CUBE_SIZE[0]} {CUBE_SIZE[1]} {CUBE_SIZE[2]}"
#           pos="{TARGET_POS[0]} {TARGET_POS[1]} {TARGET_POS[2]}" rgba="0 1 0 1" contype="1" conaffinity="1"/>
# """
#
# # Вставляем в worldbody
# index = xml_content.rfind("</worldbody>")
# if index != -1:
#     xml_content = xml_content[:index] + new_geom + xml_content[index:]
#
# # Сохраняем обновленный XML-файл
# with open(NEW_MODEL_PATH, "w", encoding="utf-8") as f:
#     f.write(xml_content)
#
# # Загружаем обновленную модель MuJoCo
# model = mujoco.MjModel.from_xml_path(NEW_MODEL_PATH)
# data = mujoco.MjData(model)
#
# model.opt.gravity = np.array([0, 0, -9.81])
# model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
# model.opt.noslip_iterations = 50
# model.opt.timestep = 0.002
# data.qvel[:] = 2
#
# # Определяем индексы частей хвата
# gripper_inner_site_id = model.site("gripper_inner_site").id
# gripper_upper_site_id = model.site("gripper_upper_site").id
#
# # Ограничения суставов
# JOINT_LIMITS = {f'joint_{i}': (-90, 90) for i in range(5)}
# JOINT_LIMITS['joint_5'] = (-15, 90)
# JOINT_LIMITS_RAD = {k: (np.radians(v[0]), np.radians(v[1])) for k, v in JOINT_LIMITS.items()}
#
# TOLERANCE = 10.0
# MAX_ANGLE_STEP = np.radians(0.5)
# SLEEP_TIME = 0.002
#
#
# def objective(joint_angles, target_pos, phase=1):
#     for i, key in enumerate(JOINT_LIMITS.keys()):
#         data.qpos[i] = np.clip(joint_angles[i], *JOINT_LIMITS_RAD[key])
#     mujoco.mj_forward(model, data)
#
#     gripper_inner_pos = data.site_xpos[gripper_inner_site_id]
#     gripper_upper_pos = data.site_xpos[gripper_upper_site_id]
#
#     error_inner = np.linalg.norm(gripper_inner_pos - target_pos)
#     error_upper = np.linalg.norm(gripper_upper_pos - (target_pos + np.array([0, 0, 10])))
#
#     # Штраф за отклонение захвата
#     penalty = 0
#     if phase == 1:
#         # Проверка за лишние движения: находим изменение положения захвата
#         prev_inner_pos = getattr(objective, 'prev_inner_pos', None)
#         if prev_inner_pos is not None:
#             penalty = np.linalg.norm(gripper_inner_pos - prev_inner_pos) * 10  # Штраф за большой шаг
#         objective.prev_inner_pos = gripper_inner_pos
#
#     if phase == 0:
#         return abs(gripper_inner_pos[2] - target_pos[2]) + penalty
#     elif phase == 1:
#         return error_inner + penalty
#     else:
#         return error_upper + GRIPPER_PARALLEL_WEIGHT * abs(gripper_upper_pos[2] - gripper_inner_pos[2])
#
#
# def evaluate_grasp(max_attempts=30):
#     attempts = []
#
#     for attempt in range(max_attempts):
#         initial_guess = np.random.uniform(
#             [v[0] for v in JOINT_LIMITS_RAD.values()],
#             [v[1] for v in JOINT_LIMITS_RAD.values()]
#         )
#
#         adjusted_target_pos = TARGET_POS.copy()
#         adjusted_target_pos[2] += CUBE_SIZE[2] * 0.7
#
#         result_phase0 = minimize(objective, initial_guess, args=(adjusted_target_pos, 0),
#                                  bounds=list(JOINT_LIMITS_RAD.values()), method='Powell')
#
#         if result_phase0.success:
#             best_qpos_phase0 = result_phase0.x
#
#         result_phase1 = minimize(objective, best_qpos_phase0, args=(adjusted_target_pos, 1),
#                                  bounds=list(JOINT_LIMITS_RAD.values()), method='Powell')
#
#         if result_phase1.success:
#             error_phase1 = objective(result_phase1.x, adjusted_target_pos, 1)
#             attempts.append((result_phase0.x, result_phase1.x, error_phase1))
#
#     # Сортируем попытки по успешности и количеству шагов
#     attempts.sort(key=lambda x: (x[2], np.linalg.norm(x[0] - x[1])))
#
#     if len(attempts) == 0:
#         return None
#
#     best_qpos_phase0, best_qpos_phase1, _ = attempts[0]
#     best_qpos_phase2 = best_qpos_phase1.copy()
#     best_qpos_phase2[-1] = np.radians(-15)
#
#     return best_qpos_phase0, best_qpos_phase1, best_qpos_phase2
#
#
# def execute_trajectory(qpos, viewer, steps=150):
#     current_qpos = np.array(data.qpos[:])
#     trajectory = np.linspace(current_qpos, qpos, steps)
#
#     for step in trajectory:
#         data.qpos[:] = step
#         mujoco.mj_forward(model, data)
#         mujoco.mj_step(model, data)
#         time.sleep(SLEEP_TIME)
#         viewer.sync()
#
#
# optimal_qpos_phase0, optimal_qpos_phase1, optimal_qpos_phase2 = evaluate_grasp()
# if optimal_qpos_phase0 is None:
#     raise RuntimeError("Не удалось найти оптимальные позиции!")
#
# initial_qpos = np.array([0, 0, 0, 0, 0, 0])
# data.qpos[:] = initial_qpos
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     viewer.cam.azimuth = 90
#     viewer.cam.elevation = -20
#     viewer.cam.distance = 1750.0
#     viewer.cam.lookat[:] = TARGET_POS
#     viewer.cam.trackbodyid = -1
#     viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
#     execute_trajectory(optimal_qpos_phase0, viewer, steps=150)
#     execute_trajectory(optimal_qpos_phase1, viewer, steps=200)
#     time.sleep(0.5)
#     execute_trajectory(optimal_qpos_phase2, viewer, steps=100)
#     while viewer.is_running():
#         viewer.sync()
#
# if os.path.exists(NEW_MODEL_PATH):
#     os.remove(NEW_MODEL_PATH)

#-----------------------------------+- STABIL
# import numpy as np
# import mujoco
# import mujoco.viewer
# import time
# import os
# import pandas as pd
# from scipy.optimize import minimize
# from scipy.interpolate import CubicSpline
#
# # Пути к файлам
# MODEL_PATH = "C:\\Users\\2005s\\Desktop\\FRIDAY_MINI_SIM\\FR_MINI_SIM.xml"
# NEW_MODEL_PATH = "C:\\Users\\2005s\\Desktop\\FRIDAY_MINI_SIM\\modified_robot.xml"
# CSV_PATH = "C:\\Users\\2005s\\PycharmProjects\\STEP_URDF_CONVERTOR\\zones.csv"
#
# # === Настройки ===
# CUBE_SIZE = (10, 20, 120)
# selected_zone_name = 'diag_xyz_-100_100_0'
# ALIGNMENT_THRESHOLD = 10.0
# GRIPPER_PARALLEL_WEIGHT = 100
#
# # Загружаем зоны из CSV
# zones_df = pd.read_csv(CSV_PATH)
# if selected_zone_name not in zones_df['name'].values:
#     raise ValueError(f"Зона '{selected_zone_name}' не найдена в CSV!")
# selected_zone = zones_df[zones_df['name'] == selected_zone_name].iloc[0]
# TARGET_POS = np.array([selected_zone['x'], selected_zone['y'], selected_zone['z']])
#
# # Загружаем XML модель
# with open(MODEL_PATH, "r", encoding="utf-8") as f:
#     xml_content = f.read()
#
# # Добавляем новый объект (кубик)
# new_geom = f"""
#     <geom name="target_cube" type="box" size="{CUBE_SIZE[0]} {CUBE_SIZE[1]} {CUBE_SIZE[2]}"
#           pos="{TARGET_POS[0]} {TARGET_POS[1]} {TARGET_POS[2]}" rgba="0 1 0 1" contype="1" conaffinity="1"/>
# """
#
# # Вставляем в worldbody
# index = xml_content.rfind("</worldbody>")
# if index != -1:
#     xml_content = xml_content[:index] + new_geom + xml_content[index:]
#
# # Сохраняем обновленный XML-файл
# with open(NEW_MODEL_PATH, "w", encoding="utf-8") as f:
#     f.write(xml_content)
#
# # Загружаем обновленную модель MuJoCo
# model = mujoco.MjModel.from_xml_path(NEW_MODEL_PATH)
# data = mujoco.MjData(model)
#
# model.opt.gravity = np.array([0, 0, -9.81])
# model.opt.solver = mujoco.mjtSolver.mjSOL_CG  # Используем более стабильный решатель
# model.opt.noslip_iterations = 100  # Увеличиваем количество итераций решателя
# model.opt.timestep = 0.002
# data.qvel[:] = 2
#
# # Определяем индексы частей хвата
# gripper_inner_site_id = model.site("gripper_inner_site").id
# gripper_upper_site_id = model.site("gripper_upper_site").id
#
# # Ограничения суставов
# JOINT_LIMITS = {f'joint_{i}': (-90, 90) for i in range(5)}
# JOINT_LIMITS['joint_5'] = (-15, 90)
# JOINT_LIMITS_RAD = {k: (np.radians(v[0]), np.radians(v[1])) for k, v in JOINT_LIMITS.items()}
#
# TOLERANCE = 10.0
# MAX_ANGLE_STEP = np.radians(0.5)
# SLEEP_TIME = 0.002
#
#
# def objective(joint_angles, target_pos, phase=1):
#     for i, key in enumerate(JOINT_LIMITS.keys()):
#         data.qpos[i] = np.clip(joint_angles[i], *JOINT_LIMITS_RAD[key])
#     mujoco.mj_forward(model, data)
#
#     gripper_inner_pos = data.site_xpos[gripper_inner_site_id]
#     gripper_upper_pos = data.site_xpos[gripper_upper_site_id]
#
#     error_inner = np.linalg.norm(gripper_inner_pos - target_pos)
#     error_upper = np.linalg.norm(gripper_upper_pos - (target_pos + np.array([0, 0, 10])))
#
#     if phase == 0:
#         # Добавляем сдвиг по осям X и Y
#         error_xy = np.linalg.norm(gripper_inner_pos[:2] - target_pos[:2])  # только X и Y
#         return abs(gripper_inner_pos[2] - target_pos[2]) + error_xy  # минимизация по всем осям
#     elif phase == 1:
#         return error_inner
#     else:
#         return error_upper + GRIPPER_PARALLEL_WEIGHT * abs(gripper_upper_pos[2] - gripper_inner_pos[2])
#
# def evaluate_grasp(max_attempts=30):
#     best_qpos_phase0 = None
#     best_qpos_phase1 = None
#     best_error = float('inf')
#
#     for attempt in range(max_attempts):
#         initial_guess = np.random.uniform(
#             [v[0] for v in JOINT_LIMITS_RAD.values()],
#             [v[1] for v in JOINT_LIMITS_RAD.values()]
#         )
#
#         adjusted_target_pos = TARGET_POS.copy()
#         adjusted_target_pos[2] += CUBE_SIZE[2] * 0.7
#
#         # Фаза 0: Поиск позы для начальной позиции
#         result_phase0 = minimize(objective, initial_guess, args=(adjusted_target_pos, 0),
#                                  bounds=list(JOINT_LIMITS_RAD.values()), method='Powell')
#
#         if result_phase0.success:
#             best_qpos_phase0 = result_phase0.x
#
#             # Фаза 1: Поиск позы для фазы 1, начиная с позиции фазы 0
#             result_phase1 = minimize(objective, best_qpos_phase0, args=(adjusted_target_pos, 1),
#                                      bounds=list(JOINT_LIMITS_RAD.values()), method='Powell')
#
#             if result_phase1.success:
#                 error_phase1 = objective(result_phase1.x, adjusted_target_pos, 1)
#                 if error_phase1 < best_error:
#                     best_qpos_phase1 = result_phase1.x
#                     best_error = error_phase1
#
#     if best_qpos_phase1 is None:
#         return None
#
#     best_qpos_phase2 = best_qpos_phase1.copy()
#     best_qpos_phase2[-1] = np.radians(-15)
#
#     return best_qpos_phase0, best_qpos_phase1, best_qpos_phase2
#
# def execute_trajectory(qpos, viewer, steps=150):
#     current_qpos = np.array(data.qpos[:])
#
#     # Сглаживание траектории с использованием кубического сплайна
#     times = np.linspace(0, 1, 2)  # только две точки: начальная и конечная
#     spline = CubicSpline(times, np.array([current_qpos, qpos]), bc_type='natural')
#     trajectory = spline(np.linspace(0, 1, steps))
#
#     for step in trajectory:
#         data.qpos[:] = step
#         mujoco.mj_forward(model, data)
#         mujoco.mj_step(model, data)
#         time.sleep(SLEEP_TIME)
#         viewer.sync()
#
#
# optimal_qpos_phase0, optimal_qpos_phase1, optimal_qpos_phase2 = evaluate_grasp()
# if optimal_qpos_phase0 is None:
#     raise RuntimeError("Не удалось найти оптимальные позиции!")
#
# initial_qpos = np.array([0, 0, 0, 0, 0, 0])
# data.qpos[:] = initial_qpos
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     viewer.cam.azimuth = 90
#     viewer.cam.elevation = -20
#     viewer.cam.distance = 1750.0
#     viewer.cam.lookat[:] = TARGET_POS
#     viewer.cam.trackbodyid = -1
#     viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
#     execute_trajectory(optimal_qpos_phase0, viewer, steps=150)
#     execute_trajectory(optimal_qpos_phase1, viewer, steps=200)
#     time.sleep(0.5)
#     execute_trajectory(optimal_qpos_phase2, viewer, steps=100)
#     while viewer.is_running():
#         viewer.sync()
#
# if os.path.exists(NEW_MODEL_PATH):
#     os.remove(NEW_MODEL_PATH)


# import numpy as np
# import mujoco
# import mujoco.viewer
# import time
# import os
# import pandas as pd
# from scipy.optimize import minimize
# from scipy.interpolate import CubicSpline
#
# # Пути к файлам
# MODEL_PATH = "C:\\Users\\2005s\\Desktop\\FRIDAY_MINI_SIM\\FR_MINI_SIM.xml"
# NEW_MODEL_PATH = "C:\\Users\\2005s\\Desktop\\FRIDAY_MINI_SIM\\modified_robot.xml"
# CSV_PATH = "C:\\Users\\2005s\\PycharmProjects\\STEP_URDF_CONVERTOR\\zones.csv"
#
# # === Настройки ===
# CUBE_SIZE = (10, 20, 120)
# selected_zone_name = 'diag_xyz_-100_100_0'
# ALIGNMENT_THRESHOLD = 10.0
# GRIPPER_PARALLEL_WEIGHT = 100
#
# # Загружаем зоны из CSV
# zones_df = pd.read_csv(CSV_PATH)
# if selected_zone_name not in zones_df['name'].values:
#     raise ValueError(f"Зона '{selected_zone_name}' не найдена в CSV!")
# selected_zone = zones_df[zones_df['name'] == selected_zone_name].iloc[0]
# TARGET_POS = np.array([selected_zone['x'], selected_zone['y'], selected_zone['z']])
#
# # Загружаем XML модель
# with open(MODEL_PATH, "r", encoding="utf-8") as f:
#     xml_content = f.read()
#
# # Добавляем новый объект (кубик)
# new_geom = f"""
#     <geom name="target_cube" type="box" size="{CUBE_SIZE[0]} {CUBE_SIZE[1]} {CUBE_SIZE[2]}"
#           pos="{TARGET_POS[0]} {TARGET_POS[1]} {TARGET_POS[2]}" rgba="0 1 0 1" contype="1" conaffinity="1"/>
# """
#
# # Вставляем в worldbody
# index = xml_content.rfind("</worldbody>")
# if index != -1:
#     xml_content = xml_content[:index] + new_geom + xml_content[index:]
#
# # Сохраняем обновленный XML-файл
# with open(NEW_MODEL_PATH, "w", encoding="utf-8") as f:
#     f.write(xml_content)
#
# # Загружаем обновленную модель MuJoCo
# model = mujoco.MjModel.from_xml_path(NEW_MODEL_PATH)
# data = mujoco.MjData(model)
#
# model.opt.gravity = np.array([0, 0, -9.81])
# model.opt.solver = mujoco.mjtSolver.mjSOL_CG  # Используем более стабильный решатель
# model.opt.noslip_iterations = 100  # Увеличиваем количество итераций решателя
# model.opt.timestep = 0.002
# data.qvel[:] = 2
#
# # Определяем индексы частей хвата
# gripper_inner_site_id = model.site("gripper_inner_site").id
# gripper_upper_site_id = model.site("gripper_upper_site").id
#
# # Ограничения суставов
# JOINT_LIMITS = {f'joint_{i}': (-90, 90) for i in range(5)}
# JOINT_LIMITS['joint_5'] = (-15, 90)
# JOINT_LIMITS_RAD = {k: (np.radians(v[0]), np.radians(v[1])) for k, v in JOINT_LIMITS.items()}
#
# TOLERANCE = 10.0
# MAX_ANGLE_STEP = np.radians(0.5)
# SLEEP_TIME = 0.002
#
#
# def objective(joint_angles, target_pos, phase=1):
#     for i, key in enumerate(JOINT_LIMITS.keys()):
#         data.qpos[i] = np.clip(joint_angles[i], *JOINT_LIMITS_RAD[key])
#     mujoco.mj_forward(model, data)
#
#     gripper_inner_pos = data.site_xpos[gripper_inner_site_id]
#     gripper_upper_pos = data.site_xpos[gripper_upper_site_id]
#
#     error_inner = np.linalg.norm(gripper_inner_pos - target_pos)
#     error_upper = np.linalg.norm(gripper_upper_pos - (target_pos + np.array([0, 0, 10])))
#
#     if phase == 0:
#         # Добавляем сдвиг по осям X и Y
#         error_xy = np.linalg.norm(gripper_inner_pos[:2] - target_pos[:2])  # только X и Y
#         return abs(gripper_inner_pos[2] - target_pos[2]) + error_xy  # минимизация по всем осям
#     elif phase == 1:
#         return error_inner
#     else:
#         return error_upper + GRIPPER_PARALLEL_WEIGHT * abs(gripper_upper_pos[2] - gripper_inner_pos[2])
#
#
# def evaluate_grasp(max_attempts=30):
#     best_qpos_phase0 = None
#     best_qpos_phase1 = None
#     best_error = float('inf')
#
#     for attempt in range(max_attempts):
#         initial_guess = np.random.uniform(
#             [v[0] for v in JOINT_LIMITS_RAD.values()],
#             [v[1] for v in JOINT_LIMITS_RAD.values()]
#         )
#
#         adjusted_target_pos = TARGET_POS.copy()
#         adjusted_target_pos[2] += CUBE_SIZE[2] * 0.7
#
#         # Фаза 0: Поиск позы для начальной позиции
#         result_phase0 = minimize(objective, initial_guess, args=(adjusted_target_pos, 0),
#                                  bounds=list(JOINT_LIMITS_RAD.values()), method='Powell')
#
#         if result_phase0.success:
#             best_qpos_phase0 = result_phase0.x
#
#             # Фаза 1: Поиск позы для фазы 1, начиная с позиции фазы 0
#             result_phase1 = minimize(objective, best_qpos_phase0, args=(adjusted_target_pos, 1),
#                                      bounds=list(JOINT_LIMITS_RAD.values()), method='Powell')
#
#             if result_phase1.success:
#                 error_phase1 = objective(result_phase1.x, adjusted_target_pos, 1)
#                 if error_phase1 < best_error:
#                     best_qpos_phase1 = result_phase1.x
#                     best_error = error_phase1
#
#     if best_qpos_phase1 is None:
#         return None
#
#     best_qpos_phase2 = best_qpos_phase1.copy()
#     best_qpos_phase2[-1] = np.radians(-15)
#
#     return best_qpos_phase0, best_qpos_phase1, best_qpos_phase2
#
#
# def execute_trajectory(qpos, viewer, steps=150):
#     current_qpos = np.array(data.qpos[:])
#
#     # Сглаживание траектории с использованием кубического сплайна
#     times = np.linspace(0, 1, 2)  # только две точки: начальная и конечная
#     spline = CubicSpline(times, np.array([current_qpos, qpos]), bc_type='natural')
#     trajectory = spline(np.linspace(0, 1, steps))
#
#     for step in trajectory:
#         data.qpos[:] = step
#         mujoco.mj_forward(model, data)
#         mujoco.mj_step(model, data)
#         time.sleep(SLEEP_TIME)
#         viewer.sync()
#
#
# def grasp_command():
#     optimal_qpos_phase0, optimal_qpos_phase1, optimal_qpos_phase2 = evaluate_grasp()
#     if optimal_qpos_phase0 is None:
#         raise RuntimeError("Не удалось найти оптимальные позиции!")
#
#     initial_qpos = np.array([0, 0, 0, 0, 0, 0])
#     data.qpos[:] = initial_qpos
#     with mujoco.viewer.launch_passive(model, data) as viewer:
#         viewer.cam.azimuth = 90
#         viewer.cam.elevation = -20
#         viewer.cam.distance = 1750.0
#         viewer.cam.lookat[:] = TARGET_POS
#         viewer.cam.trackbodyid = -1
#         viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
#         execute_trajectory(optimal_qpos_phase0, viewer, steps=150)
#         execute_trajectory(optimal_qpos_phase1, viewer, steps=200)
#         time.sleep(0.5)
#         execute_trajectory(optimal_qpos_phase2, viewer, steps=100)
#         while viewer.is_running():
#             viewer.sync()
#
#
# if __name__ == "__main__":
#     while True:
#         command = input("Введите команду (например, 'grasp' для захвата): ").strip().lower()
#
#         if command == 'grasp':
#             grasp_command()
#         elif command == 'exit':
#             print("Выход из программы...")
#             break
#         else:
#             print("Неизвестная команда, попробуйте снова.")



# import numpy as np
# import mujoco
# import mujoco.viewer
# import time
# import os
# import pandas as pd
# from scipy.optimize import minimize
# from scipy.interpolate import CubicSpline
#
# # Пути к файлам
# MODEL_PATH = "C:\\Users\\2005s\\Desktop\\FRIDAY_MINI_SIM\\FR_MINI_SIM.xml"
# NEW_MODEL_PATH = "C:\\Users\\2005s\\Desktop\\FRIDAY_MINI_SIM\\modified_robot.xml"
# CSV_PATH = "C:\\Users\\2005s\\PycharmProjects\\STEP_URDF_CONVERTOR\\zones.csv"
#
# # === Настройки ===
# CUBE_SIZE = (10, 20, 120)
# selected_zone_name = 'diag_xyz_-100_100_0'
# ALIGNMENT_THRESHOLD = 10.0
# GRIPPER_PARALLEL_WEIGHT = 100
#
# # Загружаем зоны из CSV
# zones_df = pd.read_csv(CSV_PATH)
# if selected_zone_name not in zones_df['name'].values:
#     raise ValueError(f"Зона '{selected_zone_name}' не найдена в CSV!")
# selected_zone = zones_df[zones_df['name'] == selected_zone_name].iloc[0]
# TARGET_POS = np.array([selected_zone['x'], selected_zone['y'], selected_zone['z']])
#
# # Загружаем XML модель
# with open(MODEL_PATH, "r", encoding="utf-8") as f:
#     xml_content = f.read()
#
# # Добавляем новый объект (кубик)
# new_geom = f"""
#     <geom name="target_cube" type="box" size="{CUBE_SIZE[0]} {CUBE_SIZE[1]} {CUBE_SIZE[2]}"
#           pos="{TARGET_POS[0]} {TARGET_POS[1]} {TARGET_POS[2]}" rgba="0 1 0 1" contype="1" conaffinity="1"/>
# """
#
# # Вставляем в worldbody
# index = xml_content.rfind("</worldbody>")
# if index != -1:
#     xml_content = xml_content[:index] + new_geom + xml_content[index:]
#
# # Сохраняем обновленный XML-файл
# with open(NEW_MODEL_PATH, "w", encoding="utf-8") as f:
#     f.write(xml_content)
#
# # Загружаем обновленную модель MuJoCo
# model = mujoco.MjModel.from_xml_path(NEW_MODEL_PATH)
# data = mujoco.MjData(model)
#
# model.opt.gravity = np.array([0, 0, -9.81])
# model.opt.solver = mujoco.mjtSolver.mjSOL_CG  # Используем более стабильный решатель
# model.opt.noslip_iterations = 100  # Увеличиваем количество итераций решателя
# model.opt.timestep = 0.002
# data.qvel[:] = 2
#
# # Определяем индексы частей хвата
# gripper_inner_site_id = model.site("gripper_inner_site").id
# gripper_upper_site_id = model.site("gripper_upper_site").id
#
# # Ограничения суставов
# JOINT_LIMITS = {f'joint_{i}': (-90, 90) for i in range(5)}
# JOINT_LIMITS['joint_5'] = (-15, 90)
# JOINT_LIMITS_RAD = {k: (np.radians(v[0]), np.radians(v[1])) for k, v in JOINT_LIMITS.items()}
#
# TOLERANCE = 10.0
# MAX_ANGLE_STEP = np.radians(0.5)
# SLEEP_TIME = 0.002
#
#
# def objective(joint_angles, target_pos, phase=1):
#     for i, key in enumerate(JOINT_LIMITS.keys()):
#         data.qpos[i] = np.clip(joint_angles[i], *JOINT_LIMITS_RAD[key])
#     mujoco.mj_forward(model, data)
#
#     gripper_inner_pos = data.site_xpos[gripper_inner_site_id]
#     gripper_upper_pos = data.site_xpos[gripper_upper_site_id]
#
#     error_inner = np.linalg.norm(gripper_inner_pos - target_pos)
#     error_upper = np.linalg.norm(gripper_upper_pos - (target_pos + np.array([0, 0, 10])))
#
#     if phase == 0:
#         # Добавляем сдвиг по осям X и Y
#         error_xy = np.linalg.norm(gripper_inner_pos[:2] - target_pos[:2])  # только X и Y
#         return abs(gripper_inner_pos[2] - target_pos[2]) + error_xy  # минимизация по всем осям
#     elif phase == 1:
#         return error_inner
#     else:
#         return error_upper + GRIPPER_PARALLEL_WEIGHT * abs(gripper_upper_pos[2] - gripper_inner_pos[2])
#
#
# def evaluate_grasp(max_attempts=30):
#     best_qpos_phase0 = None
#     best_qpos_phase1 = None
#     best_error = float('inf')
#
#     for attempt in range(max_attempts):
#         initial_guess = np.random.uniform(
#             [v[0] for v in JOINT_LIMITS_RAD.values()],
#             [v[1] for v in JOINT_LIMITS_RAD.values()]
#         )
#
#         adjusted_target_pos = TARGET_POS.copy()
#         adjusted_target_pos[2] += CUBE_SIZE[2] * 0.7
#
#         # Фаза 0: Поиск позы для начальной позиции
#         result_phase0 = minimize(objective, initial_guess, args=(adjusted_target_pos, 0),
#                                  bounds=list(JOINT_LIMITS_RAD.values()), method='Powell')
#
#         if result_phase0.success:
#             best_qpos_phase0 = result_phase0.x
#
#             # Фаза 1: Поиск позы для фазы 1, начиная с позиции фазы 0
#             result_phase1 = minimize(objective, best_qpos_phase0, args=(adjusted_target_pos, 1),
#                                      bounds=list(JOINT_LIMITS_RAD.values()), method='Powell')
#
#             if result_phase1.success:
#                 error_phase1 = objective(result_phase1.x, adjusted_target_pos, 1)
#                 if error_phase1 < best_error:
#                     best_qpos_phase1 = result_phase1.x
#                     best_error = error_phase1
#
#     if best_qpos_phase1 is None:
#         return None
#
#     best_qpos_phase2 = best_qpos_phase1.copy()
#     best_qpos_phase2[-1] = np.radians(-15)
#
#     return best_qpos_phase0, best_qpos_phase1, best_qpos_phase2
#
#
# def execute_trajectory(qpos, viewer, steps=150):
#     current_qpos = np.array(data.qpos[:])
#
#     # Сглаживание траектории с использованием кубического сплайна
#     times = np.linspace(0, 1, 2)  # только две точки: начальная и конечная
#     spline = CubicSpline(times, np.array([current_qpos, qpos]), bc_type='natural')
#     trajectory = spline(np.linspace(0, 1, steps))
#
#     for step in trajectory:
#         data.qpos[:] = step
#         mujoco.mj_forward(model, data)
#         mujoco.mj_step(model, data)
#         time.sleep(SLEEP_TIME)
#         viewer.sync()
#
#
# def grasp_command():
#     optimal_qpos_phase0, optimal_qpos_phase1, optimal_qpos_phase2 = evaluate_grasp()
#     if optimal_qpos_phase0 is None:
#         raise RuntimeError("Не удалось найти оптимальные позиции!")
#
#     initial_qpos = np.array([0, 0, 0, 0, 0, 0])
#     data.qpos[:] = initial_qpos
#     with mujoco.viewer.launch_passive(model, data) as viewer:
#         viewer.cam.azimuth = 90
#         viewer.cam.elevation = -20
#         viewer.cam.distance = 1750.0
#         viewer.cam.lookat[:] = TARGET_POS
#         viewer.cam.trackbodyid = -1
#         viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
#         execute_trajectory(optimal_qpos_phase0, viewer, steps=150)
#         execute_trajectory(optimal_qpos_phase1, viewer, steps=200)
#         time.sleep(0.5)
#         execute_trajectory(optimal_qpos_phase2, viewer, steps=100)
#         while viewer.is_running():
#             viewer.sync()
#
# def objective_take(joint_angles, gripper_inner_target_z, gripper_upper_target_z):
#     """
#     Функция для оптимизации, которая минимизирует смещение по осям X и Y, при этом поднимая по оси Z.
#     """
#     # Применяем новые углы суставов
#     for i, key in enumerate(JOINT_LIMITS.keys()):
#         data.qpos[i] = np.clip(joint_angles[i], *JOINT_LIMITS_RAD[key])
#     mujoco.mj_forward(model, data)
#
#     # Получаем текущие позиции gripper_inner и gripper_upper
#     gripper_inner_pos = data.site_xpos[gripper_inner_site_id]
#     gripper_upper_pos = data.site_xpos[gripper_upper_site_id]
#
#     # Ошибка по осям X и Y (чем меньше, тем лучше)
#     error_xy = np.linalg.norm(gripper_inner_pos[:2] - gripper_upper_pos[:2])
#
#     # Мы поднимаем зоны только по оси Z, учитывая их целевые значения Z
#     error_z = abs(gripper_inner_pos[2] - gripper_inner_target_z) + abs(gripper_upper_pos[2] - gripper_upper_target_z)
#
#     return error_z + error_xy
#
# def take():
#     """
#     Метод, который поднимет gripper_inner и gripper_upper вдоль оси Z, минимизируя смещение по X и Y.
#     Выполняется после хвата (grasp_command).
#     """
#     # Выполняем команду захвата
#     grasp_command()
#
#     # Начальные положения gripper_inner и gripper_upper
#     gripper_inner_pos = data.site_xpos[gripper_inner_site_id]
#     gripper_upper_pos = data.site_xpos[gripper_upper_site_id]
#
#     # Сохраняем их текущие координаты X и Y
#     gripper_inner_x, gripper_inner_y, gripper_inner_z = gripper_inner_pos
#     gripper_upper_x, gripper_upper_y, gripper_upper_z = gripper_upper_pos
#
#     # Выводим текущие позиции для отладки
#     print(f"gripper_inner_pos: {gripper_inner_pos}")
#     print(f"gripper_upper_pos: {gripper_upper_pos}")
#
#     # Целевые значения Z для подъема (например, на 50 единиц выше)
#     gripper_inner_target_z = gripper_inner_z + 50
#     gripper_upper_target_z = gripper_upper_z + 50
#
#     # Инициализация начальных углов суставов
#     initial_joint_angles = np.array([data.qpos[i] for i in range(len(data.qpos))])
#
#     # Минимизация функции ошибки (подъем по оси Z)
#     result = minimize(objective_take, initial_joint_angles, args=(gripper_inner_target_z, gripper_upper_target_z),
#                       bounds=list(JOINT_LIMITS_RAD.values()), method='Powell')
#
#     if result.success:
#         # Применяем найденные углы суставов
#         best_qpos = result.x
#         data.qpos[:] = best_qpos
#         mujoco.mj_forward(model, data)
#
#         # Плавно обновляем позиции в симуляторе
#         with mujoco.viewer.launch_passive(model, data) as viewer:
#             viewer.cam.azimuth = 90
#             viewer.cam.elevation = -20
#             viewer.cam.distance = 1750.0
#             viewer.cam.lookat[:] = TARGET_POS
#             viewer.cam.trackbodyid = -1
#             viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
#
#             time.sleep(0.5)
#             while viewer.is_running():
#                 viewer.sync()
#     else:
#         raise RuntimeError("Не удалось найти оптимальные позиции для подъема!")
#
# if __name__ == "__main__":
#     while True:
#         command = input("Введите команду (например, 'take' для захвата и подъема): ").strip().lower()
#
#         if command == 'take':
#             take()
#         elif command == "grab":
#             grasp_command()
#         elif command == 'exit':
#             print("Выход из программы...")
#             break
#         else:
#             print("Неизвестная команда, попробуйте снова.")

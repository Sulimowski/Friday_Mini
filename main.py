#-------------------------Without full phys
import numpy as np
import mujoco
import mujoco.viewer
import time
import os
from scipy.optimize import minimize

# === Пути ===
MODEL_PATH = "C:\\Users\\2005s\\PycharmProjects\\Mujoco-good\\good_mukoco\\FRIDAY_MINI.xml"
NEW_MODEL_PATH = "C:\\Users\\2005s\\PycharmProjects\\Mujoco-good\\good_mukoco\\modified_robot.xml"

# === Настройки сцены ===
CUBE_SIZE = (0.02, 0.02, 0.06)
TARGET_POS = np.array([0.4, 0.2, 0.06])
SLEEP_TIME = 0.01
GRIPPER_PARALLEL_WEIGHT = 100
TOLERANCE = 0.01
MAX_STEPS = 5000
EARLY_STOP_THRESHOLD = 0.0025

# === Добавляем куб в XML с полной физикой ===
def inject_cube_to_model():
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        xml = f.read()

    new_geom = f'''
        <body name="target_cube" pos="{TARGET_POS[0]} {TARGET_POS[1]} {TARGET_POS[2]}" >
            <geom type="box" size="{CUBE_SIZE[0]} {CUBE_SIZE[1]} {CUBE_SIZE[2]}" rgba="0 1 0 1"
                  mass="0.1" contype="1" conaffinity="1" condim="3" friction="1 0.005 0.0001"/>
            <joint type="free" />
        </body>
    '''

    xml = xml.replace("</worldbody>", new_geom + "\n</worldbody>")

    with open(NEW_MODEL_PATH, "w", encoding="utf-8") as f:
        f.write(xml)

inject_cube_to_model()

# === Инициализация модели ===
model = mujoco.MjModel.from_xml_path(NEW_MODEL_PATH)
data = mujoco.MjData(model)

# === Идентификаторы ===
gripper_inner_site_id = model.site("gripper_inner_site").id
gripper_upper_site_id = model.site("gripper_upper_site").id
actuator_ids = [model.actuator(f"servo_{i}").id for i in range(6)]

# === Ограничения суставов ===
JOINT_LIMITS_DEG = {
    'joint_0': (-154, 154),
    'joint_1': (-90, 90),
    'joint_2': (-90, 90),
    'joint_3': (-90, 90),
    'joint_4': (-90, 90),
    'joint_5': (-6, 90),
}
JOINT_LIMITS_RAD = {k: (np.radians(v[0]), np.radians(v[1])) for k, v in JOINT_LIMITS_DEG.items()}

# === Управление ===
def set_ctrl(qpos_target, kp=8.0):
    current_qpos = data.qpos[:6]
    error = qpos_target - current_qpos
    for i, actuator_id in enumerate(actuator_ids):
        torque = kp * error[i]
        data.ctrl[actuator_id] = np.clip(torque, -3.0, 3.0)

def get_gripper_positions():
    inner = data.site_xpos[gripper_inner_site_id].copy()
    upper = data.site_xpos[gripper_upper_site_id].copy()
    return inner, upper

# === Оценка одной попытки ===
def test_attempt(model, ctrl_values, target_pos):
    tmp_data = mujoco.MjData(model)
    mujoco.mj_resetData(model, tmp_data)
    mujoco.mj_forward(model, tmp_data)

    traj = []
    for _ in range(100):
        qpos_snapshot = tmp_data.qpos[:6].copy()
        traj.append(qpos_snapshot)
        for i, actuator_id in enumerate(actuator_ids):
            error = ctrl_values[i] - tmp_data.qpos[i]
            tmp_data.ctrl[actuator_id] = np.clip(8.0 * error, -3.0, 3.0)
        mujoco.mj_step(model, tmp_data)

    inner = tmp_data.site_xpos[gripper_inner_site_id]
    err = np.linalg.norm(inner - target_pos)

    return err, traj

# === Оптимизация ===
def evaluate_grasp(max_attempts=10):
    adjusted_target = TARGET_POS.copy()
    adjusted_target[2] += CUBE_SIZE[2] * 0.7

    best_error = float('inf')
    best_traj = None

    for attempt in range(max_attempts):
        print(f"\n⚙️ Попытка {attempt + 1}")
        base = np.radians([0, 0, 0, 0, 0, 0])
        noise = np.random.uniform(-0.2, 0.2, size=6)
        initial = np.clip(base + noise,
                          [v[0] for v in JOINT_LIMITS_RAD.values()],
                          [v[1] for v in JOINT_LIMITS_RAD.values()])

        res = minimize(
            lambda x: test_attempt(model, x, adjusted_target)[0],
            initial,
            bounds=list(JOINT_LIMITS_RAD.values()),
            method='Powell',
            options={'xtol': 1e-3, 'ftol': 1e-4, 'maxiter': 500}
        )

        if res.success:
            err, traj = test_attempt(model, res.x, adjusted_target)
            print(f"👉 Ошибка: {err:.4f}")
            if err < best_error:
                best_error = err
                best_traj = traj

            if best_error <= EARLY_STOP_THRESHOLD:
                print("🚀 Ранняя остановка: достигнута достаточная точность!")
                break

    print(f"\n✅ Лучшая ошибка: {best_error:.4f}")
    return best_error, best_traj

# === Воспроизведение траектории ===
def replay_trajectory(trajectory, viewer=None):
    for qpos in trajectory:
        data.qpos[:6] = qpos
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
        time.sleep(SLEEP_TIME)

# === Основная команда ===
def grasp_command():
    result = evaluate_grasp()
    if result is None:
        raise RuntimeError("Не удалось найти оптимальные позиции!")

    _, best_traj = result

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.75
        viewer.cam.lookat[:] = TARGET_POS
        viewer.cam.trackbodyid = -1
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        replay_trajectory(best_traj, viewer)

        while viewer.is_running():
            viewer.sync()

# === Основной цикл ===
if __name__ == "__main__":
    print("🟢 Симуляция активна. Команды: grab / exit")
    while True:
        cmd = input("👉 Команда: ").strip().lower()
        if cmd == 'grab':
            grasp_command()
        elif cmd == 'exit':
            break
        else:
            print("⚠️ Неизвестная команда")

# === Очистка ===
if os.path.exists(NEW_MODEL_PATH):
    os.remove(NEW_MODEL_PATH)
    print(f"🗑️ Временный файл {NEW_MODEL_PATH} удалён")


#-----------------------------With full phys
# import numpy as np
# import mujoco
# import mujoco.viewer
# import time
# import os
# from scipy.optimize import minimize
#
# # === Пути ===
# MODEL_PATH = "C:\\Users\\2005s\\Desktop\\good_mukoco\\FRIDAY_MINI.xml"
# NEW_MODEL_PATH = "C:\\Users\\2005s\\Desktop\\good_mukoco\\modified_robot.xml"
#
# # === Настройки сцены ===
# CUBE_SIZE = (0.02, 0.02, 0.06)
# TARGET_POS = np.array([0.4, 0.2, 0.06])
# SLEEP_TIME = 0.01
# GRIPPER_PARALLEL_WEIGHT = 100
# TOLERANCE = 0.01
# MAX_STEPS = 5000
# EARLY_STOP_THRESHOLD = 0.0025
#
# # === Добавляем куб в XML с полной физикой ===
# def inject_cube_to_model():
#     with open(MODEL_PATH, "r", encoding="utf-8") as f:
#         xml = f.read()
#
#     new_geom = f'''
#         <body name="target_cube" pos="{TARGET_POS[0]} {TARGET_POS[1]} {TARGET_POS[2]}" >
#             <geom type="box" size="{CUBE_SIZE[0]} {CUBE_SIZE[1]} {CUBE_SIZE[2]}" rgba="0 1 0 1"
#                   mass="200" contype="1" conaffinity="1" solimp="0.95 0.95 0.01" solref="0.02 1" condim="3" friction="2 0.2 0.1"/>
#             <joint type="free" />
#         </body>
#     '''
#
#     xml = xml.replace("</worldbody>", new_geom + "\n</worldbody>")
#
#     with open(NEW_MODEL_PATH, "w", encoding="utf-8") as f:
#         f.write(xml)
#
# inject_cube_to_model()
#
# # === Инициализация модели ===
# model = mujoco.MjModel.from_xml_path(NEW_MODEL_PATH)
# data = mujoco.MjData(model)
#
# # === Идентификаторы ===
# gripper_inner_site_id = model.site("gripper_inner_site").id
# gripper_upper_site_id = model.site("gripper_upper_site").id
# actuator_ids = [model.actuator(f"servo_{i}").id for i in range(6)]
#
# # === Ограничения суставов ===
# JOINT_LIMITS_DEG = {
#     'joint_0': (-154, 154),
#     'joint_1': (-90, 90),
#     'joint_2': (-90, 90),
#     'joint_3': (-90, 90),
#     'joint_4': (-90, 90),
#     'joint_5': (-6, 90),
# }
# JOINT_LIMITS_RAD = {k: (np.radians(v[0]), np.radians(v[1])) for k, v in JOINT_LIMITS_DEG.items()}
#
# # === Управление ===
# def set_ctrl(qpos_target, kp=8.0):
#     current_qpos = data.qpos[:6]
#     error = qpos_target - current_qpos
#     for i, actuator_id in enumerate(actuator_ids):
#         torque = kp * error[i]
#         data.ctrl[actuator_id] = np.clip(torque, -3.0, 3.0)
#
# def get_gripper_positions():
#     inner = data.site_xpos[gripper_inner_site_id].copy()
#     upper = data.site_xpos[gripper_upper_site_id].copy()
#     return inner, upper
#
# # === Оценка одной попытки ===
# def test_attempt(model, ctrl_values, target_pos):
#     tmp_data = mujoco.MjData(model)
#     mujoco.mj_resetData(model, tmp_data)
#     mujoco.mj_forward(model, tmp_data)
#
#     traj = []
#     for _ in range(100):
#         qpos_snapshot = tmp_data.qpos[:6].copy()
#         traj.append(qpos_snapshot)
#         for i, actuator_id in enumerate(actuator_ids):
#             error = ctrl_values[i] - tmp_data.qpos[i]
#             tmp_data.ctrl[actuator_id] = np.clip(8.0 * error, -3.0, 3.0)
#         mujoco.mj_step(model, tmp_data)
#
#     inner = tmp_data.site_xpos[gripper_inner_site_id]
#     err = np.linalg.norm(inner - target_pos)
#     # Добавим финальную позу N раз, чтобы "заморозить" руку
#     final_qpos = traj[-1]
#     hold_steps = int(1.0 / SLEEP_TIME)  # 1 секунда
#
#     for _ in range(hold_steps):
#         traj.append(final_qpos.copy())
#
#     return err, traj
#
#
# # === Оптимизация ===
# def evaluate_grasp(max_attempts=10):
#     adjusted_target = TARGET_POS.copy()
#     adjusted_target[2] += CUBE_SIZE[2] * 0.7
#
#     best_error = float('inf')
#     best_traj = None
#
#     for attempt in range(max_attempts):
#         print(f"\n⚙️ Попытка {attempt + 1}")
#         base = np.radians([0, 0, 0, 0, 0, 0])
#         noise = np.random.uniform(-0.2, 0.2, size=6)
#         initial = np.clip(base + noise,
#                           [v[0] for v in JOINT_LIMITS_RAD.values()],
#                           [v[1] for v in JOINT_LIMITS_RAD.values()])
#
#         res = minimize(
#             lambda x: test_attempt(model, x, adjusted_target)[0],
#             initial,
#             bounds=list(JOINT_LIMITS_RAD.values()),
#             method='Powell',
#             options={'xtol': 1e-3, 'ftol': 1e-4, 'maxiter': 500}
#         )
#
#         if res.success:
#             err, traj = test_attempt(model, res.x, adjusted_target)
#             print(f"👉 Ошибка: {err:.4f}")
#             if err < best_error:
#                 best_error = err
#                 best_traj = traj
#
#             if best_error <= EARLY_STOP_THRESHOLD:
#                 print("🚀 Ранняя остановка: достигнута достаточная точность!")
#                 break
#
#     print(f"\n✅ Лучшая ошибка: {best_error:.4f}")
#     print("🔁 Начальная поза:", best_traj[0])
#     print("🏁 Финальная поза:", best_traj[-1])
#     return best_error, best_traj
#
# # === Воспроизведение траектории ===
# def replay_trajectory(trajectory, viewer=None):
#     for qpos in trajectory:
#         data.qpos[:6] = qpos
#         mujoco.mj_step(model, data)
#         if viewer:
#             viewer.sync()
#         time.sleep(SLEEP_TIME)
#
# # === Основная команда ===
# def grasp_command():
#     result = evaluate_grasp()
#     if result is None:
#         raise RuntimeError("Не удалось найти оптимальные позиции!")
#
#     _, best_traj = result
#
#     with mujoco.viewer.launch_passive(model, data) as viewer:
#         viewer.cam.azimuth = 90
#         viewer.cam.elevation = -20
#         viewer.cam.distance = 1.75
#         viewer.cam.lookat[:] = TARGET_POS
#         viewer.cam.trackbodyid = -1
#         viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
#         mujoco.mj_resetData(model, data)
#         mujoco.mj_forward(model, data)
#         replay_trajectory(best_traj, viewer)
#
#         while viewer.is_running():
#             viewer.sync()
#
# # === Основной цикл ===
# if __name__ == "__main__":
#     print("🟢 Симуляция активна. Команды: grab / exit")
#     while True:
#         cmd = input("👉 Команда: ").strip().lower()
#         if cmd == 'grab':
#             grasp_command()
#         elif cmd == 'exit':
#             break
#         else:
#             print("⚠️ Неизвестная команда")
#
# # === Очистка ===
# if os.path.exists(NEW_MODEL_PATH):
#     os.remove(NEW_MODEL_PATH)
#     print(f"🗑️ Временный файл {NEW_MODEL_PATH} удалён")
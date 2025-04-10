import pybullet as p
import pybullet_data
import time

# === Подключение к PyBullet ===
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# === Загрузка плоскости ===
p.loadURDF("plane.urdf")

# === Загрузка URDF-робота с включёнными самостолкновениями ===
robot_id = p.loadURDF(
    r"C:\Users\2005s\PycharmProjects\FRIDAY_RL\pretty_mukoco\FRIDAY_MINI.urdf",
    useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION
)

# === Создание куба для взаимодействия ===
target_pos = [0.3, 0.2, 0.05]  # центр куба
cube_half_extents = [0.02, 0.02, 0.05]  # половинки размеров

cube_visual = p.createVisualShape(
    shapeType=p.GEOM_BOX,
    halfExtents=cube_half_extents,
    rgbaColor=[0, 1, 0, 1]
)

cube_collision = p.createCollisionShape(
    shapeType=p.GEOM_BOX,
    halfExtents=cube_half_extents
)

cube_id = p.createMultiBody(
    baseMass=1.0,
    baseCollisionShapeIndex=cube_collision,
    baseVisualShapeIndex=cube_visual,
    basePosition=target_pos
)

# === Добавление сайтов только спереди и сзади куба, прикреплённых к нему ===
side_extent = [0.0005, 0.02, 0.04]
side_offsets = [
    [ cube_half_extents[0] + side_extent[0], 0, 0],   # перед (+X)
    [-cube_half_extents[0] - side_extent[0], 0, 0],   # зад (-X)
]
side_colors = [
    [1, 0, 1, 1],   # Фиолетовый — перед
    [1, 1, 0, 1],   # Жёлтый — зад
]
side_labels = ["FRONT", "BACK"]
site_ids = []

for offset, color in zip(side_offsets, side_colors):
    # Позиция сайта в мировых координатах
    site_world_pos = [
        target_pos[0] + offset[0],
        target_pos[1] + offset[1],
        target_pos[2] + offset[2]  # поднятие вверх
    ]

    # Визуал и коллизия
    side_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=side_extent, rgbaColor=color)
    side_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=side_extent)

    # Создание сайта
    site_id = p.createMultiBody(
        baseMass=0.0001,
        baseCollisionShapeIndex=side_collision,
        baseVisualShapeIndex=side_visual,
        basePosition=site_world_pos
    )

    # Привязка к кубу
    p.createConstraint(
        parentBodyUniqueId=cube_id,
        parentLinkIndex=-1,
        childBodyUniqueId=site_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[
            offset[0], offset[1], offset[2] + 0.01
        ],
        childFramePosition=[0, 0, 0]
    )

    site_ids.append(site_id)

# === Создание слайдеров для управления суставами ===
num_joints = p.getNumJoints(robot_id)
slider_ids = []
controlled_joints = []

for joint_index in range(num_joints):
    joint_info = p.getJointInfo(robot_id, joint_index)
    joint_name = joint_info[1].decode('utf-8')
    joint_type = joint_info[2]

    if joint_type == p.JOINT_REVOLUTE:
        lower_limit = joint_info[8]
        upper_limit = joint_info[9]
        slider_id = p.addUserDebugParameter(joint_name, lower_limit, upper_limit, 0.0)
        slider_ids.append(slider_id)
        controlled_joints.append(joint_index)

print(f"Управляемые суставы: {controlled_joints}")

# === Основной цикл симуляции ===
while True:
    # === Найдём индексы линков
    inner_site_index = None
    upper_site_index = None

    for i in range(p.getNumJoints(robot_id)):
        link_name = p.getJointInfo(robot_id, i)[12].decode("utf-8")
        if "gripper_inner_site" in link_name:
            inner_site_index = i
        elif "gripper_upper_site" in link_name:
            upper_site_index = i

    # === Создаём визуальные маркеры (сферки)
    marker_inner_id = None
    marker_upper_id = None

    if inner_site_index is not None:
        sphere_visual_inner = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.01,
            rgbaColor=[1, 1, 0, 1]  # жёлтый
        )
        marker_inner_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=sphere_visual_inner,
            basePosition=[0, 0, 0]
        )

    if upper_site_index is not None:
        sphere_visual_upper = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.01,
            rgbaColor=[1, 0, 1, 1]  # фиолетовый
        )
        marker_upper_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=sphere_visual_upper,
            basePosition=[0, 0, 0]
        )

    # === Внутренний цикл симуляции ===
    while True:
        # Управление суставами
        for slider_id, joint_index in zip(slider_ids, controlled_joints):
            target_pos = p.readUserDebugParameter(slider_id)
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=40
            )

        # Обновление позиций сайтов руки
        if inner_site_index is not None:
            site_pos_inner = p.getLinkState(robot_id, inner_site_index, computeForwardKinematics=True)[4]
            p.resetBasePositionAndOrientation(marker_inner_id, site_pos_inner, [0, 0, 0, 1])
            p.addUserDebugText("INNER", site_pos_inner, textColorRGB=[1, 1, 0], textSize=1.2, lifeTime=0.1)

        if upper_site_index is not None:
            site_pos_upper = p.getLinkState(robot_id, upper_site_index, computeForwardKinematics=True)[4]
            p.resetBasePositionAndOrientation(marker_upper_id, site_pos_upper, [0, 0, 0, 1])
            p.addUserDebugText("UPPER", site_pos_upper, textColorRGB=[1, 0, 1], textSize=1.2, lifeTime=0.1)

        # === Подписи для сайтов куба ===
        for site_id, label in zip(site_ids, side_labels):
            pos, _ = p.getBasePositionAndOrientation(site_id)
            p.addUserDebugText(label, pos, textColorRGB=[1, 1, 1], textSize=1.2, lifeTime=0.1)

        # Шаг симуляции
        p.stepSimulation()
        time.sleep(1 / 240)

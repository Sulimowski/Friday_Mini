import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from grasp_env_dynamic import GraspEnv  # твоя новая версия среды

# === Настройки
num_epochs = 60
steps_per_epoch = 10000
reward_window = deque(maxlen=40)
reward_threshold = 50.0
stability_threshold = 0.5
patience = 30
min_delta = 1.0

# === Папки
os.makedirs("models/fine_tuned_dynamic", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# === Среда
venv = DummyVecEnv([lambda: Monitor(GraspEnv(gui=False), filename=None)])
vec_path = Path("models/vecnormalize.pkl")
if vec_path.exists():
    print("🔄 Загружаем нормализацию...")
    venv = VecNormalize.load(str(vec_path), venv)

venv.training = True
venv.norm_reward = False

# === Загрузка старой модели
model_path = Path("models/best_model.zip")
if model_path.exists():
    print("📦 Загрузка существующей модели...")
    model = PPO.load(str(model_path), env=venv)
else:
    raise FileNotFoundError("Модель models/best_model.zip не найдена. Сначала обучи базовую.")

# === Обучение
best_reward = -np.inf
no_improve_epochs = 0
rewards_log = []
total_timesteps = 0

plt.ion()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

for epoch in range(num_epochs):
    print(f"\n🌟 Epoch {epoch + 1}/{num_epochs}")
    model.learn(total_timesteps=steps_per_epoch, reset_num_timesteps=False)
    total_timesteps += steps_per_epoch

    mean_reward, std_reward = evaluate_policy(model, venv, n_eval_episodes=5, deterministic=True)
    print(f"📊 Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    rewards_log.append((epoch + 1, mean_reward))

    model.save(f"models/fine_tuned_dynamic/epoch_{epoch + 1}")
    reward_window.append(mean_reward)

    if mean_reward > best_reward + min_delta:
        best_reward = mean_reward
        no_improve_epochs = 0
        model.save("models/fine_tuned_dynamic/best_model")
        venv.save("models/fine_tuned_dynamic/vecnormalize.pkl")
        print("💾 Новый лучший результат сохранён.")
    else:
        no_improve_epochs += 1
        print(f"⏳ No improvement. Patience: {no_improve_epochs}/{patience}")

        if len(reward_window) == reward_window.maxlen:
            std_recent = np.std(reward_window)
            max_recent = max(reward_window)
            if std_recent < stability_threshold and max_recent < reward_threshold:
                print(f"🛑 Early stopping: Rewards stabilized ({std_recent:.2f}) and goal not reached.")
                break

        if no_improve_epochs >= patience:
            print(f"🛑 Early stopping на {epoch + 1} эпохе.")
            break

    # === График
    epochs = [e[0] for e in rewards_log]
    rewards = [e[1] for e in rewards_log]
    steps = [i * steps_per_epoch for i in range(1, len(rewards_log) + 1)]

    ax1.clear()
    ax2.clear()

    ax1.plot(epochs, rewards, label="Mean Reward", color='tab:blue', marker='o')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Reward", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    ax2.plot(epochs, steps, label="Total Timesteps", color='tab:orange', linestyle="--")
    ax2.set_ylabel("Total Timesteps", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    ax1.legend(loc="upper left")
    fig.tight_layout()
    plt.pause(0.1)

# === Финал
plt.ioff()
plt.savefig("logs/training_curve_dynamic.png")
plt.show()

# === Сохраняем лог
df = pd.DataFrame(rewards_log, columns=["epoch", "mean_reward"])
df.to_csv("logs/reward_log_dynamic.csv", index=False)

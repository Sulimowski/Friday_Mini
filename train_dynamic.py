import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from grasp_env_dynamic import GraspEnv  # обязательно обновлённый с правильной obs

# === Папки
os.makedirs("models/fine_tuned", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# === Среда без загрузки старой нормализации
# После создания новой VecNormalize
venv = DummyVecEnv([lambda: Monitor(GraspEnv(gui=False))])
venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=5.0)

from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
old_stats = VecNormalize.load("models/vecnormalize.pkl", venv)
venv.obs_rms = old_stats.obs_rms

# Загрузим модель
model = PPO.load("models/best_model.zip", env=venv)

# Сохраняем старые веса до смены политики
old_policy_weights = model.policy.state_dict()

# === Переопределяем политику под новую размерность наблюдений
model.policy = model.policy_class(
    model.observation_space,
    model.action_space,
    model.lr_schedule,
    use_sde=model.use_sde
).to(model.device)

# Переносим старые веса в новую политику
model.policy.load_state_dict(old_policy_weights, strict=False)

# Устанавливаем обновлённую среду
model.set_env(venv)


# === Настройки обучения
num_epochs = 60
steps_per_epoch = 10000
reward_window = deque(maxlen=40)
reward_threshold = 50.0
stability_threshold = 0.5
rewards_log = []
patience = 30
min_delta = 1.0
no_improve_epochs = 0
best_reward = -float("inf")

# === Онлайн график
plt.ion()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# === Цикл обучения
for epoch in range(num_epochs):
    print(f"\n\U0001F31F Epoch {epoch + 1}/{num_epochs}")
    model.learn(total_timesteps=steps_per_epoch, reset_num_timesteps=False)

    # Оценка
    mean_reward, std_reward = evaluate_policy(model, venv, n_eval_episodes=5, deterministic=True)
    print(f"\U0001F4CA Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    rewards_log.append((epoch + 1, mean_reward))

    # Сохраняем модель
    model.save(f"models/fine_tuned/fine_tuned_epoch_{epoch + 1}")

    reward_window.append(mean_reward)

    if mean_reward > best_reward + min_delta:
        best_reward = mean_reward
        no_improve_epochs = 0
        model.save("models/fine_tuned/fine_tuned_best")
        venv.save("models/fine_tuned/vecnormalize.pkl")
        print("\U0001F4BE New best fine-tuned model saved.")
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
            print(f"🛑 Early stopping triggered at epoch {epoch + 1}.")
            break

    # === Живой график
    epochs = [e[0] for e in rewards_log]
    rewards = [e[1] for e in rewards_log]
    steps = [i * steps_per_epoch for i in range(1, len(rewards_log) + 1)]

    ax1.clear()
    ax2.clear()

    line1 = ax1.plot(epochs, rewards, label="Mean Reward", color='tab:blue', marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Reward", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    line2 = ax2.plot(epochs, steps, label="Total Timesteps", color='tab:orange', linestyle="--")
    ax2.set_ylabel("Total Timesteps", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    fig.tight_layout()
    plt.pause(0.1)

plt.ioff()
plt.savefig("logs/training_curve_finetuned.png")
plt.show()

# Сохраняем лог
df = pd.DataFrame(rewards_log, columns=["epoch", "mean_reward"])
df.to_csv("logs/reward_log_finetuned.csv", index=False)

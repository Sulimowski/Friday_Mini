# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# from stable_baselines3 import PPO
# from grasp_env import GraspEnv
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.evaluation import evaluate_policy
# from collections import deque
# import numpy as np
#
# reward_window = deque(maxlen=40)
# reward_threshold = 50.0  # цель не достигнута
# stability_threshold = 0.5
#
# # === Папки
# os.makedirs("models", exist_ok=True)
# os.makedirs("logs", exist_ok=True)
#
# # === Среда
# venv = DummyVecEnv([lambda: Monitor(GraspEnv(gui=False), filename=None)])
# venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=5.0)
#
# # === PPO конфиг
# model = PPO(
#     "MlpPolicy",
#     venv,
#     verbose=1,
#     device="cuda",
#     n_steps=1024,
#     batch_size=64,
#     gae_lambda=0.95,
#     gamma=0.99,
#     n_epochs=10,
#     ent_coef=0.01,
#     learning_rate=3e-4,
#     clip_range=0.2
# )
#
# # === Настройки обучения
# num_epochs = 30
# steps_per_epoch = 7000
# total_timesteps_so_far = 0
# rewards_log = []
#
# # === Early Stopping параметры
# patience = 20
# min_delta = 1.0
# no_improve_epochs = 0
# best_reward = -float("inf")
#
# # === Онлайн график
# plt.ion()
# fig, ax = plt.subplots()
#
# # === Цикл обучения
# for epoch in range(num_epochs):
#     print(f"\n🌟 Epoch {epoch + 1}/{num_epochs}")
#     model.learn(total_timesteps=steps_per_epoch, reset_num_timesteps=False)
#     total_timesteps_so_far += steps_per_epoch
#
#     # === Оценка нормализованная и реальная
#     mean_norm_reward, std_norm = evaluate_policy(model, venv, n_eval_episodes=5, deterministic=True)
#
#     raw_env = GraspEnv(gui=False)
#     mean_raw_reward, std_raw = evaluate_policy(model, raw_env, n_eval_episodes=5, deterministic=True)
#
#     print(f"📊 Norm reward: {mean_norm_reward:.2f} ± {std_norm:.2f}")
#     print(f"📉 Raw reward : {mean_raw_reward:.2f} ± {std_raw:.2f}")
#
#     rewards_log.append((epoch + 1, mean_norm_reward, mean_raw_reward))
#
#     # === Сохраняем модель этой эпохи
#     model.save(f"models/ppo_grasp_epoch_{epoch + 1}")
#
#     # === Early stopping
#     reward_window.append(mean_raw_reward)
#
#     if mean_raw_reward > best_reward + min_delta:
#         best_reward = mean_raw_reward
#         no_improve_epochs = 0
#         model.save("models/best_model")
#         venv.save("models/vecnormalize.pkl")
#         print("💾 New best model saved.")
#     else:
#         no_improve_epochs += 1
#         print(f"⏳ No improvement. Patience: {no_improve_epochs}/{patience}")
#
#         if len(reward_window) == reward_window.maxlen:
#             std_recent = np.std(reward_window)
#             max_recent = max(reward_window)
#             if std_recent < stability_threshold and max_recent < reward_threshold:
#                 print(f"🛑 Early stopping: Rewards stabilized ({std_recent:.2f}) and goal not reached.")
#                 break
#
#         if no_improve_epochs >= patience:
#             print(f"🛑 Early stopping triggered at epoch {epoch + 1}.")
#             break
#
#     # === Живой график
#     epochs = [e[0] for e in rewards_log]
#     norm_rewards = [e[1] for e in rewards_log]
#     raw_rewards = [e[2] for e in rewards_log]
#     steps = [i * steps_per_epoch for i in range(1, len(rewards_log) + 1)]
#
#     ax.clear()
#     ax2 = ax.twinx()
#
#     line1, = ax.plot(epochs, raw_rewards, label="Raw Mean Reward", marker="o", linewidth=2)
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Raw Mean Reward")
#     ax.set_title("📈 PPO Learning Progress")
#     ax.grid(True)
#
#     line2, = ax2.plot(epochs, steps, label="Total Timesteps", color="orange", linestyle="--")
#     ax2.set_ylabel("Total Timesteps")
#
#     lines = [line1, line2]
#     labels = [line.get_label() for line in lines]
#     ax.legend(lines, labels, loc="lower right")
#
#     fig.tight_layout()
#     plt.pause(0.1)
#
# # === Финал
# plt.ioff()
# plt.savefig("logs/training_curve.png")
# plt.show()
#
# # === Сохраняем лог
# df = pd.DataFrame(rewards_log, columns=["epoch", "norm_reward", "raw_reward"])
# df.to_csv("logs/reward_log.csv", index=False)

import os
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from grasp_env import GraspEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from collections import deque
import numpy as np

reward_window = deque(maxlen=40)
reward_threshold = 50.0  # цель не достигнута
stability_threshold = 0.5

# === Папки
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# === Среда
venv = DummyVecEnv([lambda: Monitor(GraspEnv(gui=False), filename=None)])
venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=5.0)

from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    venv,
    verbose=1,
    device="cpu",  # или "cpu", если без GPU
    n_steps=4096,           # Больше шагов — больше опыта
    batch_size=256,         # Сбалансировано
    n_epochs=15,            # Лучше обучается на редких хороших эпизодах
    learning_rate=2e-4,     # Плавное обучение
    clip_range=0.15,        # Строже ограничивает изменения политики
    gamma=0.99,             # Стандартная дисконтировка
    gae_lambda=0.95,        # Хороший GAE
    ent_coef=0.01,          # Немного энтропии для исследования
)


# === Настройки обучения
num_epochs = 50
steps_per_epoch = 10000
total_timesteps_so_far = 0
rewards_log = []

# === Early Stopping параметры
patience = 30
min_delta = 1.0
no_improve_epochs = 0
best_reward = -float("inf")

# === Онлайн график
plt.ion()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # создаём ось один раз, а не на каждой итерации

# === Цикл обучения
for epoch in range(num_epochs):
    print(f"\n🌟 Epoch {epoch + 1}/{num_epochs}")
    model.learn(total_timesteps=steps_per_epoch, reset_num_timesteps=False)
    total_timesteps_so_far += steps_per_epoch

    # === Оценка
    mean_reward, std_reward = evaluate_policy(model, venv, n_eval_episodes=5, deterministic=True)
    print(f"📊 Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    rewards_log.append((epoch + 1, mean_reward))

    # === Сохраняем модель этой эпохи
    model.save(f"models/ppo_grasp_epoch_{epoch + 1}")

    # === Early stopping
    reward_window.append(mean_reward)

    # === Улучшенная проверка остановки
    if mean_reward > best_reward + min_delta:
        best_reward = mean_reward
        no_improve_epochs = 0
        model.save("models/best_model")
        venv.save("models/vecnormalize.pkl")
        print("💾 New best model saved.")
    else:
        no_improve_epochs += 1

        print(f"⏳ No improvement. Patience: {no_improve_epochs}/{patience}")

        # если за 10 эпох все +/- одинаковые и не достигнута цель
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

    # График средней награды
    line1 = ax1.plot(epochs, rewards, label="Mean Reward", color='tab:blue', marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Reward", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    # График total timesteps
    line2 = ax2.plot(epochs, steps, label="Total Timesteps", color='tab:orange', linestyle="--")
    ax2.set_ylabel("Total Timesteps", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Легенда
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    fig.tight_layout()
    plt.pause(0.1)

# === Финал
plt.ioff()
plt.savefig("logs/training_curve.png")
plt.show()

# === Сохраняем лог
df = pd.DataFrame(rewards_log, columns=["epoch", "mean_reward"])
df.to_csv("logs/reward_log.csv", index=False)
#
# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# from stable_baselines3 import PPO
# from grasp_env import GraspEnv
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.evaluation import evaluate_policy
# from collections import deque
# import numpy as np
# from pathlib import Path
#
# # === Настройки
# reward_window = deque(maxlen=40)
# reward_threshold = 50.0
# stability_threshold = 0.5
#
# # === Папки
# os.makedirs("models/fine_tuned", exist_ok=True)
# os.makedirs("logs", exist_ok=True)
#
# # === Среда
# venv = DummyVecEnv([lambda: Monitor(GraspEnv(gui=False), filename=None)])
# venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=5.0)
#
# # === Загрузка нормализации среды
# vec_path = Path("models/vecnormalize.pkl")
# if vec_path.exists():
#     print("🔄 Загружаем нормализацию среды...")
#     venv = VecNormalize.load(str(vec_path), venv)
#
# # === Загрузка модели
# best_model_path = Path("models/best_model.zip")
# if best_model_path.exists():
#     print("📂 Загружаем существующую модель для дообучения...")
#     model = PPO.load(str(best_model_path), env=venv)
# else:
#     print("📦 Начинаем обучение с нуля...")
#     model = PPO(
#         "MlpPolicy",
#         venv,
#         verbose=1,
#         device="cpu",  # или "cpu"
#         n_steps=4096,
#         batch_size=256,
#         n_epochs=15,
#         learning_rate=2e-4,
#         clip_range=0.15,
#         gamma=0.99,
#         gae_lambda=0.95,
#         ent_coef=0.01,
#     )
#
# # === Настройки обучения
# num_epochs = 60
# steps_per_epoch = 10000
# total_timesteps_so_far = 0
# rewards_log = []
#
# # === Early stopping
# patience = 35
# min_delta = 1.0
# no_improve_epochs = 0
# best_reward = -float("inf")
#
# # === Онлайн график
# plt.ion()
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
#
# # === Цикл обучения
# for epoch in range(num_epochs):
#     print(f"\n🌟 Epoch {epoch + 1}/{num_epochs}")
#     model.learn(total_timesteps=steps_per_epoch, reset_num_timesteps=False)
#     total_timesteps_so_far += steps_per_epoch
#
#     # === Оценка
#     mean_reward, std_reward = evaluate_policy(model, venv, n_eval_episodes=5, deterministic=True)
#     print(f"📊 Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
#     rewards_log.append((epoch + 1, mean_reward))
#
#     # === Сохраняем модель этой эпохи
#     model.save(f"models/fine_tuned/fine_tuned_epoch_{epoch + 1}")
#
#     # === Early stopping логика
#     reward_window.append(mean_reward)
#
#     if mean_reward > best_reward + min_delta:
#         best_reward = mean_reward
#         no_improve_epochs = 0
#         model.save("models/fine_tuned/fine_tuned_best")
#         venv.save("models/fine_tuned/vecnormalize.pkl")
#         print("💾 New best fine-tuned model saved.")
#     else:
#         no_improve_epochs += 1
#         print(f"⏳ No improvement. Patience: {no_improve_epochs}/{patience}")
#
#         if len(reward_window) == reward_window.maxlen:
#             std_recent = np.std(reward_window)
#             max_recent = max(reward_window)
#             if std_recent < stability_threshold and max_recent < reward_threshold:
#                 print(f"🛑 Early stopping: Rewards stabilized ({std_recent:.2f}) and goal not reached.")
#                 break
#
#         if no_improve_epochs >= patience:
#             print(f"🛑 Early stopping triggered at epoch {epoch + 1}.")
#             break
#
#     # === Живой график
#     epochs = [e[0] for e in rewards_log]
#     rewards = [e[1] for e in rewards_log]
#     steps = [i * steps_per_epoch for i in range(1, len(rewards_log) + 1)]
#
#     ax1.clear()
#     ax2.clear()
#
#     line1 = ax1.plot(epochs, rewards, label="Mean Reward", color='tab:blue', marker="o")
#     ax1.set_xlabel("Epoch")
#     ax1.set_ylabel("Mean Reward", color='tab:blue')
#     ax1.tick_params(axis='y', labelcolor='tab:blue')
#     ax1.grid(True)
#
#     line2 = ax2.plot(epochs, steps, label="Total Timesteps", color='tab:orange', linestyle="--")
#     ax2.set_ylabel("Total Timesteps", color='tab:orange')
#     ax2.tick_params(axis='y', labelcolor='tab:orange')
#
#     lines = line1 + line2
#     labels = [l.get_label() for l in lines]
#     ax1.legend(lines, labels, loc="upper left")
#
#     fig.tight_layout()
#     plt.pause(0.1)
#
# # === Финал
# plt.ioff()
# plt.savefig("logs/training_curve_finetuned.png")
# plt.show()
#
# # === Сохраняем лог
# df = pd.DataFrame(rewards_log, columns=["epoch", "mean_reward"])
# df.to_csv("logs/reward_log_finetuned.csv", index=False)

# === Эволюционное обучение PPO с мутациями и параллелизмом ===
# === Эволюционное обучение PPO с мутациями и графиком ===
# import os
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from joblib import Parallel, delayed
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.evaluation import evaluate_policy
# from grasp_env import GraspEnv
#
# NUM_AGENTS = 3
# EPOCHS_PER_AGENT = 30
# STEPS_PER_EPOCH = 5000
# GENERATIONS = 5
#
# SAVE_DIR = "evo_models"
# os.makedirs(SAVE_DIR, exist_ok=True)
#
# best_overall_reward = -np.inf
# best_overall_model_path = None
# reward_history = []
#
#
# def mutate_model(model, mutation_std=0.01):
#     with torch.no_grad():
#         for param in model.policy.parameters():
#             noise = torch.randn_like(param) * mutation_std
#             param.add_(noise)
#     if hasattr(model, 'learning_rate'):
#         model.learning_rate = lambda: np.clip(model.learning_rate() * np.random.uniform(0.9, 1.1), 1e-5, 1e-2)
#     if hasattr(model, 'gamma'):
#         model.gamma = float(np.clip(model.gamma * np.random.uniform(0.95, 1.05), 0.8, 0.999))
#     return model
#
#
# def train_and_evaluate_agent(agent_idx, generation, base_model_path=None):
#     print(f"[GEN {generation}] Training agent #{agent_idx + 1}")
#     venv = DummyVecEnv([lambda: Monitor(GraspEnv(gui=False))])
#     venv = VecNormalize(venv, norm_obs=True, norm_reward=False)
#
#     if base_model_path and os.path.exists(base_model_path):
#         model = PPO.load(base_model_path, env=venv)
#         model = mutate_model(model)
#     else:
#         model = PPO("MlpPolicy", venv, verbose=0, learning_rate=3e-4)
#
#     model.learn(total_timesteps=EPOCHS_PER_AGENT * STEPS_PER_EPOCH)
#
#     env_eval = GraspEnv(gui=False)
#     mean_reward, _ = evaluate_policy(model, env_eval, n_eval_episodes=5)
#     print(f"✅ Agent #{agent_idx + 1} reward: {mean_reward:.2f}")
#
#     model_path = f"{SAVE_DIR}/gen{generation}_agent{agent_idx + 1}.zip"
#     model.save(model_path)
#
#     return mean_reward, model_path
#
#
# # === График
# plt.ion()
# fig, ax = plt.subplots(figsize=(10, 6))
#
# for gen in range(GENERATIONS):
#     print(f"\n=== GENERATION {gen + 1}/{GENERATIONS} ===")
#     base_path = best_overall_model_path if best_overall_model_path else None
#
#     results = Parallel(n_jobs=NUM_AGENTS)(
#         delayed(train_and_evaluate_agent)(i, gen + 1, base_path) for i in range(NUM_AGENTS)
#     )
#
#     rewards, model_paths = zip(*results)
#     reward_history.append(rewards)
#     best_idx = int(np.argmax(rewards))
#
#     print(f"\n🏆 Best agent of generation {gen + 1}: Agent #{best_idx + 1} with reward {rewards[best_idx]:.2f}")
#
#     if rewards[best_idx] > best_overall_reward:
#         best_overall_reward = rewards[best_idx]
#         best_overall_model_path = model_paths[best_idx]
#         print("🌟 New best overall model saved!")
#
#     # === Обновляем график
#     ax.clear()
#     for ag in range(NUM_AGENTS):
#         ag_rewards = [gen_rewards[ag] for gen_rewards in reward_history]
#         ax.plot(range(1, len(ag_rewards) + 1), ag_rewards, marker='o', label=f"Agent {ag+1}")
#
#     ax.set_xlabel("Generation")
#     ax.set_ylabel("Reward")
#     ax.set_title("📈 Reward per Agent per Generation")
#     ax.grid(True)
#     ax.legend(loc="upper left")
#     fig.tight_layout()
#     plt.pause(0.1)
#
# plt.ioff()
# plt.savefig("evo_models/evolution_curve.png")
# plt.show()
#
# print("\n✅ Эволюционное обучение завершено. Лучший результат:", best_overall_reward)

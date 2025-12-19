# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import pickle
import os
from env import RacingEnv
from model import GaussianPolicy, QNetwork

# === Гиперпараметры ===
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 1e-4
LR_CRITIC = 3e-4
BUFFER_SIZE = 100_000
BATCH_SIZE = 256
MAX_STEPS = 15000
SAVE_EVERY = 50
EPOCH = 250
BEST_PATH = "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (torch.FloatTensor(state).to(device),
                torch.FloatTensor(action).to(device),
                torch.FloatTensor(reward).unsqueeze(1).to(device),
                torch.FloatTensor(next_state).to(device),
                torch.BoolTensor(done).unsqueeze(1).to(device))
    
    def __len__(self):
        return len(self.buffer)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def train():
    env = RacingEnv(seed=42, max_steps=MAX_STEPS)
    state_dim = env.STATE_DIM
    action_dim = 1

    policy = GaussianPolicy().to(device)
    q_net = QNetwork().to(device)
    target_q_net = QNetwork().to(device)
    target_q_net.load_state_dict(q_net.state_dict())

    policy_optimizer = optim.Adam(policy.parameters(), lr=LR_ACTOR)
    q_optimizer = optim.Adam(q_net.parameters(), lr=LR_CRITIC)

    buffer = ReplayBuffer(BUFFER_SIZE)

    total_steps = 0
    best_score = -1

    for episode in range(EPOCH):
        state = env.reset()
        episode_reward = 0
        episode_score = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = policy.sample(state_tensor)
            action = action.cpu().numpy()[0, 0]
            next_state, reward, done, info = env.step(action)

            buffer.push(state, [action], reward, next_state, done)
            state = next_state
            episode_reward += reward
            episode_score = info.get("score", 0)
            total_steps += 1

            # Обучение
            if len(buffer) > BATCH_SIZE:
                states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

                # Обучаем Q-сети
                with torch.no_grad():
                    next_actions, next_log_probs, _, _ = policy.sample(next_states)
                    q1_next, q2_next = target_q_net(next_states, next_actions)
                    q_next = torch.min(q1_next, q2_next) - next_log_probs
                    q_target = rewards + (1 - dones.float()) * GAMMA * q_next

                q1, q2 = q_net(states, actions)
                q_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

                q_optimizer.zero_grad()
                q_loss.backward()
                q_optimizer.step()

                # Обучаем политику
                new_actions, log_probs, _, _ = policy.sample(states)
                q1_new, q2_new = q_net(states, new_actions)
                q_new = torch.min(q1_new, q2_new)
                policy_loss = (log_probs - q_new).mean()

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                # Обновляем target-сети
                soft_update(target_q_net, q_net, TAU)

        # Логирование
        if episode % 10 == 0:
            print(f"Эпизод {episode:4d} | Награда: {episode_reward:7.1f} | Счет: {episode_score:3d} | Шаги: {env.steps:4d}")

        # Сохранение лучшей модели
        if episode_score > best_score:
            best_score = episode_score
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'best_score': best_score,
                'episode': episode
            }, BEST_PATH)
            print(f"  -> Новая лучшая модель! Счет: {best_score}")

        # Сохранение каждые N эпизодов
        if episode % SAVE_EVERY == 0:
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'q_net_state_dict': q_net.state_dict(),
                'episode': episode,
                'best_score': best_score
            }, f"checkpoint_{episode}.pth")

if __name__ == "__main__":
    train()
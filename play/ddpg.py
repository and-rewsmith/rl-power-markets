import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
from collections import deque
import os
from datetime import datetime

# Hyperparameters
ENV = 'Pendulum-v1'
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 64
EPISODES = 200
EVAL_EPISODES = 5
RENDER_EPISODES = 2
SAVE_DIR = 'saved_models'
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Actor Network


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

# Critic Network


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

# Replay Buffer


class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)

    def add(self, s, a, r, s_, d):
        self.buffer.append((s, a, r, s_, d))

    def sample(self):
        batch = random.sample(self.buffer, BATCH_SIZE)
        s, a, r, s_, d = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(s).to(device),
            torch.FloatTensor(a).to(device),
            torch.FloatTensor(r).unsqueeze(1).to(device),
            torch.FloatTensor(s_).to(device),
            torch.FloatTensor(d).unsqueeze(1).to(device)
        )

# Soft update


def soft_update(target, source):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(TAU * s_param.data + (1 - TAU) * t_param.data)


# Environment setup
env = gym.make(ENV)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

actor = Actor(state_dim, action_dim, max_action).to(device)
actor_target = Actor(state_dim, action_dim, max_action).to(device)
actor_target.load_state_dict(actor.state_dict())

critic = Critic(state_dim, action_dim).to(device)
critic_target = Critic(state_dim, action_dim).to(device)
critic_target.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

replay_buffer = ReplayBuffer()

# Training loop
best_reward = float('-inf')
episode_rewards = []  # Track all rewards

for episode in range(EPISODES):
    state, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = actor(state_tensor).cpu().data.numpy()[0]
        action += np.random.normal(0, 0.1, size=action_dim)  # exploration noise
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        replay_buffer.add(state, action, reward, next_state, float(done))
        state = next_state
        episode_reward += reward

        if len(replay_buffer.buffer) > BATCH_SIZE:
            s, a, r, s_, d = replay_buffer.sample()

            with torch.no_grad():
                target_q = critic_target(s_, actor_target(s_))
                target_value = r + (1 - d) * GAMMA * target_q

            q_value = critic(s, a)
            critic_loss = nn.MSELoss()(q_value, target_value)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            actor_loss = -critic(s, actor(s)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            soft_update(critic_target, critic)
            soft_update(actor_target, actor)

    episode_rewards.append(episode_reward)
    if episode_reward > best_reward:
        best_reward = episode_reward
        print(f"Episode {episode}, New Best Reward: {best_reward:.2f}")
    else:
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Best: {best_reward:.2f}")

print(f"\nTraining finished! Final Best Reward: {best_reward:.2f}")
print(f"Average of last 10 episodes: {np.mean(episode_rewards[-10:]):.2f}")

env.close()

# Add evaluation and rendering of best model
input("\nRendering final model performance...")
env = gym.make(ENV, render_mode="human")

# Save final weights
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(SAVE_DIR, f'ddpg_final_{timestamp}.pth')
torch.save({
    'actor_state_dict': actor.state_dict(),
    'critic_state_dict': critic.state_dict(),
}, save_path)

for episode in range(RENDER_EPISODES):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = actor(state_tensor).cpu().data.numpy()[0]
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        episode_reward += reward

    print(f"Render Episode {episode}, Reward: {episode_reward:.2f}")

env.close()

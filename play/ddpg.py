import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random

# Hyperparameters
BUFFER_SIZE = 100000
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
LEARNING_RATE = 3e-4
EPISODES = 200
NOISE_SCALE = 0.2
MIN_NOISE = 0.05
WARMUP_STEPS = 5000  # Add warmup period before training
HIDDEN_SIZE = 256  # Larger networks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(reward).unsqueeze(1).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(done).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)


def main():
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = Actor(state_dim, action_dim, max_action).to(device)
    actor_target = Actor(state_dim, action_dim, max_action).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic = Critic(state_dim, action_dim).to(device)
    critic_target = Critic(state_dim, action_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    noise_scale = NOISE_SCALE

    total_steps = 0
    best_reward = float('-inf')

    for episode in range(EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            total_steps += 1

            # Only start using policy after warmup
            if total_steps < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = actor(state_tensor).cpu().numpy()[0]
                    noise = np.random.normal(0, max_action * noise_scale)
                    action = np.clip(action + noise, -max_action, max_action)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Store transition
            replay_buffer.add(state, action, reward, next_state, float(done))
            state = next_state

            # Only train after warmup period
            if len(replay_buffer) > BATCH_SIZE and total_steps > WARMUP_STEPS:
                # Sample from buffer
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

                # Compute target Q
                with torch.no_grad():
                    next_actions = actor_target(next_states)
                    target_Q = critic_target(next_states, next_actions)
                    target_Q = rewards + (1 - dones) * GAMMA * target_Q

                # Update critic
                current_Q = critic(states, actions)
                critic_loss = nn.MSELoss()(current_Q, target_Q)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Update actor
                actor_loss = -critic(states, actor(states)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Update target networks
                for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                    target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
                for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                    target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        # Track best reward
        best_reward = max(best_reward, episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Best: {best_reward:.2f}, Steps: {total_steps}")

    env.close()


if __name__ == "__main__":
    main()

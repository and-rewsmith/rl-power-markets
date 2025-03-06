import random
import torch
import wandb
import numpy as np
from collections import deque

from rl_power_markets.benchmarks.markets.simple import SimpleMarket

"""
This file functions as the DQN implementation of the market learning harness.
"""

# Hyperparameters
LR = 0.0001
GAMMA = 0.7
TAU = 0.005
BUFFER_SIZE = 100000
BATCH_SIZE = 64
HIDDEN_SIZE = 256
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Action space discretization
NUM_ACTIONS = 100
ACTION_MULTIPLIERS = torch.linspace(1.0, 3.0, NUM_ACTIONS)


def initialize_wandb() -> None:
    wandb.init(
        project="rl-power-markets",
        config={
            "architecture": "DQN",
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "buffer_size": BUFFER_SIZE,
            "tau": TAU,
            "gamma": GAMMA,
            "epsilon_start": EPSILON_START,
            "epsilon_end": EPSILON_END,
            "epsilon_decay": EPSILON_DECAY,
            "num_actions": NUM_ACTIONS,
        }
    )


class DQN(torch.nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, num_hours: int, num_actions: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_hours * num_actions),
        )
        self.num_hours = num_hours
        self.num_actions = num_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        out = self.net(x)
        # Reshape to (batch_size, num_hours, num_actions)
        return out.view(batch_size, self.num_hours, self.num_actions)


class ReplayBuffer:
    def __init__(self, market: SimpleMarket) -> None:
        self.buffer: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = deque(maxlen=BUFFER_SIZE)
        self.batch_size = market.batch_size
        self.obs_size = market.obs_size
        self.num_hours = market.num_hours

    def add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor) -> None:
        # Store single items from the batch
        for i in range(self.batch_size):
            self.buffer.append((
                state[i],
                action[i],
                reward[i],
                next_state[i]
            ))

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, BATCH_SIZE)
        state = torch.stack([b[0] for b in batch])
        action = torch.stack([b[1] for b in batch])
        reward = torch.stack([b[2] for b in batch])
        next_state = torch.stack([b[3] for b in batch])

        assert state.shape == (BATCH_SIZE, self.obs_size)
        assert action.shape == (BATCH_SIZE, self.num_hours)  # Stores action indices
        assert reward.shape == (BATCH_SIZE, 1)
        assert next_state.shape == (BATCH_SIZE, self.obs_size)
        return state, action, reward, next_state


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    initialize_wandb()

    market = SimpleMarket()
    episodes = market.episodes
    timesteps = market.timesteps

    # Initialize networks
    dqn = DQN(obs_size=market.obs_size, hidden_size=HIDDEN_SIZE,
              num_hours=market.num_hours, num_actions=NUM_ACTIONS)
    dqn_target = DQN(obs_size=market.obs_size, hidden_size=HIDDEN_SIZE,
                     num_hours=market.num_hours, num_actions=NUM_ACTIONS)
    dqn_target.load_state_dict(dqn.state_dict())

    optimizer = torch.optim.Adam(dqn.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(market)
    epsilon = EPSILON_START
    max_reward_so_far = float('-inf')

    for episode in episodes:
        market.reset()
        state = market.obtain_state()
        episode_reward: float = 0

        for timestep in timesteps:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                # Random actions (indices)
                action_indices = torch.randint(0, NUM_ACTIONS, (market.batch_size, market.num_hours))
            else:
                with torch.no_grad():
                    q_values = dqn(state)
                    action_indices = q_values.argmax(dim=2)

            # Convert action indices to actual multiplier values
            action_multipliers = ACTION_MULTIPLIERS[action_indices]
            assert action_multipliers.shape == (market.batch_size, market.num_hours)

            # Step environment
            next_state, reward = market.step(action_multipliers)
            assert next_state.shape == (market.batch_size, market.obs_size)
            assert reward.shape == (market.batch_size, 1)

            # Store transition
            replay_buffer.add(state, action_indices, reward, next_state)
            episode_reward += reward.mean().item()
            state = next_state.detach()

            wandb.log({
                "episode_reward": episode_reward,
                "epsilon": epsilon
            })

            # Train if enough samples
            if len(replay_buffer.buffer) > BATCH_SIZE:
                # Sample from replay buffer
                states, actions, rewards, next_states = replay_buffer.sample()

                # Compute target Q value
                with torch.no_grad():
                    next_q_values = dqn_target(next_states)
                    next_q_max = next_q_values.max(dim=2)[0]
                    target_q = rewards + GAMMA * next_q_max.unsqueeze(1)

                # Compute current Q value
                current_q = dqn(states)
                current_q_actions = torch.gather(
                    current_q,
                    2,
                    actions.unsqueeze(2).expand(-1, -1, 1)
                ).squeeze(2)

                # Compute loss and update
                loss = torch.nn.functional.mse_loss(current_q_actions, target_q.squeeze(1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=1.0)
                optimizer.step()

                # Soft update target network
                soft_update(dqn_target, dqn, TAU)

                # Logging
                wandb.log({
                    "loss": loss.item(),
                    "q_value": current_q_actions.mean().item(),
                    "train_reward": rewards.mean().item(),
                })

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        max_reward_so_far = max(max_reward_so_far, episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Max Reward: {max_reward_so_far:.2f}")

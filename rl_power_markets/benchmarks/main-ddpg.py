import random

import torch
import wandb
import numpy as np
from collections import deque

from rl_power_markets.benchmarks.markets.full_market import FullMarket
from rl_power_markets.model.agent import Critic, Actor
from rl_power_markets.benchmarks.markets.simple import SimpleMarket

"""
This file functions as the harness that is parameterized by market type.
"""


def initialize_wandb() -> None:
    wandb.init(
        project="rl-power-markets",
        config={
            "architecture": "DDPG",
            "lr_actor": LR_ACTOR,
            "lr_critic": LR_CRITIC,
            "batch_size": BATCH_SIZE,
            "buffer_size": BUFFER_SIZE,
            "tau": TAU,
            "gamma": GAMMA,
        }
    )


# Hyperparameters
LR_ACTOR = 0.0001
LR_CRITIC = 0.0001
GAMMA = 0.7
TAU = 0.005
BUFFER_SIZE = 100000
BATCH_SIZE = 8
ACTOR_HIDDEN_SIZE = 256
CRITIC_HIDDEN_SIZE = 256


class ReplayBuffer:
    def __init__(self, market: SimpleMarket) -> None:
        self.buffer: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = deque(maxlen=BUFFER_SIZE)
        self.batch_size = market.batch_size
        self.obs_size = market.obs_size
        self.num_actions = market.num_actions

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
        # Stack individual tensors
        state = torch.stack([b[0] for b in batch])
        action = torch.stack([b[1] for b in batch])
        reward = torch.stack([b[2] for b in batch])
        next_state = torch.stack([b[3] for b in batch])

        assert state.shape == (BATCH_SIZE, self.obs_size)
        assert action.shape == (BATCH_SIZE, self.num_actions)
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

    market = FullMarket()
    episodes = market.episodes
    timesteps = market.timesteps

    # Initialize networks
    actor = Actor(obs_size=market.obs_size, hidden_size=ACTOR_HIDDEN_SIZE, num_actions=market.num_actions)
    actor_target = Actor(obs_size=market.obs_size, hidden_size=ACTOR_HIDDEN_SIZE, num_actions=market.num_actions)
    actor_target.load_state_dict(actor.state_dict())

    critic = Critic(obs_size=market.obs_size + market.num_actions, hidden_size=CRITIC_HIDDEN_SIZE)
    critic_target = Critic(obs_size=market.obs_size + market.num_actions, hidden_size=CRITIC_HIDDEN_SIZE)
    critic_target.load_state_dict(critic.state_dict())

    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)

    replay_buffer = ReplayBuffer(market)
    max_reward_so_far = float('-inf')

    for episode in episodes:
        market.reset()
        state = market.obtain_state()
        episode_reward: float = 0

        for timestep in timesteps:
            # Get action and add exploration noise
            action = actor(state)
            noise = torch.normal(0, 0.3, size=action.shape)
            action = torch.clamp(action + noise, min=1.0)  # Ensure multiplier >= 1.0
            assert action.shape == (market.batch_size, market.num_actions)

            # Step environment
            next_state, reward = market.step(action)
            assert next_state.shape == (market.batch_size, market.obs_size)
            assert reward.shape == (market.batch_size, 1)

            # Store transition
            replay_buffer.add(state, action, reward, next_state)
            episode_reward += reward.mean().item()
            state = next_state.detach()

            wandb.log({
                "episode_reward": episode_reward,
            })

            # Train if enough samples
            if len(replay_buffer.buffer) > BATCH_SIZE:
                # Sample from replay buffer
                states, actions, rewards, next_states = replay_buffer.sample()

                # Compute target Q value
                with torch.no_grad():
                    target_actions = actor_target(next_states)
                    target_q = critic_target(next_states, target_actions).detach()
                    target_value = rewards + GAMMA * target_q
                assert target_value.shape == (BATCH_SIZE, 1)

                # Update critic
                critic_output = critic(states.detach(), actions.detach())
                current_q = critic_output
                assert current_q.shape == (BATCH_SIZE, 1)
                # critic_loss = torch.nn.functional.mse_loss(current_q, target_value.detach())
                td_error = target_value - current_q
                critic_loss = td_error ** 2
                critic_loss = critic_loss.mean()

                optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                optimizer_critic.step()

                # Update actor
                actor_actions = actor(states.detach())
                actor_loss = -critic(states.detach(), actor_actions).mean()

                optimizer_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                optimizer_actor.step()

                # Soft update targets
                soft_update(critic_target, critic, TAU)
                soft_update(actor_target, actor, TAU)

                # Logging
                wandb.log({
                    "critic_loss": critic_loss.item(),
                    "actor_loss": actor_loss.item(),
                    "train_q_value": current_q.mean().item(),
                    "train_reward": rewards.mean().item(),
                    "average_ui_status": market.u_i.mean().item(),
                    "average_gi_status": market.g_i.mean().item(),
                    "td_error": td_error.mean().item(),
                    "critic_output": critic_output.mean().item(),
                })

        max_reward_so_far = max(max_reward_so_far, episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Max Reward: {max_reward_so_far:.2f}")

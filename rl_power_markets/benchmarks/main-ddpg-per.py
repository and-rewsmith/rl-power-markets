import random

import torch
import wandb
import numpy as np
from collections import deque

from rl_power_markets.benchmarks.markets.full_market_linear import FullSimpleMarket
from rl_power_markets.model.agent import Critic, Actor
from rl_power_markets.benchmarks.markets.simple import SimpleMarket

"""
This file functions as the harness that is parameterized by market type.
"""


def initialize_wandb() -> None:
    wandb.init(
        project="rl-power-markets",
        config={
            "architecture": "DDPG-PER",
            "lr_actor": LR_ACTOR,
            "lr_critic": LR_CRITIC,
            "batch_size": BATCH_SIZE,
            "buffer_size": BUFFER_SIZE,
            "tau": TAU,
            "gamma": GAMMA,
            "beta1": BETA1,
            "beta2": BETA2,
            "noise_initial": NOISE_INITIAL,
            "noise_min": NOISE_MIN,
            "noise_decay": NOISE_DECAY,
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
BETA1 = 0.6
BETA2 = 0.4
# Add noise decay parameters
NOISE_INITIAL = 0.2
NOISE_MIN = 0.0005
NOISE_DECAY = 0.94  # Decay factor per episode


class PrioritizedReplayBuffer:
    def __init__(self, market: SimpleMarket) -> None:
        self.buffer: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.priorities = np.zeros(BUFFER_SIZE)
        self.position = 0
        self.batch_size = market.batch_size
        self.obs_size = market.obs_size
        self.num_actions = market.num_actions
        self.epsilon = 1e-6

    def add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor) -> None:
        max_priority = max(self.priorities.max(), 1.0)

        for i in range(self.batch_size):
            if len(self.buffer) < BUFFER_SIZE:
                self.buffer.append((
                    state[i],
                    action[i],
                    reward[i],
                    next_state[i]
                ))
            else:
                self.buffer[self.position] = (
                    state[i],
                    action[i],
                    reward[i],
                    next_state[i]
                )

            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % BUFFER_SIZE

    def get_probabilities(self) -> np.ndarray:
        N = len(self.buffer)
        ranks = np.argsort(-self.priorities[:N])
        ranks = np.argsort(ranks)
        probs = (1 / (ranks + 1)) ** BETA1
        probs = probs / probs.sum()
        return probs

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        N = len(self.buffer)
        probs = self.get_probabilities()

        indices = np.random.choice(N, BATCH_SIZE, p=probs)

        weights = (N * probs[indices]) ** (-BETA2)
        weights = weights / weights.max()
        weights = torch.FloatTensor(weights).reshape(-1, 1)

        batch = [self.buffer[idx] for idx in indices]
        state = torch.stack([b[0] for b in batch])
        action = torch.stack([b[1] for b in batch])
        reward = torch.stack([b[2] for b in batch])
        next_state = torch.stack([b[3] for b in batch])

        return state, action, reward, next_state, weights, indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        self.priorities[indices] = np.abs(td_errors) + self.epsilon


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    initialize_wandb()

    market = FullSimpleMarket(batch_size=BATCH_SIZE)
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

    replay_buffer = PrioritizedReplayBuffer(market)
    max_reward_so_far = float('-inf')

    # Initialize noise scale
    noise_scale = NOISE_INITIAL

    episode_counter = 0
    for episode in episodes:
        market.reset()
        state = market.obtain_state()
        episode_reward: float = 0
        episode_price: float = 0
        episode_counter += 1

        for timestep in timesteps:
            # Get action and add exploration noise with decaying scale
            action = actor(state)
            noise = torch.normal(0, noise_scale, size=action.shape)
            action = torch.clamp(action + noise, min=1.0)  # Ensure multiplier >= 1.0
            assert action.shape == (market.batch_size, market.num_actions)

            # Step environment
            next_state, reward = market.step(action)
            assert next_state.shape == (market.batch_size, market.obs_size)
            assert reward.shape == (market.batch_size, 1)

            # Store transition
            replay_buffer.add(state, action, reward, next_state)
            episode_reward += reward.mean().item()
            episode_price += market.prices.mean().item()
            state = next_state.detach()

            # Added detailed timestep logging
            wandb.log({
                "timestep_prices": market.prices.mean().item(),
                "timestep_bidding multiplier": action.mean().item(),
                "timestep_average_ui_status": market.u_i.mean().item(),
                "timestep_average_gi_status": market.g_i.mean().item(),
                "noise_scale": noise_scale,
            })

            # Train if enough samples
            if len(replay_buffer.buffer) > BATCH_SIZE:
                # Changed: Sample with priorities
                states, actions, rewards, next_states, weights, indices = replay_buffer.sample()

                # Compute target Q value
                with torch.no_grad():
                    target_actions = actor_target(next_states)
                    target_q = critic_target(next_states, target_actions).detach()
                    target_value = rewards + GAMMA * target_q
                assert target_value.shape == (BATCH_SIZE, 1)

                # Update critic (changed to use importance sampling weights and match TD error calculation)
                critic_output = critic(states.detach(), actions.detach())
                current_q = critic_output
                td_errors = (target_value.detach() - current_q).detach().numpy()

                # Calculate TD error and critic loss similar to main-ddpg.py
                td_error = target_value - current_q
                critic_loss = weights * (td_error ** 2)
                critic_loss = critic_loss.mean()

                optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                optimizer_critic.step()

                # Added: Update priorities
                replay_buffer.update_priorities(indices, td_errors.flatten())

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

                # Logging - updated to match main-ddpg.py exactly
                wandb.log({
                    "critic_loss": critic_loss.item(),
                    "actor_loss": actor_loss.item(),
                    "q_value": current_q.mean().item(),
                    "reward": rewards.mean().item(),
                    "td_error": td_error.mean().item(),
                    "critic_output": critic_output.mean().item(),
                })

        # Decay noise after each episode
        noise_scale = max(NOISE_MIN, noise_scale * NOISE_DECAY)

        # Episode-level logging
        wandb.log({
            "episode_reward": episode_reward / len(timesteps),
            "episode_price": episode_price / len(timesteps),
            "episode_counter": episode_counter,
            "noise_scale": noise_scale,
        })
        max_reward_so_far = max(max_reward_so_far, episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Max Reward: {max_reward_so_far:.2f}, Noise: {noise_scale:.4f}")

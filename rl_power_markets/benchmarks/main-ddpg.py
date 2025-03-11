import random

import torch
import wandb
import numpy as np
from collections import deque
import os

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
            "architecture": "DDPG",
            "lr_actor": LR_ACTOR,
            "lr_critic": LR_CRITIC,
            "batch_size": BATCH_SIZE,
            "buffer_size": BUFFER_SIZE,
            "tau": TAU,
            "gamma": GAMMA,
        }
    )


BATCH_SIZE = 8
# LR_ACTOR = 0.00001
# LR_CRITIC = 0.001
NOISE_MAX_SCALE = 0.1       # Maximum noise amplitude


# BATCH_SIZE = 64
LR_ACTOR = 0.0001
LR_CRITIC = 0.01
# NOISE_MAX_SCALE = 0.1       # Maximum noise amplitude

# Hyperparameters
NOISE_MIN_SCALE = 0.0001      # Minimum noise amplitude
GAMMA = 0.7
TAU = 0.005
BUFFER_SIZE = 100000
ACTOR_HIDDEN_SIZE = 256
CRITIC_HIDDEN_SIZE = 256
# Noise parameters
NOISE_PERIOD = 40           # Number of episodes for a complete cycle
NOISE_PHASE_SHIFT = 0       # Phase shift in radians
SAVE_FREQUENCY = 25


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

    market = FullSimpleMarket(BATCH_SIZE)
    episodes = market.episodes
    timesteps = market.timesteps

    # Create a tensor to store data for a batch of episodes
    # Shape: (SAVE_FREQUENCY, num_hours, 5) - Now including strategic producer's bid
    episode_data_batch = torch.zeros((SAVE_FREQUENCY, market.num_hours, 5))

    # Output file name
    output_file = 'episode_data.pt'

    # Delete the file if it exists to start fresh
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Deleted existing data file: {output_file}")

    # Initialize with empty data
    existing_data = torch.zeros((0, market.num_hours, 5))
    episodes_saved = 0
    print("Starting fresh with new data file")

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

    # Initialize noise scale
    current_noise_scale = NOISE_MAX_SCALE

    episode_counter = 0
    batch_counter = 0
    for episode in range(episodes_saved, market.num_episodes):
        # Calculate sinusoidal noise scale
        # sin oscillates between -1 and 1, so we adjust to get values between NOISE_MIN_SCALE and NOISE_MAX_SCALE
        current_noise_scale = NOISE_MIN_SCALE + (NOISE_MAX_SCALE - NOISE_MIN_SCALE) * (
            (np.sin(2 * np.pi * episode_counter / NOISE_PERIOD + NOISE_PHASE_SHIFT) + 1) / 2
        )

        market.reset()
        state = market.obtain_state()
        episode_reward: float = 0
        episode_price: float = 0

        # Initialize arrays to collect data for this episode
        episode_prices = torch.zeros(market.num_hours)
        episode_dispatch_0 = torch.zeros(market.num_hours)
        episode_dispatch_1 = torch.zeros(market.num_hours)
        episode_dispatch_2 = torch.zeros(market.num_hours)
        episode_bids_0 = torch.zeros(market.num_hours)  # Strategic producer's bids
        timestep_count = 0

        for timestep in timesteps:
            # Get action and add exploration noise with decaying scale
            action = actor(state)
            noise = torch.normal(-current_noise_scale, current_noise_scale, size=action.shape)
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

            episode_price += market.prices.mean().item()

            # Collect data for this timestep
            episode_prices += market.prices[0]  # Using first batch item
            episode_dispatch_0 += market.g_i[0]  # Strategic producer dispatch

            # Calculate and store the strategic producer's bid
            # Bid = base cost * multiplier (k_factor)
            strategic_base_cost = market.generators[market.strategic_gen]["var_cost"]
            strategic_bids = strategic_base_cost * action[0]  # Using first batch item
            episode_bids_0 += strategic_bids

            # Get dispatch for non-strategic producers from market
            # We need to modify the market class to expose this data
            if hasattr(market, 'all_generator_dispatch'):
                episode_dispatch_1 += market.all_generator_dispatch[0][1]  # Producer 1
                episode_dispatch_2 += market.all_generator_dispatch[0][2]  # Producer 2

            timestep_count += 1

            if timestep == len(timesteps) // 2:
                wandb.log({
                    "timestep_prices": market.prices.mean().item(),
                    "timestep_bidding multiplier": action.mean().item(),
                    "timestep_average_ui_status": market.u_i.mean().item(),
                    "timestep_average_gi_status": market.g_i.mean().item(),
                    "current_noise_scale": current_noise_scale,  # Log the current noise scale
                },
                    step=episode)

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
                current_q = critic(states.detach(), actions.detach())
                assert current_q.shape == (BATCH_SIZE, 1)
                critic_loss = torch.nn.functional.mse_loss(current_q, target_value.detach())

                optimizer_critic.zero_grad()
                critic_loss.backward()
                # torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                optimizer_critic.step()

                # Update actor
                actor_actions = actor(states.detach())
                actor_loss = -critic(states.detach(), actor_actions).mean()

                optimizer_actor.zero_grad()
                actor_loss.backward()
                # torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                optimizer_actor.step()

                # Soft update targets
                soft_update(critic_target, critic, TAU)
                soft_update(actor_target, actor, TAU)

                # Logging
                if timestep == len(timesteps) // 2:
                    wandb.log({
                        "critic_loss": critic_loss.item(),
                        "actor_loss": actor_loss.item(),
                        "q_value": current_q.mean().item(),
                        "reward": rewards.mean().item(),
                    },
                        step=episode)

        # Average the data over timesteps
        episode_prices /= timestep_count
        episode_dispatch_0 /= timestep_count
        episode_dispatch_1 /= timestep_count
        episode_dispatch_2 /= timestep_count
        episode_bids_0 /= timestep_count

        # Store in the current batch tensor
        batch_idx = episode_counter % SAVE_FREQUENCY
        episode_data_batch[batch_idx, :, 0] = episode_prices
        episode_data_batch[batch_idx, :, 1] = episode_dispatch_0
        episode_data_batch[batch_idx, :, 2] = episode_dispatch_1
        episode_data_batch[batch_idx, :, 3] = episode_dispatch_2
        episode_data_batch[batch_idx, :, 4] = episode_bids_0  # Strategic producer's bids

        episode_counter += 1

        # Check if it's time to save the current batch
        if episode_counter % SAVE_FREQUENCY == 0 or episode == market.num_episodes - 1:
            # If this is the last episode and not a complete batch, trim the tensor
            if episode == market.num_episodes - 1 and episode_counter % SAVE_FREQUENCY != 0:
                last_batch_size = episode_counter % SAVE_FREQUENCY
                current_batch = episode_data_batch[:last_batch_size]
            else:
                current_batch = episode_data_batch

            # Concatenate with existing data and save
            updated_data = torch.cat([existing_data, current_batch], dim=0)
            torch.save(updated_data, output_file)
            print(f"Updated data file with shape: {updated_data.shape}")

            # Update existing data reference
            existing_data = updated_data

            # Reset the batch tensor for the next set of episodes
            episode_data_batch = torch.zeros((SAVE_FREQUENCY, market.num_hours, 5))

        wandb.log({
            "episode_reward": episode_reward,
            "episode_price": episode_price / len(timesteps),
            "episode_counter": episode_counter,
            "noise_scale": current_noise_scale,  # Log the noise scale
        },
            step=episode)

        max_reward_so_far = max(max_reward_so_far, episode_reward)
        print(
            f"Episode {episode}, Reward: {episode_reward:.2f}, Max Reward: {max_reward_so_far:.2f}, Noise: {current_noise_scale:.4f}")

    print(f"Final data saved with shape: {existing_data.shape}")

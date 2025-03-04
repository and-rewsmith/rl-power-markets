import torch
import wandb

from rl_power_markets.model.agent import Critic, Actor
from rl_power_markets.benchmarks.markets.simple import SimpleMarket

"""
This file functions as the harness that is parameterized by market type.
"""


def initialize_wandb() -> None:
    wandb.init(
        project="rl-power-markets",
        config={
            "architecture": "Simple",
        }
    )


LEARNING_RATE = 0.001
DISCOUNT = 0.7
ACTOR_HIDDEN_SIZE = 64
CRITIC_HIDDEN_SIZE = 64
BATCH_SIZE = 32


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    initialize_wandb()

    market = SimpleMarket()
    episodes = market.episodes
    timesteps = market.timesteps

    actor = Actor(obs_size=market.obs_size, hidden_size=ACTOR_HIDDEN_SIZE, num_actions=market.num_actions)
    critic = Critic(obs_size=market.obs_size, hidden_size=CRITIC_HIDDEN_SIZE)
    actor_and_critic_params = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(actor_and_critic_params, lr=LEARNING_RATE)

    running_reward = torch.zeros(market.batch_size)
    for episode in episodes:
        market.reset()
        for timestep in timesteps:
            market.step_basic_bids()
            state: torch.Tensor = market.obtain_state()
            assert state.shape == (market.batch_size, market.obs_size)

            action = actor(state)
            assert action.shape == (market.batch_size, market.num_actions)

            critic_value = critic(state, action)
            assert critic_value.shape == (market.batch_size,)

            new_state, reward = market.step(action)
            running_reward += reward
            assert new_state.shape == (market.batch_size, market.obs_size)
            assert reward.shape == (market.batch_size,)
            assert running_reward.shape == (market.batch_size,)

            new_action = actor(new_state)
            new_critic_value = critic(new_state, new_action)
            assert new_action.shape == (market.batch_size, market.num_actions)
            assert new_critic_value.shape == (market.batch_size,)

            td_error = reward + (
                0.99 * new_critic_value.detach()
            ) - critic_value
            assert td_error.shape == (market.batch_size,)

            loss = td_error ** 2
            assert loss.shape == (market.batch_size,)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({
                "loss": loss.item(),
                "reward": reward.item(),
                "critic_value": critic_value.item(),
                "new_critic_value": new_critic_value.item(),
                "running_reward": running_reward,
            })

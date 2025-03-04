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
    critic = Critic(obs_size=market.obs_size + market.num_actions, hidden_size=CRITIC_HIDDEN_SIZE)
    actor_and_critic_params = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(actor_and_critic_params, lr=LEARNING_RATE)

    running_reward = torch.zeros(market.batch_size, 1)
    for episode in episodes:
        market.reset()
        print(f"Episode {episode}")
        for timestep in timesteps:
            state: torch.Tensor = market.obtain_state().detach()
            assert state.shape == (market.batch_size, market.obs_size)

            action = actor(state)
            action = 1 + action
            assert action.shape == (market.batch_size, market.num_actions)

            critic_value = critic(state, action)
            assert critic_value.shape == (market.batch_size, 1)

            new_state, reward = market.step(action)
            running_reward += reward
            assert new_state.shape == (market.batch_size, market.obs_size)
            assert reward.shape == (market.batch_size, 1)
            assert running_reward.shape == (market.batch_size, 1)

            new_action = actor(new_state)
            new_critic_value = critic(new_state, new_action)
            assert new_action.shape == (market.batch_size, market.num_actions)
            assert new_critic_value.shape == (market.batch_size, 1)

            td_error = reward + (
                0.99 * new_critic_value.detach()
            ) - critic_value
            assert td_error.shape == (market.batch_size, 1)

            loss = td_error ** 2
            assert loss.shape == (market.batch_size, 1)

            average_loss = loss.mean()
            assert average_loss.shape == ()

            optimizer.zero_grad()
            average_loss.mean().backward()
            optimizer.step()

            wandb.log({
                "loss": average_loss.item(),
                "reward": reward.mean().item(),
                "critic_value": critic_value.mean().item(),
                "new_critic_value": new_critic_value.mean().item(),
                "running_reward": running_reward.mean().item(),
            })

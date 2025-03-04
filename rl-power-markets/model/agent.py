import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This file implements the agent from the paper.
"""


class Critic(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = torch.cat([state, action], dim=-1)
        out: torch.Tensor = self.net(x)
        return out


class Actor(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.net(x)
        return result


if __name__ == "__main__":
    # Test parameters
    obs_size = 4
    hidden_size = 64
    num_actions = 2
    batch_size = 3

    # Initialize networks
    critic = Critic(obs_size + num_actions, hidden_size)
    actor = Actor(obs_size, hidden_size, num_actions)

    # Create dummy input data
    states = torch.randn(batch_size, obs_size)
    actions = torch.randn(batch_size, num_actions)

    # Test actor forward pass
    actor_output = actor(states)
    assert actor_output.shape == (
        batch_size, num_actions), f"Expected shape {(batch_size, num_actions)}, got {actor_output.shape}"

    # Test critic forward pass
    critic_output = critic(states, actions)
    assert critic_output.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {critic_output.shape}"

    print("Sanity check passed!")

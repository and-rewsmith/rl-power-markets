import torch

"""
This file implements a simple market just designed for sanity check.
"""


class SimpleMarket:
    def __init__(self, batch_size: int = 32) -> None:
        # Market parameters
        self.agent_base_cost = 20.0  # $/MWh
        self.competitor_fixed_bid = 40.0  # $/MWh
        self.batch_size = batch_size

        # Define 24-hour demand profile (MW)
        # Simple profile: higher during day, lower at night
        base_demand = 800.0
        peak_multiplier = 1.25
        off_peak_multiplier = 0.625

        # Create demand profile and expand for batch dimension
        demand = torch.tensor([
            *[base_demand * off_peak_multiplier] * 6,    # 0-5: night
            *[base_demand * peak_multiplier] * 4,        # 6-9: morning peak
            *[base_demand] * 6,                          # 10-15: mid-day
            *[base_demand * peak_multiplier] * 4,        # 16-19: evening peak
            *[base_demand] * 4                           # 20-23: evening
        ])
        self.demand = demand.expand(batch_size, 24)  # Shape: [batch_size, 24]

        # Initialize state vectors with batch dimension
        self.u_i = torch.zeros(batch_size, 24)      # commitment status
        self.g_i = torch.zeros(batch_size, 24)      # power output
        self.prices = torch.zeros(batch_size, 24)   # market clearing prices

        # Add these properties
        self.obs_size = 72  # 24 each for u_i, g_i, and prices
        self.num_actions = 24  # one multiplier per hour
        self.episodes = range(1000)  # or whatever episode count you want
        self.timesteps = range(100)  # or whatever timestep count you want

    def reset(self) -> None:
        # Reset state vectors to zero
        self.u_i.zero_()
        self.g_i.zero_()
        self.prices.zero_()

    def step_basic_bids(self, multipliers: torch.Tensor) -> torch.Tensor:
        # Input validation
        assert multipliers.shape == (
            self.batch_size, 24), f"Expected shape ({self.batch_size}, 24), got {multipliers.shape}"
        assert (multipliers >= 1.0).all(), "All multipliers must be >= 1.0"
        # TODO: fix assert statement

        # Calculate agent's bids for all hours and batches
        agent_bids = self.agent_base_cost * multipliers  # Shape: [batch_size, 24]
        # TODO: add assert statement

        # Market clearing (using broadcasting)
        agent_wins = agent_bids < self.competitor_fixed_bid  # Shape: [batch_size, 24]
        # TODO: add assert statement

        # Update state vectors using masks
        self.u_i = agent_wins.float()
        self.g_i = self.demand * agent_wins.float()
        self.prices = torch.where(agent_wins, agent_bids,
                                  torch.full_like(agent_bids, self.competitor_fixed_bid))
        # TODO: add assert statements

        # Calculate profits
        profits = torch.where(
            agent_wins,
            (agent_bids - self.agent_base_cost) * self.demand,
            torch.zeros_like(agent_bids)
        )
        # TODO: add assert statement

        # Sum profits across hours for each batch
        total_profits = profits.sum(dim=1)
        # TODO: add assert statement

        return total_profits

    def obtain_state(self) -> torch.Tensor:
        # Concatenate along hour dimension, maintaining batch dimension
        # Shape: [batch_size, 72]
        out = torch.cat([self.u_i, self.g_i, self.prices], dim=1)
        # TODO: add assert statement
        return out

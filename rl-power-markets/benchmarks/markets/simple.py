import torch
import numpy as np

"""
This file implements a simple market just designed for sanity check.
"""


class SimpleMarket:
    def __init__(self) -> None:
        # Market parameters
        self.agent_base_cost = 20.0  # $/MWh
        self.competitor_fixed_bid = 40.0  # $/MWh

        # Define 24-hour demand profile (MW)
        # Simple profile: higher during day, lower at night
        base_demand = 800
        peak_multiplier = 1.25
        off_peak_multiplier = 0.625

        self.demand = np.array([
            *[base_demand * off_peak_multiplier] * 6,    # 0-5: night
            *[base_demand * peak_multiplier] * 4,        # 6-9: morning peak
            *[base_demand] * 6,                          # 10-15: mid-day
            *[base_demand * peak_multiplier] * 4,        # 16-19: evening peak
            *[base_demand] * 4                           # 20-23: evening
        ])

        # Initialize state vectors (24 hours each)
        self.u_i = np.zeros(24)  # commitment status
        self.g_i = np.zeros(24)  # power output
        self.prices = np.zeros(24)  # market clearing prices

        # Add these properties
        self.obs_size = 72  # 24 each for u_i, g_i, and prices
        self.num_actions = 24  # one multiplier per hour
        self.episodes = range(1000)  # or whatever episode count you want
        self.timesteps = range(100)  # or whatever timestep count you want

    def reset(self) -> None:
        # Reset state vectors to zero
        self.u_i.fill(0)
        self.g_i.fill(0)
        self.prices.fill(0)

    def step_basic_bids(self, multipliers: torch.Tensor) -> float:
        # Add input validation
        assert len(multipliers) == 24, f"Expected 24 multipliers, got {len(multipliers)}"
        assert (multipliers >= 1.0).all(), "All multipliers must be >= 1.0"

        # Convert multipliers to numpy if needed
        if isinstance(multipliers, torch.Tensor):
            multipliers = multipliers.detach().numpy()

        total_profit = 0.0

        # Clear market for each hour
        for h in range(24):
            # Calculate agent's bid
            agent_bid = self.agent_base_cost * multipliers[h]

            # Market clearing
            if agent_bid < self.competitor_fixed_bid:
                # Agent wins
                self.u_i[h] = 1
                self.g_i[h] = self.demand[h]
                self.prices[h] = agent_bid
                # Calculate profit = (price - cost) * quantity
                hour_profit = (agent_bid - self.agent_base_cost) * self.demand[h]
            else:
                # Competitor wins
                self.u_i[h] = 0
                self.g_i[h] = 0
                self.prices[h] = self.competitor_fixed_bid
                hour_profit = 0.0

            total_profit += hour_profit

        return total_profit

    def obtain_state(self) -> torch.Tensor:
        # Concatenate state vectors [u_i, g_i, prices] into tensor
        state = np.concatenate([self.u_i, self.g_i, self.prices])
        return torch.FloatTensor(state)

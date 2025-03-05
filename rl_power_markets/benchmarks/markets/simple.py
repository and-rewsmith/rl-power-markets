import torch

"""
This file implements a simple market just designed for sanity check.
"""


class SimpleMarket:
    def __init__(self, batch_size: int = 32) -> None:
        self.batch_size = batch_size
        self.num_hours = 24  # Number of hours in a day

        # Market parameters
        self.agent_base_cost = 20.0  # $/MWh
        self.competitor_fixed_bid = 40.0  # $/MWh

        # Define demand profile
        base_demand = 1.0
        peak_multiplier = 1.25
        off_peak_multiplier = 0.625

        demand = torch.tensor([
            *[base_demand * off_peak_multiplier] * 6,    # 0-5: night
            *[base_demand * peak_multiplier] * 4,        # 6-9: morning peak
            *[base_demand] * 6,                          # 10-15: mid-day
            *[base_demand * peak_multiplier] * 4,        # 16-19: evening peak
            *[base_demand] * 4                           # 20-23: evening
        ])
        self.demand = demand.expand(self.batch_size, self.num_hours)
        assert self.demand.shape == (self.batch_size, self.num_hours)

        # Initialize state vectors
        self.u_i = torch.zeros(self.batch_size, self.num_hours)
        assert self.u_i.shape == (self.batch_size, self.num_hours)

        self.g_i = torch.zeros(self.batch_size, self.num_hours)
        assert self.g_i.shape == (self.batch_size, self.num_hours)

        self.prices = torch.full((self.batch_size, self.num_hours), self.competitor_fixed_bid)
        assert self.prices.shape == (self.batch_size, self.num_hours)

        # State space dimensions
        self.state_dim_per_var = self.num_hours
        self.num_state_vars = 3  # u_i, g_i, and prices
        self.obs_size = self.state_dim_per_var * self.num_state_vars
        self.num_actions = self.num_hours  # one multiplier per hour

        self.num_episodes = 100000
        self.num_timesteps = 30
        self.episodes = range(self.num_episodes)
        self.timesteps = range(self.num_timesteps)

    def reset(self) -> None:
        # Reset state vectors
        self.u_i.zero_()
        self.g_i.zero_()
        # Reset prices to competitor's fixed bid
        self.prices.fill_(0)

    def step(self, multipliers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Validate input multipliers
        assert multipliers.shape == (self.batch_size, self.num_hours)
        assert (multipliers >= 1.0).all()

        # Calculate agent bids by applying multipliers to base cost
        agent_bids = self.agent_base_cost * multipliers
        # print(f"Agent bids: {agent_bids[0]}")
        assert agent_bids.shape == (self.batch_size, self.num_hours)

        # Determine which hours the agent wins based on competitor's fixed bid
        agent_wins = agent_bids < self.competitor_fixed_bid
        assert agent_wins.shape == (self.batch_size, self.num_hours)
        # print(f"Agent wins: {agent_wins[0]}")

        # Update market state variables
        self.u_i = agent_wins.float()
        self.g_i = self.demand * agent_wins.float()
        self.prices = torch.where(agent_wins, agent_bids,
                                  torch.full_like(agent_bids, self.competitor_fixed_bid))
        assert self.u_i.shape == (self.batch_size, self.num_hours)
        assert self.g_i.shape == (self.batch_size, self.num_hours)
        assert self.prices.shape == (self.batch_size, self.num_hours)

        # Calculate profits: (bid - cost) * demand when agent wins, 0 otherwise
        profits = torch.where(
            agent_wins,
            (agent_bids - self.agent_base_cost) * self.demand,
            torch.zeros_like(agent_bids)
        )
        assert profits.shape == (self.batch_size, self.num_hours)
        # print(f"Profits: {profits[0]}")
        # input()

        # Sum profits across all hours
        total_profits = profits.sum(dim=1)
        assert total_profits.shape == (self.batch_size,)

        return self.obtain_state(), total_profits.unsqueeze(1)

    def obtain_state(self) -> torch.Tensor:
        out = torch.cat([self.u_i, self.g_i, self.prices], dim=1)
        assert out.shape == (self.batch_size, self.obs_size)
        return out

import torch
import xpress as xp
import numpy as np


class FullSimpleMarket:
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        self.num_hours = 10

        # Define a small number of generators with variable costs
        self.generators = {
            0: {"g_min": 100, "g_max": 500, "CSU": 1000.0, "CSD": 500.0, "u0": 1, "var_cost": 20.0},
            1: {"g_min": 200, "g_max": 600, "CSU": 1500.0, "CSD": 750.0, "u0": 0, "var_cost": 30.0},
            2: {"g_min": 150, "g_max": 550, "CSU": 1200.0, "CSD": 600.0, "u0": 0, "var_cost": 25.0},
        }

        # Simple demand profile
        self.demand_profile = torch.tensor([
            300, 280, 260, 250, 270, 320, 400, 450, 480, 500, 520, 530,
            510, 490, 470, 450, 430, 420, 410, 400, 390, 380, 370, 360
        ])

        self.num_actions = self.num_hours  # one multiplier per hour
        self.obs_size = self.num_hours * 3  # u_i, g_i, and prices for each hour

        # Initialize state tensors
        self.u_i = torch.zeros(self.batch_size, self.num_hours)  # commitment of strategic generator
        self.g_i = torch.zeros(self.batch_size, self.num_hours)  # output of strategic generator
        self.prices = torch.zeros(self.batch_size, self.num_hours)  # market prices

        self.num_episodes = 10000
        self.num_timesteps = 10  # One month of daily timesteps
        self.episodes = range(self.num_episodes)
        self.timesteps = range(self.num_timesteps)

        # Strategic generator ID
        self.strategic_gen = 0  # ID of strategic generator

    def reset(self) -> torch.Tensor:
        self.u_i.zero_()
        self.g_i.zero_()
        self.prices.zero_()
        return self.obtain_state()

    def step(self, multipliers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert multipliers.shape == (self.batch_size, self.num_hours)
        assert (multipliers >= 1.0).all()

        profits = torch.zeros(self.batch_size, 1)

        # Process each batch item separately
        for b in range(self.batch_size):
            k_factors = multipliers[b].detach().numpy()

            # Solve market clearing
            model = self._build_mip_model(k_factors)
            model.solve()

            # Get binary solutions
            binary_solutions = self._get_binary_solutions(model)

            # Solve pricing problem
            cont_model = self._build_lp_model(binary_solutions)
            cont_model.solve()

            # Get market prices and update state
            market_prices = self._get_market_prices(cont_model)
            self.prices[b] = torch.tensor(list(market_prices.values()))

            # Update commitment and generation
            for h in range(self.num_hours):
                self.u_i[b, h] = binary_solutions[(self.strategic_gen, h, 'u')]
                self.g_i[b, h] = model.getSolution(self.g_blocks[self.strategic_gen][h])

            # Calculate profits
            profits[b] = self._calculate_profit(model, market_prices, k_factors[0])

        return self.obtain_state(), profits

    def obtain_state(self) -> torch.Tensor:
        out = torch.cat([self.u_i, self.g_i, self.prices], dim=1)
        assert out.shape == (self.batch_size, self.obs_size)
        return out

    def _build_mip_model(self, k_factors):
        model = xp.problem('simple market')

        # Suppress solver output
        model.setControl({
            'outputlog': 0,
            'lplog': 0,
            'miplog': 0,
            'baroutput': 0,
        })

        H = range(self.num_hours)

        # Generator variables
        self.g_blocks = {}
        self.u = {}  # Make u accessible to other methods
        self.su = {}  # Make su accessible to other methods
        self.sd = {}  # Make sd accessible to other methods

        for i in self.generators:
            g_min = self.generators[i]["g_min"]
            g_max = self.generators[i]["g_max"]

            # Block variables (output per block)
            self.g_blocks[i] = {
                h: model.addVariable(lb=0, ub=g_max, name=f'g_{i}_{h}')
                for h in H
            }

            # Commitment variables
            self.u[i] = {h: model.addVariable(vartype=xp.binary, name=f'u_{i}_{h}') for h in H}
            self.su[i] = {h: model.addVariable(vartype=xp.binary, name=f'su_{i}_{h}') for h in H}
            self.sd[i] = {h: model.addVariable(vartype=xp.binary, name=f'sd_{i}_{h}') for h in H}

            # Minimum output constraint if generator is on
            for h in H:
                model.addConstraint(
                    self.g_blocks[i][h] >= g_min * self.u[i][h]
                )
                model.addConstraint(
                    self.g_blocks[i][h] <= g_max * self.u[i][h]
                )

                # Startup and shutdown constraints
                if h > 0:
                    model.addConstraint(self.u[i][h] - self.u[i][h-1] <= self.su[i][h])
                    model.addConstraint(self.u[i][h-1] - self.u[i][h] <= self.sd[i][h])
                else:
                    model.addConstraint(self.u[i][h] - self.generators[i]["u0"] <= self.su[i][h])
                    model.addConstraint(self.generators[i]["u0"] - self.u[i][h] <= self.sd[i][h])

        # Power balance constraints
        for h in H:
            model.addConstraint(
                xp.Sum(self.g_blocks[i][h] for i in self.generators) == self.demand_profile[h].item()
            )

        total_demand = 0
        for i in self.generators:
            for h in H:
                total_demand += self.g_blocks[i][h]

        # Objective function
        k = {i: {h: 1.0 for h in H} for i in self.generators}
        k[self.strategic_gen] = {h: k_factors[h] for h in H}

        objective = (
            xp.Sum(
                self.generators[i]["var_cost"] * k[i][h] * self.g_blocks[i][h]
                for i in self.generators for h in H
            ) + xp.Sum(
                self.generators[i]["CSU"] * self.su[i][h] + self.generators[i]["CSD"] * self.sd[i][h]
                for i in self.generators for h in H
            )
        )

        model.setObjective(objective, sense=xp.minimize)
        return model

    def _get_binary_solutions(self, model):
        binary_solutions = {}
        for i in self.generators:
            for h in range(self.num_hours):
                binary_solutions[(i, h, 'u')] = model.getSolution(self.u[i][h])
                binary_solutions[(i, h, 'su')] = model.getSolution(self.su[i][h])
                binary_solutions[(i, h, 'sd')] = model.getSolution(self.sd[i][h])
        return binary_solutions

    def _build_lp_model(self, binary_solutions):
        model = xp.problem('continuous market')

        # Suppress solver output
        model.setControl({
            'outputlog': 0,
            'lplog': 0,
            'baroutput': 0,
        })

        # Ensure duals are computed
        model.setControl({'lpflags': 0x40})  # Set flag to compute duals

        H = range(self.num_hours)

        # Generator block variables
        self.cont_g_blocks = {}

        for i in self.generators:
            g_min = self.generators[i]["g_min"]
            g_max = self.generators[i]["g_max"]

            self.cont_g_blocks[i] = {
                h: model.addVariable(lb=0, ub=g_max, name=f'g_{i}_{h}')
                for h in H
            }

            for h in H:
                model.addConstraint(
                    self.cont_g_blocks[i][h] >= g_min * binary_solutions[(i, h, 'u')]
                )
                model.addConstraint(
                    self.cont_g_blocks[i][h] <= g_max * binary_solutions[(i, h, 'u')]
                )

        # Power balance constraints
        self.power_balance = {}  # Store constraints for accessing duals later
        for h in H:
            self.power_balance[h] = model.addConstraint(
                xp.Sum(self.cont_g_blocks[i][h] for i in self.generators) == self.demand_profile[h].item()
            )

        # Objective function (no strategic bidding in pricing run)
        objective = (
            xp.Sum(
                self.generators[i]["var_cost"] * self.cont_g_blocks[i][h]
                for i in self.generators for h in H
            ) + xp.Sum(
                self.generators[i]["CSU"] * binary_solutions[(i, h, 'su')] +
                self.generators[i]["CSD"] * binary_solutions[(i, h, 'sd')]
                for i in self.generators for h in H
            )
        )

        model.setObjective(objective, sense=xp.minimize)
        return model

    def _get_market_prices(self, model):
        market_prices = {}

        for h in range(self.num_hours):
            # try:
            #     dual = model.getDuals(self.power_balance[h])
            #     if abs(dual) > 0.001:
            #         market_prices[h] = dual
            #         continue
            # except xp.ModelError:
            #     # If duals aren't available, fall back to marginal generator approach
            #     pass

            # Fallback to marginal generator approach
            max_cost = 0
            marginal_gen = None

            for i in self.generators:
                if model.getSolution(self.cont_g_blocks[i][h]) > 0.001:
                    cost = self.generators[i]["var_cost"]  # Use variable cost
                    if cost > max_cost:
                        max_cost = cost
                        marginal_gen = i

            market_prices[h] = max_cost

        return market_prices

    def _calculate_profit(self, model, market_prices, k):
        profit = 0
        total_quantity = 0
        total_revenue = 0
        total_costs = 0

        for h in range(self.num_hours):
            quantity = model.getSolution(self.g_blocks[self.strategic_gen][h])
            revenue = market_prices[h] * quantity

            # Use true costs (not inflated by k) for profit calculation
            var_cost = self.generators[self.strategic_gen]["var_cost"] * quantity
            startup_cost = self.generators[self.strategic_gen]["CSU"] * \
                model.getSolution(self.su[self.strategic_gen][h])
            shutdown_cost = self.generators[self.strategic_gen]["CSD"] * \
                model.getSolution(self.sd[self.strategic_gen][h])

            total_cost = var_cost + startup_cost + shutdown_cost
            hour_profit = revenue - total_cost
            profit += hour_profit

            total_quantity += quantity
            total_revenue += revenue
            total_costs += total_cost

        return profit

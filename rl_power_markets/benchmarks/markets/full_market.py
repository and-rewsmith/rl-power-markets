import torch
import xpress as xp
import numpy as np


class FullMarket:
    def __init__(self, batch_size: int = 32) -> None:
        self.batch_size = batch_size
        self.num_hours = 24
        self.num_blocks = 5

        self.num_actions = self.num_hours  # one multiplier per hour
        self.obs_size = self.num_hours * 3  # u_i, g_i, and prices for each hour

        # Generator parameters
        self.generators = {
            0: {"a": 9900.0, "b": 80.0, "c": 0.0070, "CSU": 55000.0, "CSD": 12000.0,
                "g_min": 650, "g_max": 3252, "RU": 1951, "RD": 1951, "UT": 8, "DT": 8, "u0": 1},
            1: {"a": 18431.0, "b": 5.5, "c": 0.0002, "CSU": 4000000.0, "CSD": 800000.0,
                "g_min": 3292, "g_max": 6584, "RU": 1317, "RD": 1317, "UT": 24, "DT": 24, "u0": 1},
            2: {"a": 17005.0, "b": 30.0, "c": 0.0007, "CSU": 325000.0, "CSD": 28500.0,
                "g_min": 2880, "g_max": 5760, "RU": 1152, "RD": 1152, "UT": 20, "DT": 20, "u0": 1},
            3: {"a": 13755.0, "b": 35.0, "c": 0.0010, "CSU": 142500.0, "CSD": 18500.0,
                "g_min": 1512, "g_max": 3781, "RU": 1512, "RD": 1512, "UT": 16, "DT": 16, "u0": 1},
            4: {"a": 9930.0, "b": 60.0, "c": 0.0064, "CSU": 72000.0, "CSD": 14400.0,
                "g_min": 667, "g_max": 3335, "RU": 1334, "RD": 1334, "UT": 10, "DT": 10, "u0": 1},
            6: {"a": 8570.0, "b": 95.0, "c": 0.0082, "CSU": 31000.0, "CSD": 10000.0,
                "g_min": 288, "g_max": 2880, "RU": 1728, "RD": 1728, "UT": 5, "DT": 5, "u0": 0},
            7: {"a": 7530.0, "b": 100.0, "c": 0.0098, "CSU": 11200.0, "CSD": 8400.0,
                "g_min": 275, "g_max": 2748, "RU": 2198, "RD": 2198, "UT": 4, "DT": 4, "u0": 0}
        }

        # Demand parameters with extreme peak characteristics
        base_demand = torch.tensor([
            15000,  # 00:00 - Night
            14000,  # 01:00 - Night
            13000,  # 02:00 - Lowest demand
            12500,  # 03:00 - Lowest demand
            13000,  # 04:00 - Starting to rise
            14500,  # 05:00 - Morning ramp begins
            17000,  # 06:00 - Morning ramp
            20000,  # 07:00 - Morning peak starts
            22000,  # 08:00 - Morning peak
            23000,  # 09:00 - Business hours
            23500,  # 10:00 - Business hours peak
            24000,  # 11:00 - Business hours peak
            24000,  # 12:00 - Business hours peak
            23500,  # 13:00 - Business hours
            23000,  # 14:00 - Afternoon dip starts
            22500,  # 15:00 - Afternoon dip
            22000,  # 16:00 - Beginning of evening ramp
            23000,  # 17:00 - Evening ramp
            25000,  # 18:00 - Evening peak
            26000,  # 19:00 - Highest evening peak
            25000,  # 20:00 - Evening peak declining
            22000,  # 21:00 - Evening decline
            18000,  # 22:00 - Late evening
            16000,  # 23:00 - Night
        ])

        # Apply demand multiplier (1.2) and peak multiplier (2.5) for peak hours (10:00-22:00)
        peak_hours = torch.tensor([1 if 10 <= h <= 21 else 0 for h in range(24)], dtype=torch.float32)
        # 1.5 additional multiplier for peak hours to achieve 2.5x
        self.base_demand_profile = base_demand * (1 + peak_hours * 2.5)
        # self.base_demand_profile = base_demand

        self.base_price = 150  # Increased base price
        self.price_step = 25   # Updated price step

        self.num_episodes = 10000
        self.num_timesteps = 10  # One month of daily timesteps
        self.episodes = range(self.num_episodes)
        self.timesteps = range(self.num_timesteps)

        # Initialize state tensors
        self.u_i = torch.zeros(self.batch_size, self.num_hours)  # commitment of strategic generator
        self.g_i = torch.zeros(self.batch_size, self.num_hours)  # output of strategic generator
        self.prices = torch.zeros(self.batch_size, self.num_hours)  # market prices

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
                self.g_i[b, h] = sum(model.getSolution(self.g_blocks[self.strategic_gen][h][b])
                                     for b in range(self.num_blocks))

            # Calculate profits
            profits[b] = self._calculate_profit(model, market_prices, k_factors[0])

        # print(self.u_i)
        # input()

        return self.obtain_state(), profits

    def obtain_state(self) -> torch.Tensor:
        out = torch.cat([self.u_i, self.g_i, self.prices], dim=1)
        assert out.shape == (self.batch_size, self.obs_size)
        return out

    def _build_mip_model(self, k_factors):
        model = xp.problem('full market')

        # Suppress solver output
        model.setControl({
            'outputlog': 0,
            'lplog': 0,
            'miplog': 0,
            'baroutput': 0,
        })

        H = range(self.num_hours)

        # Generator block variables and parameters
        self.g_blocks = {}
        lambdaG = {}
        self.u = {}  # Make u accessible to other methods
        self.su = {}  # Make su accessible to other methods
        self.sd = {}  # Make sd accessible to other methods

        for i in self.generators:
            g_min = self.generators[i]["g_min"]
            g_max = self.generators[i]["g_max"]
            step = (g_max - g_min) / self.num_blocks

            # Marginal cost for each block
            lambdaG[i] = [
                self.generators[i]["b"] + 2 * self.generators[i]["c"] * (g_min + (b + 0.5) * step)
                for b in range(self.num_blocks)
            ]

            # Block variables (output per block)
            self.g_blocks[i] = {
                h: [model.addVariable(lb=0, ub=step, name=f'g_{i}_{h}_block_{b}')
                    for b in range(self.num_blocks)]
                for h in H
            }

            # Commitment variables
            self.u[i] = {h: model.addVariable(vartype=xp.binary, name=f'u_{i}_{h}') for h in H}
            self.su[i] = {h: model.addVariable(vartype=xp.binary, name=f'su_{i}_{h}') for h in H}
            self.sd[i] = {h: model.addVariable(vartype=xp.binary, name=f'sd_{i}_{h}') for h in H}

            # Minimum output constraint if generator is on
            for h in H:
                model.addConstraint(
                    xp.Sum(self.g_blocks[i][h][b] for b in range(self.num_blocks)) >= g_min * self.u[i][h]
                )
                model.addConstraint(
                    xp.Sum(self.g_blocks[i][h][b] for b in range(self.num_blocks)) <= (g_max - g_min) * self.u[i][h]
                )

                # Startup and shutdown constraints
                if h > 0:
                    model.addConstraint(self.u[i][h] - self.u[i][h-1] <= self.su[i][h])
                    model.addConstraint(self.u[i][h-1] - self.u[i][h] <= self.sd[i][h])
                else:
                    model.addConstraint(self.u[i][h] - self.generators[i]["u0"] <= self.su[i][h])
                    model.addConstraint(self.generators[i]["u0"] - self.u[i][h] <= self.sd[i][h])

        # Demand blocks
        d = {}
        demand_blocks = {}

        for h in H:
            blocks = []
            base_demand = self.base_demand_profile[h]
            total_block_demand = 0

            for b in range(self.num_blocks):
                max_demand = base_demand * (0.7 / (b + 1))
                marginal_price = self.base_price - b * self.price_step
                blocks.append({"lambdaD": marginal_price, "d_max": max_demand})
                total_block_demand += max_demand

            demand_blocks[h] = blocks
            d[h] = [model.addVariable(lb=0, ub=blocks[c]["d_max"]) for c in range(len(blocks))]

        # Power balance constraints
        for h in H:
            model.addConstraint(
                xp.Sum(xp.Sum(self.g_blocks[i][h][b] for b in range(self.num_blocks)) for i in self.generators) ==
                xp.Sum(d[h][c] for c in range(len(demand_blocks[h])))
            )

        # Objective function
        k = {i: {h: 1.0 for h in H} for i in self.generators}
        k[self.strategic_gen] = {h: k_factors[h] for h in H}

        objective = (
            xp.Sum(
                xp.Sum(
                    lambdaG[i][b] * self.g_blocks[i][h][b] * k[i][h]
                    for b in range(self.num_blocks)
                ) + self.generators[i]["a"] * self.u[i][h]
                for i in self.generators for h in H
            ) + xp.Sum(
                self.generators[i]["CSU"] * self.su[i][h] + self.generators[i]["CSD"] * self.sd[i][h]
                for i in self.generators for h in H
            ) - xp.Sum(
                demand_blocks[h][c]["lambdaD"] * d[h][c]
                for h in H for c in range(len(demand_blocks[h]))
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
            'miplog': 0,
            'baroutput': 0,
        })

        H = range(self.num_hours)

        # Generator block variables
        self.cont_g_blocks = {}
        cont_u = {}
        cont_su = {}
        cont_sd = {}

        for i in self.generators:
            g_min = self.generators[i]["g_min"]
            g_max = self.generators[i]["g_max"]
            step = (g_max - g_min) / self.num_blocks

            self.cont_g_blocks[i] = {
                h: [model.addVariable(lb=0, ub=step, name=f'g_{i}_{h}_block_{b}')
                    for b in range(self.num_blocks)]
                for h in H
            }

            cont_u[i] = {
                h: model.addVariable(
                    lb=binary_solutions[(i, h, 'u')],
                    ub=binary_solutions[(i, h, 'u')],
                    name=f'u_{i}_{h}'
                ) for h in H
            }

            for h in H:
                model.addConstraint(
                    xp.Sum(self.cont_g_blocks[i][h][b] for b in range(self.num_blocks)) >= g_min * cont_u[i][h]
                )
                model.addConstraint(
                    xp.Sum(self.cont_g_blocks[i][h][b]
                           for b in range(self.num_blocks)) <= (g_max - g_min) * cont_u[i][h]
                )

        # Demand blocks
        cont_d = {}
        demand_blocks = {}
        for h in H:
            blocks = []
            base_demand = self.base_demand_profile[h]
            for b in range(self.num_blocks):
                max_demand = base_demand * (0.7 / (b + 1))
                marginal_price = self.base_price - b * self.price_step
                blocks.append({"lambdaD": marginal_price, "d_max": max_demand})
            demand_blocks[h] = blocks
            cont_d[h] = [model.addVariable(lb=0, ub=blocks[c]["d_max"]) for c in range(len(blocks))]

        # Power balance constraints with row names for dual extraction
        self.power_balance = {}
        for h in H:
            self.power_balance[h] = model.addConstraint(
                xp.Sum(xp.Sum(self.cont_g_blocks[i][h][b] for b in range(self.num_blocks)) for i in self.generators) ==
                xp.Sum(cont_d[h][c] for c in range(len(demand_blocks[h])))
            )

        # Objective function (no strategic bidding in pricing run)
        objective = (
            xp.Sum(
                xp.Sum(
                    (self.generators[i]["b"] + 2 * self.generators[i]["c"] *
                     (self.generators[i]["g_min"] + (b + 0.5) * step)) * self.cont_g_blocks[i][h][b]
                    for b in range(self.num_blocks)
                ) + self.generators[i]["a"] * cont_u[i][h]
                for i in self.generators for h in H
            ) - xp.Sum(
                demand_blocks[h][c]["lambdaD"] * cont_d[h][c]
                for h in H for c in range(len(demand_blocks[h]))
            )
        )

        model.setObjective(objective, sense=xp.minimize)
        return model

    def _get_market_prices(self, model):
        market_prices = {}

        for h in range(self.num_hours):
            try:
                dual = model.getDuals(self.power_balance[h])
                if abs(dual) > 0.001:
                    market_prices[h] = dual
                    continue
            except:
                pass

            # Fallback to marginal generator approach
            max_cost = 0
            marginal_gen = None
            marginal_block = None

            for i in self.generators:
                for b in range(self.num_blocks):
                    if model.getSolution(self.cont_g_blocks[i][h][b]) > 0.001:
                        cost = self.generators[i]["b"] + 2 * self.generators[i]["c"] * (
                            self.generators[i]["g_min"] + (b + 0.5) * (
                                self.generators[i]["g_max"] - self.generators[i]["g_min"]
                            ) / self.num_blocks
                        )
                        if cost > max_cost:
                            max_cost = cost
                            marginal_gen = i
                            marginal_block = b

            market_prices[h] = max_cost

        return market_prices

    def _calculate_profit(self, model, market_prices, k):
        profit = 0
        total_quantity = 0
        total_revenue = 0
        total_costs = 0

        for h in range(self.num_hours):
            quantity = sum(model.getSolution(self.g_blocks[self.strategic_gen][h][b])
                           for b in range(self.num_blocks))
            revenue = market_prices[h] * quantity

            # Use true costs (not inflated by k) for profit calculation
            gen_cost = sum(
                (self.generators[self.strategic_gen]["b"] +
                 2 * self.generators[self.strategic_gen]["c"] * (
                     self.generators[self.strategic_gen]["g_min"] + (b + 0.5) * (
                         self.generators[self.strategic_gen]["g_max"] -
                         self.generators[self.strategic_gen]["g_min"]
                     ) / self.num_blocks
                )) * model.getSolution(self.g_blocks[self.strategic_gen][h][b])
                for b in range(self.num_blocks)
            )

            no_load_cost = self.generators[self.strategic_gen]["a"] * model.getSolution(self.u[self.strategic_gen][h])
            startup_cost = self.generators[self.strategic_gen]["CSU"] * \
                model.getSolution(self.su[self.strategic_gen][h])
            shutdown_cost = self.generators[self.strategic_gen]["CSD"] * \
                model.getSolution(self.sd[self.strategic_gen][h])
            fixed_costs = no_load_cost + startup_cost + shutdown_cost

            hour_profit = revenue - (gen_cost + fixed_costs)
            profit += hour_profit

            total_quantity += quantity
            total_revenue += revenue
            total_costs += gen_cost + fixed_costs

        return profit

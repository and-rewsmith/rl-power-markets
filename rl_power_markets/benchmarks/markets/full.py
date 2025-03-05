import gurobipy as gp
from gurobipy import GRB
import torch

def solve_lower_level(generators, demand, H, C, k_factors):
    m = gp.Model("MarketClearing")
    u, su, sd, g, d = {}, {}, {}, {}, {}
    bigM = {}
    for i in generators:
        bigM[i] = sum(x[3] for x in generators[i]["blocks"])
        for h in range(H):
            u[i, h] = m.addVar(vtype=GRB.BINARY, name=f"u_{i}_{h}")
            su[i, h] = m.addVar(vtype=GRB.BINARY, name=f"su_{i}_{h}")
            sd[i, h] = m.addVar(vtype=GRB.BINARY, name=f"sd_{i}_{h}")
            for b in range(C):
                g[i, h, b] = m.addVar(lb=0.0, name=f"g_{i}_{h}_{b}")
    for h in range(H):
        for c in range(C):
            d[h, c] = m.addVar(lb=0.0, name=f"d_{h}_{c}")
    m.update()
    obj = gp.LinExpr()
    for i in generators:
        for h in range(H):
            obj += generators[i]["CNL"] * u[i, h]
            if "CSU" in generators[i]:
                obj += generators[i]["CSU"] * su[i, h]
            if "CSD" in generators[i]:
                obj += generators[i]["CSD"] * sd[i, h]
            for b in range(C):
                cost = generators[i]["b"]
                if i == 5:
                    cost *= k_factors[h]
                obj += cost * g[i, h, b]
    for h in range(H):
        for c in range(C):
            obj -= demand[h][c][1] * d[h, c]
    m.setObjective(obj, GRB.MINIMIZE)
    for h in range(H):
        lhs_gen = gp.LinExpr()
        lhs_dem = gp.LinExpr()
        for i in generators:
            for b in range(C):
                lhs_gen += g[i, h, b]
        for c in range(C):
            lhs_dem += d[h, c]
        m.addConstr(lhs_gen == lhs_dem, name=f"balance_{h}")
    for i in generators:
        for h in range(H):
            m.addConstr(gp.quicksum(g[i, h, b] for b in range(C)) <= bigM[i] * u[i, h], name=f"cap_{i}_{h}")
    for i in generators:
        for h in range(1, H):
            m.addConstr(u[i, h] - u[i, h-1] <= su[i, h], name=f"start_{i}_{h}")
            m.addConstr(u[i, h-1] - u[i, h] <= sd[i, h], name=f"shut_{i}_{h}")
            m.addConstr(su[i, h] + sd[i, h] <= 1, name=f"switch_{i}_{h}")
    for i in generators:
        for h in range(1, H):
            m.addConstr(gp.quicksum(g[i, h, b] for b in range(C)) - gp.quicksum(g[i, h-1, b] for b in range(C)) <= generators[i]["RU"], name=f"rampup_{i}_{h}")
            m.addConstr(gp.quicksum(g[i, h-1, b] for b in range(C)) - gp.quicksum(g[i, h, b] for b in range(C)) <= generators[i]["RD"], name=f"rampdown_{i}_{h}")
    for i in generators:
        UT = generators[i]["UT"]
        DT = generators[i]["DT"]
        for h in range(H - UT + 1):
            m.addConstr(gp.quicksum(1 - u[i, t] for t in range(h, h+UT)) <= UT - 1, name=f"minup_{i}_{h}")
        for h in range(H - DT + 1):
            m.addConstr(gp.quicksum(u[i, t] for t in range(h, h+DT)) <= 1, name=f"mindown_{i}_{h}")
    m.optimize()
    return m, u, g, d

class FullMarket:
    def __init__(self, batch_size: int = 32) -> None:
        self.batch_size = batch_size
        self.num_hours = 24
        
        # Initialize generators data
        self.generators = {
            1: {"b":5.5, "CNL":80000, "g_max":6.584, "RU":1.317, "RD":1.317, "CSU":400000, "CSD":80000, "UT":24, "DT":24},
            2: {"b":30.0, "CNL":6500, "g_max":5.760, "RU":1.152, "RD":1.152, "CSU":32500, "CSD":2850, "UT":20, "DT":20},
            3: {"b":35.0, "CNL":2851, "g_max":3.781, "RU":1.512, "RD":1.512, "CSU":14250, "CSD":1850, "UT":16, "DT":16},
            4: {"b":60.0, "CNL":1440, "g_max":3.335, "RU":1.334, "RD":1.334, "CSU":7200, "CSD":1440, "UT":10, "DT":10},
            5: {"b":80.0, "CNL":1100, "g_max":3.252, "RU":1.951, "RD":1.951, "CSU":5500, "CSD":1200, "UT":8, "DT":8},
            6: {"b":95.0, "CNL":620, "g_max":2.880, "RU":1.728, "RD":1.728, "CSU":3100, "CSD":1000, "UT":5, "DT":5},
            7: {"b":100.0, "CNL":560, "g_max":2.748, "RU":2.198, "RD":2.198, "CSU":1120, "CSD":840, "UT":4, "DT":4}
        }
        
        # Setup block structure for each generator
        self.C = 5  # number of blocks
        for i in self.generators:
            step = (self.generators[i]["g_max"] - 0)/self.C
            blocks = []
            for b in range(self.C):
                low = b * step
                high = (b+1) * step
                mid = 0.5*(low+high)
                slope = self.generators[i]["b"] + 2 * 0.0001 * mid
                cap = high - low
                blocks.append((low, high, slope, cap))
            self.generators[i]["blocks"] = blocks
        
        # Initialize demand data
        self.base_demand = sum(gen["g_max"] for gen in self.generators.values()) * 1.0
        self.demand = []
        for h in range(self.num_hours):
            blk = []
            cap = 10000
            dec = 5
            for c in range(self.C):
                blk.append((cap/self.C, (dec-c)*20))
            self.demand.append(blk)
        
        # Initialize state tensors
        self.u_i = torch.zeros(self.batch_size, self.num_hours)
        self.g_i = torch.zeros(self.batch_size, self.num_hours)
        self.prices = torch.zeros(self.batch_size, self.num_hours)
        
        # State space dimensions
        self.state_dim_per_var = self.num_hours
        self.num_state_vars = 3  # u_i, g_i, and prices
        self.obs_size = self.state_dim_per_var * self.num_state_vars
        self.num_actions = self.num_hours
        
        self.num_episodes = 100000
        self.num_timesteps = 30
        self.episodes = range(self.num_episodes)
        self.timesteps = range(self.num_timesteps)
    
    def reset(self) -> None:
        self.u_i.zero_()
        self.g_i.zero_()
        self.prices.zero_()
    
    def step(self, multipliers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert multipliers.shape == (self.batch_size, self.num_hours)
        assert (multipliers >= 1.0).all()
        
        profits = torch.zeros(self.batch_size, 1)
        
        # Process each batch item separately since Gurobi doesn't handle batches
        for b in range(self.batch_size):
            k_factors = multipliers[b].numpy()
            
            try:
                m, u, g, d = solve_lower_level(self.generators, self.demand, self.num_hours, self.C, k_factors)
                
                if m.Status == GRB.OPTIMAL:
                    # Update state tensors for this batch item
                    for h in range(self.num_hours):
                        self.u_i[b, h] = u[5, h].X  # Generator 5 is our strategic agent
                        self.g_i[b, h] = sum(g[5, h, c].X for c in range(self.C))
                        # Price is the dual variable of the balance constraint
                        self.prices[b, h] = m.getConstrByName(f"balance_{h}").Pi
                    
                    # Calculate profits for generator 5 (our strategic agent)
                    batch_profits = 0
                    for h in range(self.num_hours):
                        revenue = self.prices[b, h] * self.g_i[b, h]
                        cost = self.generators[5]["b"] * k_factors[h] * self.g_i[b, h]
                        if self.u_i[b, h] > 0.5:  # If unit is committed
                            cost += self.generators[5]["CNL"]
                        batch_profits += revenue - cost
                    profits[b] = batch_profits
                
            except gp.GurobiError as e:
                print(f"Gurobi error in batch {b}: {str(e)}")
                # Keep zeros for this batch item's state and profits
        
        return self.obtain_state(), profits
    
    def obtain_state(self) -> torch.Tensor:
        out = torch.cat([self.u_i, self.g_i, self.prices], dim=1)
        assert out.shape == (self.batch_size, self.obs_size)
        return out
# TODO: Clean up and remove the code below
if __name__ == "__main__":
    generators = {
        1: {"b":5.5, "CNL":80000, "g_max":6.584, "RU":1.317, "RD":1.317, "CSU":400000, "CSD":80000, "UT":24, "DT":24},
        2: {"b":30.0, "CNL":6500, "g_max":5.760, "RU":1.152, "RD":1.152, "CSU":32500, "CSD":2850, "UT":20, "DT":20},
        3: {"b":35.0, "CNL":2851, "g_max":3.781, "RU":1.512, "RD":1.512, "CSU":14250, "CSD":1850, "UT":16, "DT":16},
        4: {"b":60.0, "CNL":1440, "g_max":3.335, "RU":1.334, "RD":1.334, "CSU":7200, "CSD":1440, "UT":10, "DT":10},
        5: {"b":80.0, "CNL":1100, "g_max":3.252, "RU":1.951, "RD":1.951, "CSU":5500, "CSD":1200, "UT":8, "DT":8},
        6: {"b":95.0, "CNL":620, "g_max":2.880, "RU":1.728, "RD":1.728, "CSU":3100, "CSD":1000, "UT":5, "DT":5},
        7: {"b":100.0, "CNL":560, "g_max":2.748, "RU":2.198, "RD":2.198, "CSU":1120, "CSD":840, "UT":4, "DT":4}
    }
    for i in generators:
        C = 5
        step = (generators[i]["g_max"] - 0)/C  # Assuming g_min=0 for simplicity in this test
        blocks = []
        for b in range(C):
            low = b * step
            high = (b+1) * step
            mid = 0.5*(low+high)
            slope = generators[i]["b"] + 2 * 0.0001 * mid  # Using a fixed c value (adjust as needed)
            cap = high - low
            blocks.append((low, high, slope, cap))
        generators[i]["blocks"] = blocks
    H = 24
    C = 5
    total_capacity = sum(gen["g_max"] for gen in generators.values())
    base_demand = total_capacity * 1.0
    demand = []
    for h in range(H):
        blk = []
        cap = 10000
        dec = 5
        for c in range(C):
            blk.append((cap/C, (dec-c)*20))
        demand.append(blk)
    k_factors = [1.0 for _ in range(H)]
    m, u, g, d = solve_lower_level(generators, demand, H, C, k_factors)
    if m.Status == GRB.OPTIMAL:
        print("Optimal solution found!")
        for h in range(H):
            total_gen = sum(g[i, h, b].X for i in generators for b in range(C))
            print(f"Hour {h}: Generation = {total_gen:.2f}, Demand = {sum(d[h, c].X for c in range(C)):.2f}")
            for i in generators:
                if u[i, h].X > 0.5:
                    gen_output = sum(g[i, h, b].X for b in range(C))
                    print(f"  Gen {i}: {gen_output:.2f} MW")
    else:
        print("No optimal solution found.")


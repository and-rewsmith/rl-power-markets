import xpress as xp
import numpy as np
import torch

num_hours = 24
H = range(num_hours)
num_blocks = 5

# Define the baseline demand curve (higher during peak hours, lower off-peak)
# Increased demand to ensure all generators need to turn on
base_demand_profile = [
    20000, 19500, 19000, 18500, 18000, 17500, 18000, 20000, 22000, 24000, 25000, 26000,
    27000, 26500, 26000, 25500, 25000, 24500, 24000, 23500, 23000, 22000, 21000, 20500
]

# Define elasticities (how much price drops for each additional block)
base_price = 120  # £/MWh for first block
price_step = 30   # Drop per block
num_blocks = 3

generators = {
    1: {
        "a": 18431.0, "b": 5.5, "c": 0.0002, "CSU": 4000000.0, "CSD": 800000.0, "g_min": 3292, "g_max": 6584, "RU": 1317, "RD": 1317, "UT": 24, "DT": 24, "u0": 1, "g0": 5268
        },
    2: {
        "a": 17005.0, "b": 30.0, "c": 0.0007, "CSU": 325000.0, "CSD": 28500.0, "g_min": 2880, "g_max": 5760, "RU": 1152, "RD": 1152, "UT": 20, "DT": 20, "u0": 1, "g0": 4608
    },
    3: {
        "a": 13755.0, "b": 35.0, "c": 0.0010, "CSU": 142500.0, "CSD": 18500.0, "g_min": 1512, "g_max": 3781, "RU": 1512, "RD": 1512, "UT": 16, "DT": 16, "u0": 1, "g0": 3025
    },
    4: {
        "a": 9930.0, "b": 60.0, "c": 0.0064, "CSU": 72000.0, "CSD": 14400.0, "g_min": 667, "g_max": 3335, "RU": 1334, "RD": 1334, "UT": 10, "DT": 10, "u0": 1, "g0": 2668
    },
    5: {
        "a": 9900.0, "b": 80.0, "c": 0.0070, "CSU": 55000.0, "CSD": 12000.0, "g_min": 650, "g_max": 3252, "RU": 1951, "RD": 1951, "UT": 8, "DT": 8, "u0": 1, "g0": 2602
    },
    6: {
        "a": 8570.0, "b": 95.0, "c": 0.0082, "CSU": 31000.0, "CSD": 10000.0, "g_min": 288, "g_max": 2880, "RU": 1728, "RD": 1728, "UT": 5, "DT": 5, "u0": 0, "g0": 0
    },
    7: {
        "a": 7530.0, "b": 100.0, "c": 0.0098, "CSU": 11200.0, "CSD": 8400.0, "g_min": 275, "g_max": 2748, "RU": 2198, "RD": 2198, "UT": 4, "DT": 4, "u0": 0, "g0": 0
    }
}

# Strategic bidding parameters (k=1 for competitive behavior)
k = {i: {h: 1 for h in H} for i in generators}

# Set strategic generator to 5
strategic_gen = 5
k[strategic_gen] = {h: 1 for h in H}  # In RL setting, the agent would choose these k values strategically

# # Example demand blocks for 24 hours
# demand_blocks = {
#     h: [
#         {"lambdaD": 120, "d_max": 15000}, # pay 120 £/MWh for first 15000 MW
#         {"lambdaD": 80,  "d_max": 10000}, # pay 80 £/MWh for next 10000 MW
#         {"lambdaD": 40,  "d_max": 5000},  # pay 40 £/MWh for next 5000 MW
#     ]
#     for h in H
# }

# assert len(demand_blocks) == num_hours

# Generate blocks for each hour
demand_blocks = {}

for h in H:
    base_demand = base_demand_profile[h]
    blocks = []
    
    # Create demand blocks with higher willingness to pay
    # This ensures the market needs to turn on more expensive generators
    for b in range(num_blocks):
        # Adjusted to create larger blocks that will require more generators
        max_demand = base_demand * (0.7 / (b + 1))
        marginal_price = base_price - b * price_step
        
        blocks.append({
            "lambdaD": marginal_price,
            "d_max": max_demand
        })
    
    demand_blocks[h] = blocks

model = xp.problem('full market')

# Generator block variables and parameters
g_blocks = {}
lambdaG = {}
u = {}
su = {}
sd = {}

for i in generators:
    g_min = generators[i]["g_min"]
    g_max = generators[i]["g_max"]
    step = (g_max - g_min) / num_blocks
    
    # Marginal cost for each block (derivative of quadratic cost)
    lambdaG[i] = [
        generators[i]["b"] + 2 * generators[i]["c"] * (g_min + (b + 0.5) * step)
        for b in range(num_blocks)
    ]
    
    # Block variables (output per block)
    g_blocks[i] = {
        h: [
            model.addVariable(
                lb=0, 
                ub=step,  # Each block represents a segment of the output range
                name=f'g_{i}_{h}_block_{b}'
            )
            for b in range(num_blocks)
        ]
        for h in H
    }
    
    # Commitment variables
    u[i] = {h: model.addVariable(vartype=xp.binary, name=f'u_{i}_{h}') for h in H}
    su[i] = {h: model.addVariable(vartype=xp.binary, name=f'su_{i}_{h}') for h in H}
    sd[i] = {h: model.addVariable(vartype=xp.binary, name=f'sd_{i}_{h}') for h in H}
    
    # Minimum output constraint if generator is on
    for h in H:
        # Sum of all blocks must be at least g_min if generator is on
        model.addConstraint(
            xp.Sum(g_blocks[i][h][b] for b in range(num_blocks)) >= g_min * u[i][h]
        )
        
        # Sum of all blocks must be at most g_max - g_min if generator is on
        model.addConstraint(
            xp.Sum(g_blocks[i][h][b] for b in range(num_blocks)) <= (g_max - g_min) * u[i][h]
        )
        
        # Startup and shutdown constraints
        if h > 0:
            model.addConstraint(u[i][h] - u[i][h-1] <= su[i][h])
            model.addConstraint(u[i][h-1] - u[i][h] <= sd[i][h])
        else:
            # For the first hour, compare with initial state
            model.addConstraint(u[i][h] - generators[i]["u0"] <= su[i][h])
            model.addConstraint(generators[i]["u0"] - u[i][h] <= sd[i][h])

assert len(g_blocks) == len(generators)
assert len(g_blocks[1]) == num_hours
assert len(g_blocks[1][0]) == num_blocks

# Demand variables (one per block)
d = {
    h: [
        model.addVariable(
            lb=0, 
            ub=demand_blocks[h][c]["d_max"]
        )
        for c in range(len(demand_blocks[h]))
    ]
    for h in H
}

assert len(d) == num_hours
assert len(d[0]) == len(demand_blocks[0])

# Total demand per hour = sum of blocks
total_demand = {
    h: xp.Sum(d[h][c] for c in range(len(demand_blocks[h])))
    for h in H
}

assert len(total_demand) == num_hours

# Power balance: sum of all blocks for all generators = demand
for h in H:
    model.addConstraint(
        xp.Sum(xp.Sum(g_blocks[i][h][b] for b in range(num_blocks)) for i in generators) == total_demand[h]
    )

# Objective: Minimize total generation cost minus demand utility
objective = (
    # Generation costs: variable + no-load + startup/shutdown
    xp.Sum(
        xp.Sum(
            lambdaG[i][b] * g_blocks[i][h][b] * k[i][h] # Variable cost
            for b in range(num_blocks)
        ) + generators[i]["a"] * u[i][h]        # No-load cost
        for i in generators for h in H
    ) + xp.Sum(
        generators[i]["CSU"] * su[i][h] + generators[i]["CSD"] * sd[i][h]  # Startup/shutdown
        for i in generators for h in H
    ) 
    # Demand utility (subtract from cost)
    - xp.Sum(
        demand_blocks[h][c]["lambdaD"] * d[h][c]
        for h in H for c in range(len(demand_blocks[h]))
    )
)

model.setObjective(objective, sense=xp.minimize)
model.solve()

# Print results
print("Objective value:", model.getObjVal())
for h in H:
    print(f"\nHour {h}:")
    for i in generators:
        total_gen = sum(model.getSolution(g_blocks[i][h][b]) for b in range(num_blocks))
        status = "ON" if model.getSolution(u[i][h]) > 0.5 else "OFF"
        startup = "STARTUP" if model.getSolution(su[i][h]) > 0.5 else ""
        shutdown = "SHUTDOWN" if model.getSolution(sd[i][h]) > 0.5 else ""
        print(f"Generator {i}: {total_gen:.1f} MW, Status: {status} {startup} {shutdown}")
    for c in range(len(demand_blocks[h])):
        print(f"Demand block {c}: {model.getSolution(d[h][c]):.1f} MW")

# Store binary solutions from MIP for use in the LP model
print("\nStoring binary variable solutions...")
binary_solutions = {}
for i in generators:
    for h in H:
        binary_solutions[(i, h, 'u')] = model.getSolution(u[i][h])
        binary_solutions[(i, h, 'su')] = model.getSolution(su[i][h])
        binary_solutions[(i, h, 'sd')] = model.getSolution(sd[i][h])

# Create LP model with fixed binary variables for price determination
print("\nCreating new continuous model...")
cont_model = xp.problem('continuous market')

# Variables for continuous model
cont_g_blocks = {}
cont_u = {}
cont_su = {}
cont_sd = {}

for i in generators:
    g_min = generators[i]["g_min"]
    g_max = generators[i]["g_max"]
    step = (g_max - g_min) / num_blocks
    
    # Block variables (output per block)
    cont_g_blocks[i] = {
        h: [
            cont_model.addVariable(
                lb=0, 
                ub=step,
                name=f'g_{i}_{h}_block_{b}'
            )
            for b in range(num_blocks)
        ]
        for h in H
    }
    
    # Fixed binary variables (as continuous with fixed bounds)
    cont_u[i] = {h: cont_model.addVariable(
        lb=binary_solutions[(i, h, 'u')], 
        ub=binary_solutions[(i, h, 'u')], 
        name=f'u_{i}_{h}'
    ) for h in H}
    
    cont_su[i] = {h: cont_model.addVariable(
        lb=binary_solutions[(i, h, 'su')], 
        ub=binary_solutions[(i, h, 'su')], 
        name=f'su_{i}_{h}'
    ) for h in H}
    
    cont_sd[i] = {h: cont_model.addVariable(
        lb=binary_solutions[(i, h, 'sd')], 
        ub=binary_solutions[(i, h, 'sd')], 
        name=f'sd_{i}_{h}'
    ) for h in H}
    
    # Minimum output constraint if generator is on
    for h in H:
        # Sum of all blocks must be at least g_min if generator is on
        cont_model.addConstraint(
            xp.Sum(cont_g_blocks[i][h][b] for b in range(num_blocks)) >= g_min * cont_u[i][h]
        )
        
        # Sum of all blocks must be at most g_max - g_min if generator is on
        cont_model.addConstraint(
            xp.Sum(cont_g_blocks[i][h][b] for b in range(num_blocks)) <= (g_max - g_min) * cont_u[i][h]
        )

# Demand variables for continuous model
cont_d = {
    h: [
        cont_model.addVariable(
            lb=0, 
            ub=demand_blocks[h][c]["d_max"]
        )
        for c in range(len(demand_blocks[h]))
    ]
    for h in H
}

# Total demand per hour = sum of blocks
cont_total_demand = {
    h: xp.Sum(cont_d[h][c] for c in range(len(demand_blocks[h])))
    for h in H
}

# Power balance constraints for continuous model
cont_power_balance = {}
for h in H:
    cont_power_balance[h] = cont_model.addConstraint(
        xp.Sum(
            xp.Sum(cont_g_blocks[i][h][b] for b in range(num_blocks)) 
            for i in generators
        ) == cont_total_demand[h]
    )

# Objective: Minimize total generation cost minus demand utility
cont_objective = (
    # Generation costs: variable + no-load + startup/shutdown
    xp.Sum(
        xp.Sum(
            lambdaG[i][b] * cont_g_blocks[i][h][b]  # Variable cost
            for b in range(num_blocks)
        ) + generators[i]["a"] * cont_u[i][h]        # No-load cost
        for i in generators for h in H
    ) + xp.Sum(
        generators[i]["CSU"] * cont_su[i][h] + generators[i]["CSD"] * cont_sd[i][h]  # Startup/shutdown
        for i in generators for h in H
    ) 
    # Demand utility (subtract from cost)
    - xp.Sum(
        demand_blocks[h][c]["lambdaD"] * cont_d[h][c]
        for h in H for c in range(len(demand_blocks[h]))
    )
)

cont_model.setObjective(cont_objective, sense=xp.minimize)

# Solve LP with fixed binary variables to get dual values
print("Solving continuous problem...")
cont_model.solve()

# Print LP results
print("\nLP Results:")
print("Objective value:", cont_model.getObjVal())

# Calculate market prices from dual values or marginal generator costs
print("\nMarket Prices:")
market_prices = {}
for h in H:
    # Try using the row number
    dual = cont_model.getDuals(h)
    
    # If dual is zero, use marginal generator approach
    if abs(dual) < 0.001:
        # Use marginal generator approach
        max_cost = 0
        for i in generators:
            for b in range(num_blocks):
                if cont_model.getSolution(cont_g_blocks[i][h][b]) > 0.001:  # Block is producing
                    if lambdaG[i][b] > max_cost:
                        max_cost = lambdaG[i][b]
        
        market_prices[h] = max_cost
        print(f"Hour {h}: {max_cost:.2f} £/MWh (using marginal generator cost)")
    else:
        market_prices[h] = dual
        print(f"Hour {h}: {dual:.2f} £/MWh (using row number)")

# Calculate profit for strategic generator (Generator 5)
strategic_gen = 5
profit = 0
for h in H:
    # Revenue (market price * quantity)
    quantity = sum(model.getSolution(g_blocks[strategic_gen][h][b]) for b in range(num_blocks))
    revenue = market_prices[h] * quantity
    
    # Costs - use true costs (without k) for profit calculation
    # The k factor only affects the bid, not the actual cost to the generator
    gen_cost = sum(lambdaG[strategic_gen][b] * model.getSolution(g_blocks[strategic_gen][h][b]) 
                  for b in range(num_blocks))
    no_load_cost = generators[strategic_gen]["a"] * model.getSolution(u[strategic_gen][h])
    startup_cost = generators[strategic_gen]["CSU"] * model.getSolution(su[strategic_gen][h])
    shutdown_cost = generators[strategic_gen]["CSD"] * model.getSolution(sd[strategic_gen][h])
    
    total_cost = gen_cost + no_load_cost + startup_cost + shutdown_cost
    
    # Profit = Revenue - Cost
    profit += revenue - total_cost

print(f"\nGenerator 5's profit: {profit:.2f} £")
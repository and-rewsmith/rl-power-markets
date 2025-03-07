import xpress as xp
import numpy as np
import torch

# Initialize Xpress with community license
xp.init('/Users/taylorkergan/Documents/repos/rl-power-markets/env/lib/python3.11/site-packages/xpress/license/community-xpauth.xpr')

# Market parameters
num_hours = 24
H = range(num_hours)
num_blocks = 5  # Number of blocks for both generation and demand

# Define the baseline demand curve (higher during peak hours, lower off-peak)
base_demand_profile = [
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
]

# Generator parameters dict structure:
# a: No-load cost (C^NL)
# b, c: Variable cost coefficients
# CSU: Startup cost (C^U)
# CSD: Shutdown cost (C^D)
# g_min, g_max: Generation limits
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

def solve_market(k_values=None, strategic_gen=5, base_price=120, price_step=30):
    model = xp.problem('lower_level_market')
    model.setControl({'outputlog': 0, 'lplog': 0, 'miplog': 0})
    
    # Strategic bidding parameters (k=1 for competitive behavior)
    k = {i: {h: 1.0 for h in H} for i in generators}
    k[strategic_gen].update({h: k_values for h in H})

    # Generation variables and parameters
    g = {}  # g[i][h][b]: generation of block b of generator i at hour h
    lambda_g = {}  # λ^G[i][b]: marginal cost of block b for generator i
    u = {}  # u[i][h]: commitment status
    su = {}  # su[i][h]: startup indicator
    sd = {}  # sd[i][h]: shutdown indicator

    for i in generators:
        g_min = generators[i]["g_min"]
        g_max = generators[i]["g_max"]
        step = (g_max - g_min) / num_blocks
        
        lambda_g[i] = [generators[i]["b"] + 2 * generators[i]["c"] * (g_min + (b + 0.5) * step) for b in range(num_blocks)]
        
        g[i] = {h: [model.addVariable(lb=0, ub=step, name=f'g_{i}_{h}_{b}') for b in range(num_blocks)] for h in H}
        u[i] = {h: model.addVariable(vartype=xp.binary, name=f'u_{i}_{h}') for h in H}
        su[i] = {h: model.addVariable(vartype=xp.binary, name=f'su_{i}_{h}') for h in H}
        sd[i] = {h: model.addVariable(vartype=xp.binary, name=f'sd_{i}_{h}') for h in H}
        
        for h in H:
            model.addConstraint(xp.Sum(g[i][h][b] for b in range(num_blocks)) >= g_min * u[i][h])
            model.addConstraint(xp.Sum(g[i][h][b] for b in range(num_blocks)) <= (g_max - g_min) * u[i][h])
            
            if h > 0:
                model.addConstraint(u[i][h] - u[i][h-1] <= su[i][h])
                model.addConstraint(u[i][h-1] - u[i][h] <= sd[i][h])
            else:
                model.addConstraint(u[i][h] - generators[i]["u0"] <= su[i][h])
                model.addConstraint(generators[i]["u0"] - u[i][h] <= sd[i][h])

    # Demand variables and parameters
    d = {}  # d[h][c]: demand of block c at hour h
    lambda_d = {}  # λ^D[h][c]: willingness to pay for block c at hour h
    
    for h in H:
        base_demand = base_demand_profile[h]
        blocks = []
        for b in range(num_blocks):
            max_demand = base_demand * (0.7 / (b + 1))
            marginal_price = base_price - b * price_step
            blocks.append({"lambda_d": marginal_price, "d_max": max_demand})
        
        lambda_d[h] = blocks
        d[h] = [model.addVariable(lb=0, ub=blocks[c]["d_max"]) for c in range(len(blocks))]

    # Power balance constraint (4)
    for h in H:
        model.addConstraint(
            xp.Sum(xp.Sum(g[i][h][b] for b in range(num_blocks)) for i in generators) == 
            xp.Sum(d[h][c] for c in range(len(lambda_d[h])))
        )

    # Objective (3): min generation costs - demand utility
    objective = (
        xp.Sum(xp.Sum(lambda_g[i][b] * g[i][h][b] * k[i][h] for b in range(num_blocks)) + 
               generators[i]["a"] * u[i][h] for i in generators for h in H) +
        xp.Sum(generators[i]["CSU"] * su[i][h] + generators[i]["CSD"] * sd[i][h] 
               for i in generators for h in H) -
        xp.Sum(lambda_d[h][c]["lambda_d"] * d[h][c] 
               for h in H for c in range(len(lambda_d[h])))
    )

    model.setObjective(objective, sense=xp.minimize)
    return model, g, u, su, sd, d, lambda_d, lambda_g

def solve_continuous_model(model, g, u, su, sd, d, lambda_d, lambda_g):
    cont_model = xp.problem('continuous market')
    cont_model.setControl({'outputlog': 0, 'lplog': 0, 'miplog': 0})

    # Get binary solutions
    binary_solutions = {}
    for i in generators:
        for h in H:
            binary_solutions[(i, h, 'u')] = model.getSolution(u[i][h])
            binary_solutions[(i, h, 'su')] = model.getSolution(su[i][h])
            binary_solutions[(i, h, 'sd')] = model.getSolution(sd[i][h])

    # Create continuous variables with fixed binary values
    cont_g = {}
    cont_u = {}
    cont_d = {}

    for i in generators:
        g_min = generators[i]["g_min"]
        g_max = generators[i]["g_max"]
        step = (g_max - g_min) / num_blocks
        
        cont_g[i] = {
            h: [cont_model.addVariable(lb=0, ub=step, name=f'g_{i}_{h}_{b}')
                for b in range(num_blocks)]
            for h in H
        }
        
        cont_u[i] = {
            h: cont_model.addVariable(
                lb=binary_solutions[(i, h, 'u')], 
                ub=binary_solutions[(i, h, 'u')], 
                name=f'u_{i}_{h}'
            ) for h in H
        }
        
        for h in H:
            cont_model.addConstraint(
                xp.Sum(cont_g[i][h][b] for b in range(num_blocks)) >= g_min * cont_u[i][h]
            )
            cont_model.addConstraint(
                xp.Sum(cont_g[i][h][b] for b in range(num_blocks)) <= (g_max - g_min) * cont_u[i][h]
            )

    # Fix demand values from MIP solution
    for h in H:
        cont_d[h] = [
            cont_model.addVariable(
                lb=model.getSolution(d[h][c]),
                ub=model.getSolution(d[h][c]),
                name=f'd_{h}_{c}'
            )
            for c in range(len(lambda_d[h]))
        ]

    # Power balance constraints
    power_balance = {}
    for h in H:
        power_balance[h] = cont_model.addConstraint(
            xp.Sum(xp.Sum(cont_g[i][h][b] for b in range(num_blocks)) for i in generators) == 
            xp.Sum(cont_d[h][c] for c in range(len(lambda_d[h])))
        )

    # Objective: min generation costs - demand utility (without strategic bidding)
    cont_objective = (
        xp.Sum(
            xp.Sum(lambda_g[i][b] * cont_g[i][h][b] for b in range(num_blocks)) + 
            generators[i]["a"] * cont_u[i][h] for i in generators for h in H
        ) - xp.Sum(
            lambda_d[h][c]["lambda_d"] * cont_d[h][c]
            for h in H for c in range(len(lambda_d[h]))
        )
    )

    cont_model.setObjective(cont_objective, sense=xp.minimize)
    cont_model.solve()
    
    return cont_model, power_balance

def calculate_market_prices(model, cont_model, power_balance, g):
    market_prices = {}
    for h in H:
        try:
            dual = cont_model.getDuals(power_balance[h])
            if abs(dual) > 0.001:
                market_prices[h] = dual
                continue
        except:
            pass
        
        # Use marginal generator approach if duals not available
        max_cost = 0
        for i in generators:
            g_min = generators[i]["g_min"]
            g_max = generators[i]["g_max"]
            step = (g_max - g_min) / num_blocks
            
            for b in range(num_blocks):
                if model.getSolution(g[i][h][b]) > 0.001:
                    cost = generators[i]["b"] + 2 * generators[i]["c"] * (
                        g_min + (b + 0.5) * step
                    )
                    max_cost = max(max_cost, cost)
        market_prices[h] = max_cost
    
    return market_prices

def calculate_generator_profit(model, market_prices, g, u, su, sd, generator_id):
    profit = 0
    for h in H:
        quantity = sum(model.getSolution(g[generator_id][h][b]) for b in range(num_blocks))
        revenue = market_prices[h] * quantity
        
        gen_cost = sum(
            (generators[generator_id]["b"] + 
             2 * generators[generator_id]["c"] * (
                 generators[generator_id]["g_min"] + (b + 0.5) * (
                     generators[generator_id]["g_max"] - 
                     generators[generator_id]["g_min"]
                 ) / num_blocks
             )) * model.getSolution(g[generator_id][h][b])
            for b in range(num_blocks)
        )
        
        no_load_cost = generators[generator_id]["a"] * model.getSolution(u[generator_id][h])
        startup_cost = generators[generator_id]["CSU"] * model.getSolution(su[generator_id][h])
        shutdown_cost = generators[generator_id]["CSD"] * model.getSolution(sd[generator_id][h])
        
        profit += revenue - (gen_cost + no_load_cost + startup_cost + shutdown_cost)
    
    return profit

def test_strategic_bidding(
    k_values=None,
    demand_multiplier=1.0,
    peak_multiplier=1.2,
    strategic_gen_id=5,
    base_price=120,
    price_step=30,
    analyze_peak_hours=False,
    h=24
):
    print("\nTesting strategic bidding with parameters:")
    print(f"Demand multiplier: {demand_multiplier}")
    print(f"Peak multiplier: {peak_multiplier}")
    print(f"Strategic generator: {strategic_gen_id}")
    print(f"Base price: {base_price}")
    print(f"Price step: {price_step}")
    print(f"Number of blocks: {num_blocks}")
    
    H = range(h)
    if k_values is None:
        k_values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    # Modify demand profile based on multipliers
    modified_demand = []
    peak_hours = range(10, 22)  # 10:00 to 22:00
    
    for h in range(24):
        base = base_demand_profile[h] * demand_multiplier
        if h in peak_hours:
            base *= peak_multiplier
        modified_demand.append(base)
    
    results = {}
    hourly_data = {k: {} for k in k_values}

    for k_test in k_values:
        print(f"\nTesting k = {k_test}")
        
        # Solve lower level problem
        model, g, u, su, sd, d, lambda_d, lambda_g = solve_market(
            k_test, 
            strategic_gen_id,
            base_price=base_price,
            price_step=price_step
        )
        model.solve()

        # Solve continuous model for price determination
        cont_model, power_balance = solve_continuous_model(model, g, u, su, sd, d, lambda_d, lambda_g)
        
        # Calculate market prices
        market_prices = calculate_market_prices(model, cont_model, power_balance, g)
        
        # Calculate profits and statistics
        profit = calculate_generator_profit(model, market_prices, g, u, su, sd, strategic_gen_id)
        
        # Calculate detailed results
        total_quantity = 0
        peak_profit = 0
        offpeak_profit = 0
        peak_quantity = 0
        offpeak_quantity = 0
        peak_hours_committed = 0
        offpeak_hours_committed = 0
        hourly_data[k_test] = {}

        for h in H:
            quantity = sum(model.getSolution(g[strategic_gen_id][h][b]) for b in range(num_blocks))
            revenue = market_prices[h] * quantity
            
            gen_cost = sum(
                (generators[strategic_gen_id]["b"] + 
                 2 * generators[strategic_gen_id]["c"] * (
                     generators[strategic_gen_id]["g_min"] + (b + 0.5) * (
                         generators[strategic_gen_id]["g_max"] - 
                         generators[strategic_gen_id]["g_min"]
                     ) / num_blocks
                 )) * model.getSolution(g[strategic_gen_id][h][b])
                for b in range(num_blocks)
            )
            
            no_load_cost = generators[strategic_gen_id]["a"] * model.getSolution(u[strategic_gen_id][h])
            startup_cost = generators[strategic_gen_id]["CSU"] * model.getSolution(su[strategic_gen_id][h])
            shutdown_cost = generators[strategic_gen_id]["CSD"] * model.getSolution(sd[strategic_gen_id][h])
            
            hour_profit = revenue - (gen_cost + no_load_cost + startup_cost + shutdown_cost)

            # Store hourly data
            hourly_data[k_test][h] = {
                'quantity': quantity,
                'price': market_prices[h],
                'revenue': revenue,
                'total_cost': gen_cost + no_load_cost + startup_cost + shutdown_cost,
                'profit': hour_profit,
                'committed': model.getSolution(u[strategic_gen_id][h]) > 0.5
            }

            # Separate peak/off-peak statistics
            if h in peak_hours:
                peak_profit += hour_profit
                if quantity > 0:
                    peak_quantity += quantity
                    peak_hours_committed += 1
            else:
                offpeak_profit += hour_profit
                if quantity > 0:
                    offpeak_quantity += quantity
                    offpeak_hours_committed += 1

        results[k_test] = {
            'total_profit': profit,
            'peak_profit': peak_profit,
            'offpeak_profit': offpeak_profit,
            'avg_peak_quantity': peak_quantity / peak_hours_committed if peak_hours_committed > 0 else 0,
            'avg_offpeak_quantity': offpeak_quantity / offpeak_hours_committed if offpeak_hours_committed > 0 else 0,
            'peak_hours_committed': peak_hours_committed,
            'offpeak_hours_committed': offpeak_hours_committed
        }

    # Print summary results
    print("\nStrategic Bidding Results Summary:")
    print("-" * 100)
    print("k-value | Total Profit (£) | Peak Profit (£) | Off-Peak Profit (£) | Peak Hours | Off-Peak Hours")
    print("-" * 100)
    for k_test in k_values:
        r = results[k_test]
        print(f"{k_test:7.1f} | {r['total_profit']:14.2f} | {r['peak_profit']:13.2f} | {r['offpeak_profit']:16.2f} | {r['peak_hours_committed']:10d} | {r['offpeak_hours_committed']:14d}")

    # Print detailed hourly analysis if requested
    if analyze_peak_hours:
        print("\nDetailed Peak Hours Analysis (10:00-22:00):")
        print("-" * 80)
        print("Hour | ", end="")
        for k in k_values:
            print(f"k={k:.1f} Profit (£) | ", end="")
        print("\n" + "-" * 80)
        
        for h in peak_hours:
            print(f"{h:4d} | ", end="")
            for k in k_values:
                data = hourly_data[k][h]
                print(f"{data['profit']:14.2f} | ", end="")
            print()

    return results, hourly_data

if __name__ == "__main__":
    # Test different market competitiveness scenarios
    
    print("\nScenario 1: High Demand with High Willingness to Pay") # looks good for bidding
    test_strategic_bidding(
        demand_multiplier=1.4,    # Significantly higher demand
        peak_multiplier=1.5,      # Stronger peak demand
        base_price=180,           # Higher willingness to pay
        price_step=20,            # Smaller price drops between blocks
        k_values=[1.0, 1.05, 1.1, 1.15, 1.2],  # Test smaller k increments
        analyze_peak_hours=False
    )
    
    print("\nScenario 2: Extreme Peak Demand") # looks good for bidding
    test_strategic_bidding(
        demand_multiplier=1.2,
        peak_multiplier=2.5,      # Very high peak multiplier
        base_price=150,
        price_step=25,
        k_values=[1.0, 1.1, 1.2, 1.3],
        analyze_peak_hours=False
    )
    
    print("\nScenario 3: Scarcity with High Prices")
    test_strategic_bidding(
        demand_multiplier=1.3,
        peak_multiplier=1.4,
        base_price=200,           # Very high base price
        price_step=15,            # Small price steps
        k_values=[1.0, 1.1, 1.2],
        analyze_peak_hours=False
    )

    print("\nScenario 4: Extreme Scarcity")
    test_strategic_bidding(
        demand_multiplier=1.6,     # Very high base demand
        peak_multiplier=2.0,       # Double demand during peaks
        base_price=250,            # Very high base price
        price_step=10,             # Minimal price drops
        k_values=[1.0, 1.02, 1.05, 1.07, 1.1],  # Very fine k increments
        analyze_peak_hours=False
    )
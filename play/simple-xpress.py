import xpress as xp

# Simple model with just a few generators and hours
num_hours = 3
H = range(num_hours)

# Simplified generator data
generators = {
    1: {"g_min": 100, "g_max": 500, "cost": 50},
    2: {"g_min": 50, "g_max": 300, "cost": 70},
    3: {"g_min": 30, "g_max": 200, "cost": 90}
}

# Create model
model = xp.problem('test market')

# Variables
g = {}  # Generation
u = {}  # Unit commitment (binary)

for i in generators:
    g[i] = {h: model.addVariable(lb=0, ub=generators[i]["g_max"], name=f'g_{i}_{h}') for h in H}
    u[i] = {h: model.addVariable(vartype=xp.binary, name=f'u_{i}_{h}') for h in H}

# Demand for each hour
demand = {0: 400, 1: 600, 2: 500}

# Constraints
power_balance = {}
for h in H:
    # Power balance: sum of generation = demand
    power_balance[h] = model.addConstraint(
        xp.Sum(g[i][h] for i in generators) == demand[h]
    )
    
    # Min/max generation for each generator
    for i in generators:
        # Min generation if unit is on
        model.addConstraint(
            g[i][h] >= generators[i]["g_min"] * u[i][h]
        )
        
        # Max generation if unit is on
        model.addConstraint(
            g[i][h] <= generators[i]["g_max"] * u[i][h]
        )

# Objective: minimize cost
objective = xp.Sum(
    generators[i]["cost"] * g[i][h]
    for i in generators for h in H
)

model.setObjective(objective, sense=xp.minimize)

# Step 1: Solve the MIP
print("Solving MIP...")
model.solve()

# Print MIP results
print("\nMIP Results:")
try:
    print("Objective value:", model.attributes.objval)  # Use attributes.objval instead of getObjVal
except:
    print("Objective value:", model.getObjVal())  # Fallback to getObjVal if needed

# Store the binary variable solutions
binary_solutions = {}
g_solutions = {}

# Print results and store solutions
for h in H:
    print(f"\nHour {h} (Demand: {demand[h]} MW):")
    for i in generators:
        try:
            # Get and store solutions
            u_val = model.getSolution(u[i][h])
            g_val = model.getSolution(g[i][h])
            
            # Store for later use
            binary_solutions[(i, h)] = u_val
            g_solutions[(i, h)] = g_val
            
            # Print results
            status = "ON" if u_val > 0.5 else "OFF"
            print(f"Generator {i}: {g_val:.1f} MW, Status: {status}")
        except Exception as e:
            print(f"Error getting solution for generator {i}, hour {h}: {e}")
            # Default values if error
            binary_solutions[(i, h)] = 0
            g_solutions[(i, h)] = 0

# Step 2: Create a new continuous model with fixed binary variables
print("\nCreating new continuous model...")
cont_model = xp.problem('continuous market')

# Variables for continuous model
cont_g = {}  # Generation
cont_u = {}  # Fixed commitment (continuous)

for i in generators:
    cont_g[i] = {h: cont_model.addVariable(lb=0, ub=generators[i]["g_max"], name=f'g_{i}_{h}') for h in H}
    # Create continuous variables fixed to binary values
    cont_u[i] = {h: cont_model.addVariable(
        lb=binary_solutions[(i, h)], 
        ub=binary_solutions[(i, h)], 
        name=f'u_{i}_{h}'
    ) for h in H}

# Constraints for continuous model
cont_power_balance = {}
for h in H:
    # Power balance: sum of generation = demand
    cont_power_balance[h] = cont_model.addConstraint(
        xp.Sum(cont_g[i][h] for i in generators) == demand[h]
    )
    
    # Min/max generation for each generator
    for i in generators:
        # Min generation if unit is on
        cont_model.addConstraint(
            cont_g[i][h] >= generators[i]["g_min"] * cont_u[i][h]
        )
        
        # Max generation if unit is on
        cont_model.addConstraint(
            cont_g[i][h] <= generators[i]["g_max"] * cont_u[i][h]
        )

# Objective: minimize cost
cont_objective = xp.Sum(
    generators[i]["cost"] * cont_g[i][h]
    for i in generators for h in H
)

cont_model.setObjective(cont_objective, sense=xp.minimize)

# Step 3: Solve the continuous problem
print("Solving continuous problem...")
cont_model.solve()

# Print LP results and dual values
print("\nLP Results:")
try:
    print("Objective value:", cont_model.attributes.objval)
except:
    print("Objective value:", cont_model.getObjVal())

print("\nDual Values (Market Prices):")
market_prices = {}
for h in H:
    try:
        # Try using the constraint index
        constraint_index = cont_power_balance[h].getIndex()
        dual = cont_model.getDual(constraint_index)
        market_prices[h] = dual
        print(f"Hour {h}: {dual:.2f} (using constraint index)")
    except Exception as e:
        print(f"Hour {h}: Error getting dual with index - {e}")
        try:
            # Try using the row number
            dual = cont_model.getDual(h)
            market_prices[h] = dual
            print(f"Hour {h}: {dual:.2f} (using row number)")
        except Exception as e2:
            print(f"Hour {h}: Error with row number - {e2}")
            
            # If all else fails, use the marginal generator approach
            max_cost = 0
            for i in generators:
                if cont_model.getSolution(cont_g[i][h]) > 0.001:  # Generator is producing
                    if generators[i]["cost"] > max_cost:
                        max_cost = generators[i]["cost"]
            
            market_prices[h] = max_cost
            print(f"Hour {h}: {max_cost:.2f} (using marginal generator cost)")

# Print the solution for verification
print("\nContinuous Solution:")
for h in H:
    print(f"\nHour {h} (Demand: {demand[h]} MW):")
    for i in generators:
        try:
            gen = cont_model.getSolution(cont_g[i][h])
            status = "ON" if binary_solutions[(i, h)] > 0.5 else "OFF"
            print(f"Generator {i}: {gen:.1f} MW, Status: {status}, Cost: {generators[i]['cost']}")
        except Exception as e:
            print(f"Error getting solution for generator {i}, hour {h}: {e}")
            print(f"Using original solution: {g_solutions[(i, h)]:.1f} MW")
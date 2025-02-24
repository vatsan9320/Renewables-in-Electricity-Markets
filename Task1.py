from pyomo.environ import *
import all_data
import numpy as np

#import data
data=all_data.get_data()
Pmax_D=np.max(data["load"]["System demand (MW)"])
#P_max_wind=[np.mean(values) for values in data["wind_farm"].values()]
wind_farm_capacity=200

U_d=20 #Demand bid price


#create a model
model = ConcreteModel()

# Sets
model.init_conv_G = Set(initialize=[e+1 for e in range(12)])
model.init_wind_farm = Set(initialize=[e+1 for e in range(6)])


#Declare variables
model.p_demand= Var(within=NonNegativeReals, bounds=(0, Pmax_D), initialize=Pmax_D/2)
model.p_conv_G = Var(model.init_conv_G, within=NonNegativeReals, initialize=0)
model.p_wind_farm = Var(model.init_wind_farm, within=NonNegativeReals, initialize=0)

#Parameters
model.Pmax_convG=Param(model.init_conv_G, initialize={i+1: pmax for i, pmax in enumerate(data["generation_unit"]["Pmax (MW)"])})
model.Pmin_convG=Param(model.init_conv_G, initialize={i+1: pmin for i, pmin in enumerate(data["generation_unit"]["Pmin (MW)"])})
model.Pini_convG=Param(model.init_conv_G, initialize={i+1: pmax for i, pmax in enumerate(data["generation_unit"]["Pini_i"])})

model.price_conv=Param(model.init_conv_G, initialize={i+1: price for i, price in enumerate(data["generation_unit"]["Ci"])})
model.price_wind_farm=Param(model.init_wind_farm, initialize = 0)


#Objective function : Maximization of Social welfare
def objective_rule(model):
    return (U_d*model.p_demand - sum(model.price_conv[i] * model.p_conv_G[i] for i in model.init_conv_G) - sum(model.price_wind_farm[j] * model.p_wind_farm[j] for j in model.init_wind_farm))
model.social_welfare = Objective(rule=objective_rule, sense=maximize)

#constraints
def capacity_rule_conv_G(model, i):
    return model.p_conv_G[i] <= model.Pmax_convG[i]
model.capacity_conv_G_constraint = Constraint(model.init_conv_G, rule=capacity_rule_conv_G)

def capacity_rule_wind_farm(model, i):
    return model.p_wind_farm[i] <= np.mean(data["wind_farm"][f"wind_farm {i}"])*wind_farm_capacity
model.capacity_wind_farm_constraint = Constraint(model.init_wind_farm, rule=capacity_rule_wind_farm)

def balance(model):
    return model.p_demand == sum(model.p_conv_G[i] for i in model.init_conv_G) + sum(model.p_wind_farm[j] for j in model.init_wind_farm)
model.balance_constraint = Constraint(rule=balance)

# To get the dual variables
model.dual = Suffix(direction=Suffix.IMPORT)

# Create a solver 
solver = SolverFactory("gurobi", solver_io="python")  # Make sure Gurobi is installed and properly configured

# Solve the model
solution = solver.solve(model, tee=True)


model.display()

# Store the results
results = {}
results["load"] = [value(model.p_demand[key]) for key in model.p_demand]
results["conventional generators"] = [value(model.p_conv_G[key]) for key in model.p_conv_G]
results["wind farms"] = [value(model.p_wind_farm[key]) for key in model.p_wind_farm]
print(results)
print("Social Welfare =", model.social_welfare())

#print(solution)

# Extract the market-clearing price (MCP)
if solution.solver.termination_condition == TerminationCondition.optimal:
    mcp = model.dual[model.balance_constraint]
    print(f"Market-Clearing Price (MCP): {mcp:.2f}")
else:
    print("Solver did not find an optimal solution. MCP cannot be calculated.")

#Profit of each generators and Utility
profit= {}
profit["conventional generators"] = [
    prod * (mcp - price)
    for prod, price in zip(results["conventional generators"], data["generation_unit"]["Ci"])]

profit["wind farms"] = [value * mcp for value in results["wind farms"]]

utility=[value *(U_d-mcp) for value in results["load"]]

print("profit", profit, "utility", utility)
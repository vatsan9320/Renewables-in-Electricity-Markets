from pyomo.environ import *
import all_data
import numpy as np

#import data
data=all_data.get_data()
Pmax_D=np.max(data["load"]["System demand (MW)"])
P_max_wind=[np.mean(values) for values in data["wind_farm"].values()]

#autre option : [values[0] for values in dictionnaire.values()] #je récupère la production des éoliennes pour la 1ère heure de la journée

U_d=50 #Demand bid price


#create a model
model = ConcreteModel()

# Sets
# model.init_conv_G = Set(initialize=[e+1 for e in range(len(data["generation_unit"]["Unit #"]))])
# model.init_wind_farm = Set(initialize=[e+1 for e in range(len(data["wind_farm"]))])

model.init_conv_G = Set(initialize=[e+1 for e in range(12)])
model.init_wind_farm = Set(initialize=[e+1 for e in range(6)])


#Declare variables
model.p_demand= Var(within=NonNegativeReals, bounds=(0, Pmax_D), initialize=Pmax_D/2) #ça me fait la constrainte sur la demande
model.p_conv_G = Var(model.init_conv_G, within=NonNegativeReals, initialize=0)
model.p_wind_farm = Var(model.init_wind_farm, within=NonNegativeReals, initialize=0)

#Parameters
model.Pmax_convG=Param(model.init_conv_G, initialize={i+1: pmax for i, pmax in enumerate(data["generation_unit"]["Pmax (MW)"])})
model.Pmax_wind_farm=Param(model.init_wind_farm, initialize={i+1: pmax for i, pmax in enumerate(P_max_wind)})

# model.Pini_convG=Param(model.init_conv_G, initialize={i+1: pmax for i, pmax in enumerate(data["generation_unit"]["Pini_i"])})
# model.MaxRampUp=Param(model.init_conv_G, initialize={i+1: Rmax for i, Rmax in enumerate(data["generation_unit"]["RU (MW/h)"])})
# model.MaxRampDown=Param(model.init_conv_G, initialize={i+1: Rmax for i, Rmax in enumerate(data["generation_unit"]["RD (MW/h)"])})

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
    return model.p_wind_farm[i] <= model.Pmax_wind_farm[i]
model.capacity_wind_farm_constraint = Constraint(model.init_wind_farm, rule=capacity_rule_wind_farm)

def balance(model,i,j): #2 arguments car 2 model set
    return model.p_demand-sum(model.p_conv_G[i] for i in model.init_conv_G)-sum(model.p_wind_farm[j] for j in model.init_wind_farm) == 0
model.balance_constraint = Constraint(model.init_conv_G, model.init_wind_farm, rule=balance)

# def max_rampup(model,i):
#     return model.p_conv_G[i] - model.Pini_convG[i] <= model.MaxRampUp[i]
# model.maxrampup_constraint = Constraint(model.init_conv_G, rule=max_rampup)

# def max_rampdown(model,i):
#     return model.p_conv_G[i] - model.Pini_convG[i] >= model.MaxRampDown[i]
# model.maxrampdown_constraint = Constraint(model.init_conv_G, rule=max_rampdown)

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
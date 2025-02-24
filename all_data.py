import numpy as np
import pandas as pd

def get_data():
    
    generating_unit_data = {
    #Technical data of generating units 
    "Unit #": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Node": [1, 2, 7, 13, 15, 15, 16, 18, 21, 22, 23, 23],
    "Pmax (MW)": [152, 152, 350, 591, 60, 155, 155, 400, 400, 300, 300, 350],
    "Pmin (MW)": [30.4, 30.4, 75, 206.85, 12, 54.25, 54.25, 100, 100, 300, 108.5, 140],
    "R+ (MW)": [40, 40, 70, 180, 60, 30, 30, 80, 80, 0, 60, 40],
    "R- (MW)": [40, 40, 70, 180, 60, 30, 30, 80, 80, 0, 60, 40],
    "RU (MW/h)": [120, 120, 350, 240, 60, 155, 155, 280, 300, 0, 180, 240],
    "RD (MW/h)": [120, 120, 350, 240, 60, 155, 155, 280, 300, 0, 180, 240],
    "UT (h)": [8, 8, 8, 12, 4, 8, 8, 8, 0, 0, 8, 8],
    "DT (h)": [4, 4, 8, 10, 2, 8, 8, 8, 0, 0, 8, 8],

    #Costs and initial states of generating units
    "Ci": [13.32, 13.32, 20.7, 20.93, 26.11, 10.52, 6.02, 5.47, 0, 0, 10.52, 10.89],
    "Cu_i": [15, 15, 10, 8, 7, 7, 16, 0, 0, 0, 16, 16],
    "Cd_i": [14, 14, 9, 7, 5, 6, 14, 0, 0, 0, 14, 14],
    "C+_i": [15, 15, 24, 25, 28, 16, 16, 0, 0, 0, 16, 16],
    "C-_i": [11, 11, 16, 17, 23, 7, 7, 0, 0, 0, 8, 8],
    "Csu_i": [1430.4, 1430.4, 1725, 3056.7, 437, 312, 124, 0, 0, 624, 2298, 2298],
    "Pini_i": [76, 76, 0, 0, 0, 0, 312, 0, 0, 0, 240, 280],
    "Uini_i": [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
    "Tini_i": [22, 22, -2, -1, -1, -1, 10, -2, -2, -1, 24, 50]}

    #Demand data
    load_data = {
    "Hour": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    "System demand (MW)": [
        1775.835, 1669.815, 1590.3, 1563.795, 1563.795, 1590.3, 1961.37, 2279.43, 
        2517.975, 2544.48, 2544.48, 2517.975, 2517.975, 2517.975, 2464.965, 2464.965, 
        2623.995, 2650.5, 2650.5, 2544.48, 2411.955, 2199.915, 1934.865, 1669.815
    ]}
    #Wind farms data : We extract the production of the 24 first hours of each first dataset from each zone (1 zone 1 file)
    wind_farm = {}
    file_path = ["Wind farms data\scen_zone1.out", "Wind farms data\scen_zone2.out","Wind farms data\scen_zone3.out","Wind farms data\scen_zone4.out","Wind farms data\scen_zone5.out","Wind farms data\scen_zone6.out"]
    for i in range(len(file_path)):
        wind_farm[f"wind_farm {i+1}"] = pd.read_csv(file_path[i])["V1"].head(24).tolist()

    return {"generation_unit" : generating_unit_data, "load" : load_data, "wind_farm" : wind_farm}


test=get_data()

# print(np.max(test["load"]["System demand (MW)"])) #pour avoir la Demand max
# print([e+1 for e in range(len(test["generation_unit"]["Unit #"]))])
# print(len(test["generation_unit"]["Unit #"]))

print(np.mean(test["wind_farm"]["wind_farm 1"])*100)
# print([np.mean(values) for values in test["wind_farm"].values()])

offre=[152, 152, 350, 591, 60, 155, 155, 400, 400, 300, 300, 350]
for i in range(1,len(test["wind_farm"])+1):
       offre.append(100*np.mean(test["wind_farm"][f"wind_farm {i}"]))
print(offre)
    


# ####### Print Supply and Offer curves
# import matplotlib.pyplot as plt
# import numpy as np

# # Données
# demand = 2650.5  # MW
# supply_quantities = [152, 152, 350, 591, 60, 155, 155, 400, 400, 300, 300, 350, 
#                      66.02668138964893, 69.90705836674121, 69.09000109053993, 
#                      60.93791955818926, 64.88591062635679, 62.98944854324062]
# bidding_prices = [13.32, 13.32, 20.7, 20.93, 26.11, 10.52, 6.02, 5.47, 0, 0, 
#                   10.52, 10.89, 0, 0, 0, 0, 0, 0]

# # Tracer l'offre

# sorted_indices = np.argsort(bidding_prices)       # Trier par prix croissant
# print("ind",sorted_indices)
# sorted_prices = np.array(bidding_prices)[sorted_indices]
# print("pri",sorted_prices)
# sorted_supply= np.array(supply_quantities)[sorted_indices]
# sorted_cumulative_supply = np.cumsum(sorted_supply) 

# # Tracer la courbe
# plt.figure(figsize=(10, 6))
# plt.step(sorted_cumulative_supply, sorted_prices, where='post', label="Supply", color='orange')

# # Tracer la demande constante
# plt.axvline(x=demand, color='blue', linestyle='--', label="Demand (2650 MW)")

# # Ajouter des annotations et des styles
# plt.ylabel("Price")
# plt.xlabel("Quantity (MW)")
# plt.title("Supply and Demand")
# plt.legend()
# plt.grid(True)

# # Afficher le graphique
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    # Demand data from pdf file
    load_data = {
    "Hour": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    "System demand (MW)": [
        1775.835, 1669.815, 1590.3, 1563.795, 1563.795, 1590.3, 1961.37, 2279.43, 
        2517.975, 2544.48, 2544.48, 2517.975, 2517.975, 2517.975, 2464.965, 2464.965, 
        2623.995, 2650.5, 2650.5, 2544.48, 2411.955, 2199.915, 1934.865, 1669.815 ]}

    # Node Location and Distribution of the Total System Demand
    node_demand = {
    "Load #": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    "Node": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 20],
    "percentage of system load": [3.8, 3.4, 6.3, 2.6, 2.5, 4.8, 4.4, 6.0, 6.1, 6.8, 9.3, 6.8, 11.1, 3.5, 11.7, 6.4, 4.5 ],
    "Load distribution peak" : [np.max(load_data["System demand (MW)"])*x/100 for x in [3.8, 3.4, 6.3, 2.6, 2.5, 4.8, 4.4, 6.0, 6.1, 6.8, 9.3, 6.8, 11.1, 3.5, 11.7, 6.4, 4.5]],
    "Bid price" : [25, 22, 15.4, 12.5, 13, 14, 24, 15, 12.8, 17.8, 29.3, 28, 16.9, 30, 18, 16, 21]}
    
    #Wind farms data : We extract the production of the 24 first hours of each first dataset from each zone (1 zone 1 file)
    wind_farm = {}
    file_path = ["Wind farms data\scen_zone1.out", "Wind farms data\scen_zone2.out","Wind farms data\scen_zone3.out","Wind farms data\scen_zone4.out","Wind farms data\scen_zone5.out","Wind farms data\scen_zone6.out"]
    for i in range(len(file_path)):
        wind_farm[f"wind_farm {i+1}"] = pd.read_csv(file_path[i])["V1"].head(24).tolist()

    return {"generation_unit" : generating_unit_data, "load" : load_data,  "wind_farm" : wind_farm, "node_demand" : node_demand}
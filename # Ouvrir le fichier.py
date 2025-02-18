import pandas as pd

# File path of the 6 wind farms data
file_path = ["Wind farms data\scen_zone1.out", "Wind farms data\scen_zone2.out","Wind farms data\scen_zone3.out","Wind farms data\scen_zone4.out","Wind farms data\scen_zone5.out","Wind farms data\scen_zone6.out"]


# We extract the 24 first elements (for 24h) of each first wind turbine of each zone (1 zone 1 file)
wind_farm = {}

for i in range(len(file_path)):
    wind_farm[f"wind_farm {i}"] = pd.read_csv(file_path[i])["V1"].head(24).tolist()


# Afficher les résultats
print(wind_farm)

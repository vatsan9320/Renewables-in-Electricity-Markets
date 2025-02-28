import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Data
demand_quantities = [0, 0, 1000, 1500, 2000, 2650.5]
demand_prices = [20, 20, 18, 16, 14, 0]

supply_quantities = [
    152, 152, 350, 591, 60, 155, 155, 400, 400, 300, 300, 350,
    66.02668138964893, 69.90705836674121, 69.09000109053993,
    60.93791955818926, 64.88591062635679, 62.98944854324062]

supply_prices = [
    13.32, 13.32, 20.7, 20.93, 26.11, 10.52, 6.02, 5.47, 0, 0,
    10.52, 10.89, 0, 0, 0, 0, 0, 0]

# Sort supply data by ascending price
sorted_indices = np.argsort(supply_prices)
sorted_prices = np.array(supply_prices)[sorted_indices]
sorted_supply = np.array(supply_quantities)[sorted_indices]
sorted_cumulative_supply = np.cumsum(sorted_supply)

# # Print debug info (optional)
# print("Indices:", sorted_indices)
# print("Prices sorted:", sorted_prices)
# print("Supply sorted:", sorted_supply)
# print("Cumulative supply:", sorted_cumulative_supply)

# Create the figure
plt.figure(figsize=(10, 6))

# --- Supply (orange) ---
plt.step(sorted_cumulative_supply, sorted_prices, where='post',
         color='orange', label="Supply")
plt.plot(sorted_cumulative_supply, sorted_prices, marker='o', linestyle='',
         color='orange', label='_nolegend_')

# --- Demand (blue) ---
plt.step(demand_quantities, demand_prices, where='post',
         color='blue', label="Demand (2650 MW)")
plt.plot(demand_quantities, demand_prices, marker='o', linestyle='',
         color='blue', label='_nolegend_')

# Display the plot
plt.ylabel("Price")
plt.xlabel("Quantity (MW)")
plt.title("Supply and Demand")
plt.legend()
plt.grid(True)
plt.xlim(left=0)
plt.show()

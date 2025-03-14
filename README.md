# NSGA-II Algorithm for Optimal EV Charging Station Placement

This project utilizes the **NSGA-II algorithm** (Non-dominated Sorting Genetic Algorithm II) for optimizing the placement of **electric vehicle (EV) charging stations** on a road network. The optimization is performed using multiple parameters, such as station locations, battery capacities, and other factors, with the goal of achieving the most efficient charging network design.

## Stage 1: Finding the Optimal Charging Station Locations

- **`stage1_NSGA-II.py`**: The first stage of the NSGA-II algorithm is implemented here, aiming to find the best locations for the EV charging stations. The results are stored in a table in **`tables_output.txt`**.

- **`probabilities.py`** and **`weights_for_station.py`**: These files are used to design the tables used in Stage 1, based on relevant research papers and literature.

### Stage 1 Results:
After multiple runs of the algorithm, the **best solutions** (`best_solutions`) were found, which include the station locations and their evaluation based on several factors.

## Stage 2: BESS Capacity Calculation

- **`stage2_BESS_from_PV.py`**: In the second stage, the appropriate **BESS** (Battery Energy Storage Systems) capacity for each **best solution** is calculated. This calculation depends on the photovoltaic voltage at each charging station.

## Solution Classification and Analysis

- **`solutions_mif_sorted.py`**: In this file, the best solutions, along with BESS characteristics and other parameterization factors, are sorted using the **μ** factor. This allows extracting the most reliable and efficient solutions for the electric vehicle charging network.


import numpy as np

# Δεδομένα από το γράφημα %probability to km mexri na ftasei 20% ka8e aftokinito
probabilities = [0.015,0.03,0.065,0.1,0.1,0.09,0.06,0.03,0.015,0.005,0.001]
distances = [10,30,50,70,99,110,130,150,170,190,210] #drange*10km-30km wste na fortizetai sto 20%.

# Μετατρέπουμε τα δεδομένα σε πίνακα NumPy
data_matrix = np.column_stack((probabilities, distances))

#nodes
nodes_coordinates = {
    "C1": (0, 0), #C1     Αθήνα (Διόδια Ελευσίνας): (x=0, y=0)
    "C2": (16, 0), #C2     Μέγαρα: (x=16, y=0)
    "C3": (48, 0), #C3     Διόδια Ισθμού: (x=50, y=0)
    "C4": (55, 0), #C4     Κόρινθος: (x=62, y=0)
    "C5": (77, 0), #C5     Βέλο: (x=80, y=0)
    "C6": (84, 0), #C6     Διόδια Κιάτου: (x=85, y=0)
    "C7": (95, 0), #C7     Ξυλόκαστρο: (x=100, y=0)
    "C8": (129, 0), #C8     Ακράτα: (x=133, y=0)
    "C9": (138, 0), #C9     Διόδια Ελαιώνα: (x=154, y=0)
    "C10": (149, 0), #C10    Αίγιο: (x=179, y=0)
    "C11": (173, 0), #C11    Ψαθόπυργος: (x=204, y=0)
    "C12": (175, 0) #C12    Πάτρα (Διόδια Ρίου): (x=208, y=0)
}

def initialize_candidates():
    # Create array of candidate nodes
    candidates = np.array(list(nodes_coordinates.values()))
    return candidates

non_station_array = []
selected_nodes = []
candidates = initialize_candidates()
selected_node = candidates[0]
selected_nodes.append(selected_node)
print(selected_nodes)


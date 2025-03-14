import numpy as np

#μερες
#1 χειμωνας Κ
#2 χειμωνασ ΣΚ
#3 ανοιξη Κ
#4 ανοιξη ΣΚ
#5 καλοκαιρι Κ
#6 καλοκαιρι ΣΚ
#7 φθινοπωρο Κ
#8 φθινοπωρο ΣΚ

# Δεδομένα
data = {
    "Ελευσίνα προς Πάτρα": [16334, 25839, 19667, 28694, 33515, 35822, 20162, 24294],
    "Ισθμός προς Πάτρα": [11248, 20764, 13810, 22079, 24835, 24819, 14509, 20079],
    "Κιάτο προς Πάτρα": [5319, 8536, 6245, 8447, 10827, 10341, 6357, 8263],
    "Ελαιώνας προς Πάτρα": [4717, 6249, 5583, 6512, 8727, 8636, 5718, 6780],
    "Ρίο προς Πάτρα": [5818, 6811, 6747, 7325, 10137, 10515, 6933, 7278],
}

# Λίστες για τα βάρη ανά μέρα
weights_per_day = [[] for _ in range(8)]

# Υπολογισμός μέσων τιμών για κάθε κόμβο
averages = {key: np.mean(values) for key, values in data.items()}

# Υπολογισμός συνολικής μέσης τιμής
total_average = np.mean(list(averages.values()))

# Υπολογισμός βαρών κάθε κόμβου ανά μέρα
for day in range(8):
    for key, value in data.items():
        weights_per_day[day].append(np.mean(value[day]) / total_average)

# Εκτύπωση των λιστών βαρών ανά μέρα
for day, weights in enumerate(weights_per_day, 1):
    print(f"Βάρη για την ημέρα {day}:")
    for node, weight in zip(data.keys(), weights):
        print(f"{node}: {weight}")
    print()
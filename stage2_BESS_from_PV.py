import random
import math
from scipy.stats import norm
import numpy as np
from statistics import mean
import random
import math

#συναρτηση για random τιμη απο διαστημα [] βασει πιθανοτητων
def random_number_with_probability(min_val, max_val, probabilities):
    choices = np.arange(min_val, max_val+0.5, 0.5) #list of min-max value with step +0.5
    return random.choices(choices, weights=probabilities)[0] #random choice based on probabilities

#συναρτηση δινει τιμη στο Z_a/2 κανονικης κατανομης αναλογα το διαστημα εμπιστοσυνης 90,95,99%
def z_alpha_over_2(confidence_level):
    if confidence_level == 90:
        alpha = 0.10
    elif confidence_level == 95:
        alpha = 0.05
    elif confidence_level == 99:
        alpha = 0.01
    else:
        raise ValueError("Unsupported confidence level. Supported values: 90, 95, 99.")
    return norm.ppf(1 - alpha / 2)

#συναρτηση για μεση τιμη απο λιστα δεδομενων data
def mean_num(data):
    return sum(data) / len(data)

#συναρτηση τυπικης αποκλισης απο λιστα δεδομενων data
def standard_deviation(data):
    mu = mean_num(data)
    variance = sum((x - mu) ** 2 for x in data) / len(data)
    return math.sqrt(variance)

#συναρτηση αποστασης μεταξυ κομβων
def distance(p1, p2):
    #return abs(p1[0] - p2[0])  #pi[0] δομη κομβων _2
    return abs(p1 - p2)

#__________________________________________________________
#1
d_range_k_b = []
#times apo diagramma ana +0.5 vima
probabilities = [0.013, 0.015, 0.02, 0.026, 0.03, 0.038, 0.042, 0.058, 0.069, 0.077, 0.086, 0.094, 0.105, 0.11, 0.106, 0.107, 0.108, 0.1075, 0.106, 0.098, 0.09, 0.085, 0.075, 0.071, 0.06, 0.05, 0.042, 0.034, 0.028, 0.022, 0.02, 0.018, 0.016, 0.013, 0.01, 0.009, 0.008, 0.007, 0.005, 0.004, 0.003]

#10000 αμαξια 
k = 10000
k_24 = int(0.4*k)
k_30 = int(0.3*k)
k_40 = int(0.3*k)
#40% exoun 24kwh kai dianuoun apo 4-24km
for _ in range(k_24):
    random_num = random_number_with_probability(4,24,probabilities)
    d_range_k_b.append(random_num)
#30% exoun 30kwh kai dianuoun apo 7-27km
for _ in range(k_30):
    random_num = random_number_with_probability(7,27,probabilities)
    d_range_k_b.append(random_num)
#30% exoun 40kwh kai dianuoun apo 9-29km
for _ in range(k_40):
    random_num = random_number_with_probability(9,29,probabilities)
    d_range_k_b.append(random_num)

#print(d_range_k_b)

#_______________________________________________________________
#2
d_sr_k_b = []
confidence_levels = [90, 95, 99]
for level in confidence_levels:
    z_value=z_alpha_over_2(level)
    d_sr_k_b.append(mean_num(d_range_k_b) - (z_value * standard_deviation(d_range_k_b)))

EV_sr_stn = min(d_sr_k_b)

#print(d_sr_k_b)
print("aktina = ", EV_sr_stn)

#__________________________________________________________
#3
def check_inequality_3(EV_sr_stn, d_cs):
    return EV_sr_stn <= d_cs and d_cs <= 2 * EV_sr_stn

#________________________________________________________
#4
nodes = [0, 17, 47, 58, 75, 84, 97, 131, 138, 149, 176, 180] #καθε στοιχειο της λιστας ειναι το σημειο x του κομβου Ci (i=0 mexri i=12-1.)
# κομβοι - πιθανοι σταθμοι φορτισης 
# θεωρω ολοι στην ιδια ευθεια στον αξονα y και στον αξονα x ειναι η χιλιομετρικη αποσταση τους απο τον κομβο αναφορας C1:Αθηνα (Διοδια Ελευσινας)
# επιλεγω διαδρομη Αθηνα προς Πατρα
# προκυπτουν τα αντιστοιχα βαρη στους κομβουσ απο τα δεδομενα των διοδιων
#εχω 8 σεναρια, καθημερινη και σκ καθε εποχη, 2σεναρια/εποχη
scenarios = {
    "Καθημερινή": {
        "Άνοιξη": [1.46, 1.22, 1.03, 0.8, 0.6, 0.46, 0.45, 0.42, 0.41, 0.44, 0.48, 0.5],     #1o σεναριο
        "Καλοκαίρι": [2.5, 2.1, 1.85, 1.3, 0.9, 0.8, 0.73, 0.68, 0.65, 0.69, 0.73, 0.75],  #2o σεναριο
        "Φθινόπωρο": [1.5, 1.22, 1.08, 0.88, 0.6, 0.47, 0.45, 0.43, 0.42, 0.48, 0.5, 0.51],  #3o σεναριο
        "Χειμώνας": [1.22, 1.9, 0.84, 0.75, 0.47, 0.39, 0.38, 0.36, 0.35, 0.37, 0.4, 0.43]    #4o σεναριο
    },
    "Σαββατοκύριακο": {
        "Άνοιξη": [2.14, 1.8, 1.64, 1.3, 0.9, 0.63, 0.58, 0.5, 0.48, 0.5, 0.52, 0.54],     #5o σεναριο
        "Καλοκαίρι": [2.67, 2.15, 1.85, 1.35, 1.1, 0.77, 0.72, 0.67, 0.64, 0.68, 0.75, 0.78],  #6o σεναριο
        "Φθινόπωρο": [1.81, 1.67, 1.5, 1.1, 0.78, 0.61, 0.57, 0.53, 0.5, 0.52, 0.53, 0.54],  #7o σεναριο
        "Χειμώνας": [1.93, 1.73, 1.55, 1.12, 0.88, 0.63, 0.58, 0.52, 0.46, 0.48, 0.49, 0.5]    #8o σεναριο
    }
}

def f_OD_dy(i1, i2, nodes, weights):
    if i1 <= 0 or i1 > len(nodes) or i2 <= 0 or i2 > len(nodes):    #δεν θελω πανω απο 12 ή <=0 index
        return "Μη έγκυροι δείκτες κόμβων"
    if i1 == i2:
        return 0
    else:
        d_od = distance(nodes[i1-1], nodes[i2-1])       #βαζω -1 γτ στην λιστα εχω 0-11 ενω την καλω με 1-12
        return 1.5 * (weights[i1-1] * weights[i2-1]) / d_od
    
#creates table 12x12 (nodesxnodes) with weights of each scenario
def generate_table(nodes, weights):
    table = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            table[i][j] = f_OD_dy(i + 1, j + 1, nodes, weights)
    return table

#_________________________________________________________________________________________________
#5
#υπολογισμος του Σ(Σ(f_OD_dy)) για ολα τα ζευγη κομβων

def sum_of_table(table):
    total_sum = 0
    for row in table:
        for value in row:
            total_sum += value
    return total_sum

def f_OD_dy_b_t(value, SS, N):
    return N*value/SS
    
#τιμες για N_total_b_dy για καθε dy [b24, b30, b40]
#mesos ari8mos = 13339, 40% exoun 24kwh, 30% exoun 30kwh, 30% exoun 40kwh.
N_values = {       #peak-time καθε σεναριου /60 για καθε λεπτο
    "Καθημερινή": {
        "Άνοιξη": [5, 4, 4],
        "Καλοκαίρι": [8, 6, 6],
        "Φθινόπωρο": [5, 4, 4],
        "Χειμώνας": [5, 4, 4]
    },
    "Σαββατοκύριακο": {
        "Άνοιξη": [9, 7, 7],
        "Καλοκαίρι": [14, 11, 11],
        "Φθινόπωρο": [8, 6, 6],
        "Χειμώνας": [9, 7, 7]
    }
}

# Dictionary to store tables
tables_dict_f_OD_dy_b_t = {}
SS_f_OD_dy_b_t = []
#φτιαχνω την (5): f_OD_dy_b_t
for scenario, data in scenarios.items():
    tables_dict_f_OD_dy_b_t[scenario] = {}
    #print(f"Tables for {scenario}:")
    for season, weights in data.items():
        table_f_OD_dy = generate_table(nodes, weights)  #f_OD_dy για ζευγος κομβου
        #print(table_f_OD_dy)
        SS_f_OD_dy = sum_of_table(table_f_OD_dy)  # Sum of values in the current table
        #print(SS_f_OD_dy)

        N_values_for_scenario_season = N_values.get(scenario, {}).get(season)
        if N_values_for_scenario_season is None:
            print(f"No N values specified for {scenario} - {season}. Skipping...")
            continue

        tables_dict_f_OD_dy_b_t[scenario][season] = {}
        for i, b in enumerate([24, 30, 40]):
            N_total_b_dy = N_values_for_scenario_season[i]
            table_f_OD_dy_b_t = np.zeros_like(table_f_OD_dy)
            for i in range(len(table_f_OD_dy)):
                for j in range(len(table_f_OD_dy[i])):
                    table_f_OD_dy_b_t[i][j] = f_OD_dy_b_t(table_f_OD_dy[i][j], SS_f_OD_dy, N_total_b_dy)
            SS_f_OD_dy_b_t.append(sum_of_table(table_f_OD_dy_b_t))
            tables_dict_f_OD_dy_b_t[scenario][season][b] = table_f_OD_dy_b_t
            #print(f"{season} - Battery {b}: Stored as variable tables_dict_f_OD_dy_b_t['{scenario}']['{season}'][{b}]")
#print("a8roismata", SS_f_OD_dy_b_t)
#περναω τους 24 πινακες σε ξεχωριστο αρχειο, δεν χωρανε να τυπωθουν στο terminal 
with open("tables_output.txt", "w", encoding="utf-8") as f:
    for scenario, scenario_data in tables_dict_f_OD_dy_b_t.items():
        f.write(f"Scenario: {scenario}\n")
        for season, season_data in scenario_data.items():
            f.write(f"\tSeason: {season}\n")
            for battery, table in season_data.items():
                f.write(f"\t\tBattery: {battery}\n")
                f.write(f"{table}\n\n")

#________________________________________________________________________
#7
#γ=1 αν υπαρχει σταθμος στο OD μονοπατι αλλιως 0
#β=1 αν ο σταθμος μπορει να παρεχει υπηρεσια στο κ αυτοκινητο αλλιως 0
vita_OD_b_stn = 1
gama_stn=1
f_srv_b_t_stn_dy = []
for i in range(len(SS_f_OD_dy_b_t)):
    f_srv_b_t_stn_dy.append(SS_f_OD_dy_b_t[i]*vita_OD_b_stn*gama_stn)

#print(SS_f_OD_dy_b_t)
#print("roiiiiii", f_srv_b_t_stn_dy)

#___________________________________________________________________
#6
T_lbt_stn_dy_all = []   #τιμες για καθε σεναριο καθε μπαταρια
for i in range(len(f_srv_b_t_stn_dy)):
    T_lbt_stn_dy_all.append(1/f_srv_b_t_stn_dy[i])
#print("\nri8moss", T_lbt_stn_dy_all)

#λιστες καθε μπαταριας με τα σεναρια
T_lbt_stn_dy_24 = T_lbt_stn_dy_all[::3]
T_lbt_stn_dy_30 = T_lbt_stn_dy_all[1::3]
T_lbt_stn_dy_40 = T_lbt_stn_dy_all[2::3]

#μεση τιμη των λιστων
T_lb1t_stn_dy = mean_num(T_lbt_stn_dy_24)   #24kwh
T_lb2t_stn_dy = mean_num(T_lbt_stn_dy_30)   #30kwh
T_lb3t_stn_dy = mean_num(T_lbt_stn_dy_40)   #40kwh

T_lbt_stn_dy = []   #μεση τιμη των σεναριων για καθε μπαταρια
T_lbt_stn_dy.append(T_lb1t_stn_dy)
T_lbt_stn_dy.append(T_lb2t_stn_dy)
T_lbt_stn_dy.append(T_lb3t_stn_dy)

#print(T_lbt_stn_dy)

#______________________________________________________
#8
T_lbt_stn_dy_effective = (1/T_lb1t_stn_dy + 1/T_lb2t_stn_dy + 1/T_lb3t_stn_dy)**(-1)
#print(T_lbt_stn_dy_effective)

#_________________________________________________________________
#9
e=0.272 #ε απο πινακα ε(kwh/miles)
d_avg_daily=112.5 #απο gpt, εστω οτι 50% 75km, 30% 130km, 20% 180km
E_nr_d = e*d_avg_daily

#__________________________________________________________
#10
C_rate_b = [24, 30, 40]  #1C => ρυθμος φορτισης = χωρητικοτητα μπαταριας
SOC_ch_b = round(random.uniform(0.3,0.5),2) #οταν φτανουν εχουν απο 30% μεχρι 50% μπαταρια
delta_SOC_b = 0.8-SOC_ch_b
T_service_b = []
for i in range(len(C_rate_b)):
    T_service_b.append(delta_SOC_b/int(C_rate_b[i]))
#print(T_service_b)

#__________________________________________________________
#11
T_service_effective = max(T_service_b)
#print(T_service_effective)

#_________________________________________
#13
T_stn_service_b = 0.6 #36 minutes => 0.6 hours
C_spot_stn_min = math.ceil(T_stn_service_b/T_lbt_stn_dy_effective) #θεωρησαμε peak time την t που πηραμε πριν τους πινακες!!!!!!!!!!!!!!!!!!!!!!!
#print(C_spot_stn_min)

#__________________________________________________________________________________________
#12
C_spot_stn = random.randint(C_spot_stn_min, C_spot_stn_min)  #μεταξυ min και min
#print(C_spot_stn)
ro_pt_stn_dy = T_stn_service_b/(T_lbt_stn_dy_effective*C_spot_stn)
#print(ro_pt_stn_dy)

def check_inequality_12(ro_pt_stn_dy):
    return ro_pt_stn_dy < 1

#____________________________________________________________________________
#16
def calculate_sum16(C_spot_stn, ro_pt_stn_dy):
    total_sum = 0
    factorial_C_spot_stn = math.factorial(C_spot_stn)

    for n in range(C_spot_stn):     #για n=0 μεχρι n=C_spot_stn-1
        term = ((C_spot_stn*ro_pt_stn_dy)**n)/math.factorial(n)
        total_sum += term
    total_sum += ((C_spot_stn*ro_pt_stn_dy)**C_spot_stn)/(factorial_C_spot_stn*(1-ro_pt_stn_dy))
    return total_sum

sum16 = calculate_sum16(C_spot_stn, ro_pt_stn_dy)  
#print(sum1)
P0_pt_stn_dy = 1/sum16
#print(P0_pt_stn_dy)
#__________________________________________________________________________________________
#15
W_pt_stn_dy_all = []

for i in range(len(T_lbt_stn_dy_all)):
    w = (((C_spot_stn*ro_pt_stn_dy)**C_spot_stn)*ro_pt_stn_dy*T_lbt_stn_dy_all[i]*P0_pt_stn_dy)/(math.factorial(C_spot_stn)*((1-ro_pt_stn_dy)**2))
    W_pt_stn_dy_all.append(w)
#print(W_pt_stn_dy_all)

#για καθε σεναριο βρισκω μεσο W απο τυπους μπαταριων
W1 = mean_num(W_pt_stn_dy_all[:3])
W2 = mean_num(W_pt_stn_dy_all[3:6])
W3 = mean_num(W_pt_stn_dy_all[6:9])
W4 = mean_num(W_pt_stn_dy_all[9:12])
W5 = mean_num(W_pt_stn_dy_all[12:15])
W6 = mean_num(W_pt_stn_dy_all[15:18])
W7 = mean_num(W_pt_stn_dy_all[18:21])
W8 = mean_num(W_pt_stn_dy_all[21:])

W_pt_stn_dy = [W1,W2,W3,W4,W5,W6,W7,W8]
#print(W_pt_stn_dy)

#_____________________________________________________________________________________________________________________________________
#14
W_pt_stn = mean_num(W_pt_stn_dy_all)     #μεσος χρονος αναμονης απο all
#print(W_pt_stn)
W_max = int(max(W_pt_stn_dy_all))     #μεγιστος χρονος αναμονης απο all
#print(W_max)

def check_inequality_14(W_pt_stn, W_max):
    return W_pt_stn<=W_max

#__________________________________________________
#17     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
mean_n_dy = []
for day,seasons in N_values.items():
    for season, values in seasons.items():
        mean_n_dy.append(int(mean_num(values)))
#print(mean_n_dy)

P_n_stn_dy = []
for i in range(len(mean_n_dy)):
    #δεν δουλευει ως πιθανοτητα για να βγουν τα νουμερα τα κανω δια 10000.
    #αλλιως παραδειγμα μουστακα!! ->εκθετη e^.. ->κανονικοποιηση /Σ(e^..)
    P_n_stn_dy.append((((C_spot_stn*ro_pt_stn_dy)**mean_n_dy[i])/math.factorial(mean_n_dy[i]))/10000)   
#print(P_n_stn_dy)

#______________________________________________________________________________________________
#18
P_Crate_spot = 40  #kw 
P_n_t_stn_dy = []
for i in range(len(P_n_stn_dy)):
    P_n_t_stn_dy.append(P_n_stn_dy[i]*T_stn_service_b)
#print(P_n_t_stn_dy)

def calculate_sum18(P_n_t_stn_dy_i):
    total_sum = 0
    for n in range(C_spot_stn+1):   #απο n=0 μεχρι n=C_spot_stn
        term = n*P_Crate_spot*P_n_t_stn_dy_i
        total_sum += term
    return total_sum

P_chd_t_stn_dy = []
for i in range(len(P_n_t_stn_dy)):
    P_chd_t_stn_dy.append(calculate_sum18(P_n_t_stn_dy[i]))
#print(P_chd_t_stn_dy)

#___________________________________________________________________________
#19
P_exp_chd_t_stn_dy = []
a_p = 0.246
b_p = 0.9754
n_p = 1.839
V_o = 1.0
V_max = 1.05
V_min = 0.95
V_stn_t = round(random.uniform(V_min,V_max),2)  #τυχαια τιμη μεταξυ Vmin-Vmax με 2 δεκαδικα ψηφια

for i in range(len(P_chd_t_stn_dy)):
    P_exp_chd_t_stn_dy.append(P_chd_t_stn_dy[i]*(a_p*((V_stn_t/V_o)**n_p)+b_p))
print("geiaa\n",P_exp_chd_t_stn_dy)

#______________________________________________________________________________________
#20
#χρηση data απο Εικονα 4 στη διπλωματικη
ir_rated = [83.53, 91.58, 93.64, 127.96, 141.99, 151.4, 154.01, 168.19, 168.72, 141.45, 116.45, 95.66]  #απο δεκ μεχρι νοεμ. (χειμ-φθινοπ)
P_ir_rated = [5239.49, 5882.88, 5668.65, 4126.75] #το max καθε εποχης απο excel data χειμ-φθιν
#μεση τιμη καθε εποχης
ir_t_s_winter = mean_num(ir_rated[:3])
ir_t_s_spring = mean_num(ir_rated[3:6])
ir_t_s_summer = mean_num(ir_rated[6:9])
ir_t_s_automn = mean_num(ir_rated[9:])
ir_t_s = [ir_t_s_winter,ir_t_s_spring,ir_t_s_summer,ir_t_s_automn]  
#print(ir_t_s)
#ισχυς μηνων καθε εποχης
P_pvm_i_t_stn_S_all = []
ir_rated_winter = ir_rated[:3]
ir_rated_spring = ir_rated[3:6]
ir_rated_summer = ir_rated[6:9]
ir_rated_automn = ir_rated[9:]

ir_rated_mean = [mean(ir_rated_winter), mean(ir_rated_spring), mean(ir_rated_summer), mean(ir_rated_automn)]
P_pvm_i_t_stn_S = []
for i in range(len(ir_rated_mean)):
    if(ir_rated_mean[i]<=ir_t_s[i]):
        P_pvm_i_t_stn_S.append(P_ir_rated[i])
    else:
        P_pvm_i_t_stn_S.append(P_ir_rated[i]*ir_t_s[i]/ir_rated_mean[i])
#print(P_pvm_i_t_stn_S)


def Eq19(P_exp_chd_t_stn_dy):   #δινει τα αναμενομενα PV
    return P_exp_chd_t_stn_dy   #expected demand everyday on peak-time

#considering max irradiance
ir_rated_max = [max(ir_rated_winter), max(ir_rated_spring), max(ir_rated_summer), max(ir_rated_automn)] 
#print(ir_rated_max)

def Eq21(ir_rated_max):     #δινει την αναμενομενη max ακτινοβολια καθε εποχης
    P_pvm_i_t_stn_S_ = []
    for i in range(len(ir_rated_max)):
        if(ir_rated_max[i]<=ir_t_s[i]):
            P_pvm_i_t_stn_S_.append(P_ir_rated[i])
        else:
            P_pvm_i_t_stn_S_.append(P_ir_rated[i]*ir_t_s[i]/ir_rated_max[i])
    return P_pvm_i_t_stn_S_   #expected irradiance for each season

def updated_pv_output(step_value, calculated_pv_output):
    updated_pv_output = calculated_pv_output + step_value
    return updated_pv_output

def algorithm(calculated_pv_output_value, expected_pv_output):
    counter = 0
    threshold = 30
    step_value = 100
    updated_pv = updated_pv_output(0, calculated_pv_output_value)
    while counter < threshold:
        error = updated_pv - expected_pv_output
        if error < 0:
            lower_step_limit = step_value
        else:
            if error > 0:
                lower_step_limit = step_value
            else:
                optimal_step = step_value
                break
        updated_pv = updated_pv_output(step_value, calculated_pv_output_value)
        counter += 1
        step_value += 100
        if counter >= threshold:
            optimal_step = step_value
            break

    energy_required_bess = abs(updated_pv - expected_pv_output)
    bess_capacity = (energy_required_bess / 0.75) + (0.05 * (energy_required_bess / 0.75))
    return bess_capacity if calculated_pv_output_value != 0 else 0

# Solution data
solution = [
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 10.0, 0.0, 0.0],
    [0.0, 0.0, 183.5, 0.0, 0.0, 0.0, 160.4, 0.0, 0.0, 220.0, 0.0, 0.0]
]
expected_pv_output = 2766.11  # From the provided data, low sun and high traffic

# Calculate BESS capacity for non-zero values of calculated_pv_output
bess_capacities = []
for calculated_pv_output_value in solution[2]:
    bess_capacity = algorithm(calculated_pv_output_value, expected_pv_output)
    bess_capacities.append(bess_capacity)

print("BESS Capacities for each non-zero calculated PV output:")
print(bess_capacities)
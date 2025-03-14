import random
import math
from scipy.stats import norm
import numpy as np
from statistics import mean
import random
import math
import json

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
#print("aktina = ", EV_sr_stn)

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
#print(len(P_exp_chd_t_stn_dy))

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

#__________________________________________________________________________________________________________________
#22
Ff = 0.75 #τυπικη τιμη απο gpt
#με υπολογισμους απο gpt εχουμε
V_oc = [3.7, 4.2, 4.4, 3.9] #V χειμ-φθιν
I_sc = [6, 6.9, 8, 6.7] #A χειμ-φθιν
V_mp = [300, 340, 370, 330] #V χειμ-φθιν
I_mp = [5, 6, 7.2, 5.8] #A
I_oc = [7, 8.2, 9, 7.8] #A
#Ff = []
#for i in range(len(V_oc)):
    #Ff.append(V_mp[i]*I_mp[i]/(V_oc[i]*I_oc[i]))
#print(Ff)
#_______________________________________________
#25
C_v = -0.35 #%/C
C_i = 0.045 #%/C
t_amb = [8, 18, 32, 15]
t_nom = 25
t_cell = []
#μετατροπη του ir_t_s που ειναι μεση ακτινοβολια μηνα καθε εποχης /30 για μεση ακτινοβολια ημερας καθε εποχης
for i in range(len(ir_t_s)):
    ir_t_s[i] = ir_t_s[i]/30/11 #/11 ωρες που εχουμε ηλιο
#print(ir_t_s)
for i in range(len(t_amb)):
    t_cell.append(t_amb[i]+(ir_t_s[i]*((t_nom-20)/0.8)))
#print(t_cell)  #κομπλε τιμες!

#__________________________________________________________
#23
V_ir = []
for i in range(len(t_cell)):
    V_ir.append(V_oc[i] - C_v*t_cell[i])
#print(V_ir)

#________________________________________
#24
I_ir = []
for i in range(len(t_cell)):
    I_ir.append(ir_t_s[i]*(I_sc[i]+C_i*(t_cell[i]-25)))
#print(I_ir)

#_________________________________________________________
#21
#με τα δεδομενα της 20 (τιμες P) και δεδομενο το Ff Iir Vir θα βρω Np
Np_S = []
V_iv = [8.5, 12, 17, 10]    #max Volt απο τυπικο διαστημα τιμων
for i in range(len(P_pvm_i_t_stn_S)):
    Np_S.append(P_pvm_i_t_stn_S[i]/30/(Ff*V_iv[i]*I_ir[i]))  #παλι /30 για ημερησιως
#print(Np_S)
Np = math.ceil(mean_num(Np_S)) #μεσος αριθμος φωτοβολταικων για ολες τις εποχες 
#print(Np)
#______________________________________________________________
#26
htta_BESS_ch = "{:.2f}".format(random.uniform(0.9,0.95))
htta_BESS_disch = "{:.2f}".format(random.uniform(0.9,0.95))
#επιλεγω σεναριο 13/07 ΚΑΛΟΚΑΙΡΙ-ΚΑΘΗΜΕΡΙΝΗ εχουμε την μεγαλυτερη παραγωγη και λιγοτερη κινηση
PV_power_scenario = [0,0,0,0,23.59,185.73,392.21,561.13,677.6,731.2,761.43,730.97,677.02,552.73,370.62,183.68,34.97,0,0,0,0,0,0,0]
BESS_stn_Power_ch_S = []
BESS_stn_Power_disch_S = []
for i in range(len(PV_power_scenario)):
    BESS_stn_Power_ch_S.append(PV_power_scenario[i])    #ολη η παραγωγη 
    BESS_stn_Power_disch_S.append(PV_power_scenario[i])   #ολη η παραγωγη
#print(BESS_stn_Power_ch_S)
#print(BESS_stn_Power_disch_S)

BESS_Power_ch = []
cars_num_t = []
cars_num_t.append(mean([169,148,73,51,81]))
cars_num_t.append(mean([137,97,50,52,52]))  
cars_num_t.append(mean([97,76,34,34,40]))
cars_num_t.append(mean([91,71,33,25,33]))
cars_num_t.append(mean([108,81,58,35,32]))
cars_num_t.append(mean([228,130,55,60,76]))
cars_num_t.append(mean([531,296,133,95,114]))
cars_num_t.append(mean([800,510,237,162,202]))
cars_num_t.append(mean([874,656,284,263,364]))
cars_num_t.append(mean([962,687,319,271,337]))
cars_num_t.append(mean([1141,838,341,283,356]))
cars_num_t.append(mean([1121,847,375,304,312]))
cars_num_t.append(mean([1013,862,357,286,335]))
cars_num_t.append(mean([993,759,335,288,308]))
cars_num_t.append(mean([994,732,337,270,277]))
cars_num_t.append(mean([1036,676,304,250,291]))
cars_num_t.append(mean([1001,696,299,244,264]))
cars_num_t.append(mean([1182,717,279,220,255]))
cars_num_t.append(mean([1028,763,327,250,280]))
cars_num_t.append(mean([1012,757,286,227,283]))
cars_num_t.append(mean([871,556,276,222,235]))
cars_num_t.append(mean([662,468,190,164,230]))
cars_num_t.append(mean([443,743,292,131,138]))
cars_num_t.append(mean([267,258,145,184,181]))
for t in range(len(PV_power_scenario)):
    BESS_Power_ch.append(PV_power_scenario[t] - P_exp_chd_t_stn_dy[4]*(cars_num_t[t]/max(cars_num_t)))  #παραγωγη - καταναλωση
BESS_stn = [0]
for t in range(len(BESS_stn_Power_ch_S)-1):
    if(BESS_Power_ch[t]>0):
        BESS_stn.append(BESS_stn[t] + BESS_stn_Power_ch_S[t]*float(htta_BESS_ch))
    if(BESS_Power_ch[t]<=0):
        BESS_stn.append(BESS_stn[t] + BESS_stn_Power_disch_S[t]/float(htta_BESS_ch))

#print(BESS_Power_ch)
#κανονικοποιηση της χωρητικοτητας των BESS, μηχανικο trick για όρια στον γενετικό
#print(max(BESS_stn)/24) #max οριο -> περίπου 200κατι, μέγιστη τιμή κατανεμημένη στην μέρα (24 ωρες)
#print(mean(BESS_stn)/max(BESS_stn)*100) #min οριο -> περίπου 50κατι, ποσοστό μέσης τιμής σε σχέση με το max σε μία μέρα
#_____________________________________________________________________________________________
#27
#βαζουμε περιπτωσεις στην #20

#________________________________
#28
#!!!!!!!!!!! υποθετω χωρητικοτητα BESS 200kwh
BESS_SOC_stn_final = BESS_stn[-1]/(200*1000)*100 #% απο την χωρητικοτητα της μπαταριας
def check_equality_28(BESS_SOC_stn_initial,BESS_SOC_stn_final):
    return BESS_SOC_stn_initial == BESS_SOC_stn_final

#_________________________________________________________________
#29
BESS_r_i_t_Ch_S = sum(BESS_stn_Power_ch_S) #συνολικη φορτιση
def check_inequality_29(BESS_r_i_t_Ch_S, P_pvm_i_t_stn_S):
    return BESS_r_i_t_Ch_S <= sum(P_pvm_i_t_stn_S)

#__________________________________________________________
#30
def check_inequality_30(BESS_SOC_stn_min,BESS_SOC_stn_t,BESS_SOC_stn_max):
    return BESS_SOC_stn_min <= BESS_SOC_stn_t and BESS_SOC_stn_t <= BESS_SOC_stn_max

#______________________________________________________________________________________

#32
r = 0.05 #5% inerest rate
zita = r*((1+r)**Np)/((1+r)**(Np+1)-1)

#______________________________________________________________________________________
#34
C_spot_cost = 17000 #€
O_and_M = 0.05
DF = 7.7218 #discount factor
CPC = C_spot_cost*C_spot_stn*(1 + O_and_M*DF)

#______________________________________________________________________________________
#35
C_l = 370 #€/m^2
A_s = 25 #in m^2  A_spot_Cs
tax = 0.24  #foros ellados
LC = C_l*A_s*C_spot_stn*(1 + tax*DF)

#______________________________________________________________________________________
#33
DS = 65000 #€ DC_CS charging station
C1_CS = DS
C2_CS = CPC + LC
DC = 37000 #€ DC_PV photovoltaic station
C3_PV = DC 
UC = 0.1*C3_PV #€ 10% peripou tou sunolikou kostous
C4_PV = UC + LC 

#______________________________________________________________________________________
#31

N_stn = 12  #komvoi
#x_stn_Ch = [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1] #tyxaio 8a to paroume genetika
#y_stn_Ch = [11, 0, 0, 0, 11, 0, 11, 0, 0, 11, 0, 11]   #αριθμοσ φορτιστων απο γενετικο αλγοριθμο
#y_stn_PV = [30, 0, 0, 0, 30, 0, 30, 0, 0, 30, 0, 30]   #αριθμος χωρητικοτητας φωτοβολταικων απο γενετικο αλγοριθμο
#x_stn_PV = x_stn_Ch #θεωρω οπου σταθμοι εκει και τα φωτοβολταικα

def sum_31_1(x_stn_Ch, y_stn_Ch):
    sum31_1 = 0
    for i in range(N_stn):
        if(x_stn_Ch[i]==1):
            sum31_1 += C1_CS*x_stn_Ch[i] + C2_CS*y_stn_Ch[i]   #αλλαγη y_stn_CH βασει ολου του ατομου (2η σειρα)
    return sum31_1

def sum_31_2(x_stn_PV, y_stn_PV):
    sum31_2 = 0
    for i in range(N_stn):
        if(x_stn_PV[i]==1):
            sum31_2 += C3_PV*x_stn_PV[i] + C4_PV*y_stn_PV[i]   #αλλαγη y_stn_PV βασει ολου του ατομου (3η σειρα)
    return sum31_2

def OF1(x_stn_Ch,x_stn_PV, y_stn_Ch, y_stn_PV):
    return zita*(sum_31_1(x_stn_Ch, y_stn_Ch) + sum_31_2(x_stn_PV, y_stn_PV))
#OF1 = OF1(x_stn_Ch,x_stn_PV,y_stn_Ch,y_stn_PV)
#print(OF1)

#______________________________________________________________________________________
#37   
def calculate_N_B(x_stn_Ch):
    N_B = 0
    for i in range(len(x_stn_Ch)):
        if(x_stn_Ch[i] == 1):
            N_B += 1
    return N_B 

def OF2_minV_maxV_(x_stn_Ch):
    N_B = calculate_N_B(x_stn_Ch)
    #print(N_B)
    G_winter = [[random.uniform(0.1, 0.6) for j in range(N_B)] for i in range(N_B)]
    G_spring = [[random.uniform(0.1, 0.7) for j in range(N_B)] for i in range(N_B)]
    G_summer = [[random.uniform(0.1, 0.9) for j in range(N_B)] for i in range(N_B)]
    G_autumn = [[random.uniform(0.1, 0.8) for j in range(N_B)] for i in range(N_B)]

    V = [random.randint (220,400) for i in range(N_B)]  #220 - 400 Volts
    V_ = [[random.randint(220,240) for j in range(24)] for i in range(N_B)] #220 - 240 Volts
    #print(V_)
    min_V_ = min(min(V_))
    max_V_ = max(max(V_))
    #print(min_V_, max_V_)
    theta = [[random.uniform(0, 2*math.pi) for j in range(N_B)] for i in range(N_B)]
    
    P_loss_winter = 0
    P_loss_spring = 0
    P_loss_summer = 0
    P_loss_autumn = 0

    for t in range(24):
        for i in range(N_B):
            for j in range(N_B):
                P_loss_winter += G_winter[i][j] * (V_[i][t]**2 + V_[j][t]**2 - 2 * V[i] * V[j] * math.cos(theta[i][j]))
                P_loss_spring += G_spring[i][j] * (V_[i][t]**2 + V_[j][t]**2 - 2 * V[i] * V[j] * math.cos(theta[i][j]))
                P_loss_summer += G_summer[i][j] * (V_[i][t]**2 + V_[j][t]**2 - 2 * V[i] * V[j] * math.cos(theta[i][j]))
                P_loss_autumn += G_autumn[i][j] * (V_[i][t]**2 + V_[j][t]**2 - 2 * V[i] * V[j] * math.cos(theta[i][j]))
    
    P_loss_S_dy = [0.2*P_loss_winter, 0.3*P_loss_spring, 0.5*P_loss_summer, 0.3*P_loss_autumn]
    #print(P_loss_S_dy)
    P_mean = sum(P_loss_S_dy)/len(P_loss_S_dy)
    return P_mean, min_V_, max_V_, P_loss_S_dy

#______________________________________________________________________________________
#36
#OF2 = OF2_minV_maxV_(x_stn_Ch)[0]
#print(OF2)

#______________________________________________________________________________________
#38-41 tis afinoume gia tin wra

#______________________________________________________________________________________
#42
def OF3 (x_stn_Ch):
    N_B = calculate_N_B(x_stn_Ch)

    sum42_winter = 0
    sum42_spring = 0
    sum42_summer = 0
    sum42_autumn = 0
    V_dy_winter = [[random.randint(220,230) for j in range(24)] for i in range(N_B)]
    V_dy_spring = [[random.randint(220,240) for j in range(24)] for i in range(N_B)]
    V_dy_summer = [[random.randint(210,230) for j in range(24)] for i in range(N_B)]
    V_dy_autumn = [[random.randint(220,240) for j in range(24)] for i in range(N_B)]
    for t in range(24):
        for i in range(N_B):
            sum42_winter += (abs(V_dy_winter[i][t] - V_o)/V_o)*100
            sum42_spring += (abs(V_dy_spring[i][t] - V_o)/V_o)*100
            sum42_summer += (abs(V_dy_summer[i][t] - V_o)/V_o)*100
            sum42_autumn += (abs(V_dy_autumn[i][t] - V_o)/V_o)*100
        sum42_winter_ = 1/N_B * sum42_winter
        sum42_spring_ = 1/N_B * sum42_spring
        sum42_summer_ = 1/N_B * sum42_summer
        sum42_autumn_ = 1/N_B * sum42_autumn
    V_dev_dy_winter = 1/24 * sum42_winter_
    V_dev_dy_spring = 1/24 * sum42_spring_
    V_dev_dy_summer = 1/24 * sum42_summer_
    V_dev_dy_autumn = 1/24 * sum42_autumn_

    V_dev_dy_S = [V_dev_dy_winter, V_dev_dy_spring, V_dev_dy_summer, V_dev_dy_autumn]
    #print(V_dev_dy_S)
    return max(V_dev_dy_S)
#print(V_dev_dy_S)

#______________________________________________________________________________________
#43
#OF3 = OF3(x_stn_Ch) #minimize
#print(OF3)

#______________________________________________________________________________________
#44
def check_inequality_44(V_, x_stn_Ch):
    N_B = calculate_N_B(x_stn_Ch)
    V_i_min = OF2_minV_maxV_(x_stn_Ch)[1]
    V_i_max = OF2_minV_maxV_(x_stn_Ch)[2]
    for t in range(24):
        for i in range(N_B):
            if(V_i_min <= V_[i][t] <= V_i_max): #ισως απο 3η σειρα ατομου?
                return True

#______________________________________________________________________________________
#45
def get_delta_OD(x_stn_Ch):
    delta_OD = []
    for i in range(N_stn):
        row = []
        for j in range(N_stn):
            row.append(0)
        delta_OD.append(row)
    for i in range(N_stn):
        for j in range(N_stn):
            if (i==j and x_stn_Ch[i]==1):
                delta_OD[i][j] = 1
            if(i!=j and (x_stn_Ch[i]==1 or x_stn_Ch[j]==1)):
                delta_OD[i][j] = 1
            if(i<j):
                for k in range(i+1,j):
                    if(x_stn_Ch[k]==1):
                        delta_OD[i][j] = 1
                        break
            if(i>j):
                for k in range(j+1,i):
                    if(x_stn_Ch[k]==1):
                        delta_OD[i][j] = 1
                        break
    return delta_OD

SS_f_OD_dy_b_t_delta_OD = []
SS_f_OD_dy_b_t_delta_OD_ = []

def OF4(x_stn_Ch):
    delta_OD = get_delta_OD(x_stn_Ch)
    #12x12 to delta_OD
    new_table_dictionary = []
    for scenario, scenario_data in tables_dict_f_OD_dy_b_t.items():
        for season, season_data in scenario_data.items():
            for battery, table in season_data.items():
                new_table = [[0] * N_stn for _ in range(N_stn)]
                for i in range(N_stn):
                    for j in range(N_stn):
                        value = table[i][j] * delta_OD[i][j]
                        new_table[i][j] = value
                new_table_dictionary.append(new_table)
    #print("edwwww\n", new_table_dictionary)

    SS_f_OD_dy_b_t_delta_OD = []
    for element in new_table_dictionary:
        element_sum = 0
        for inner_list in element:
            inner_list_sum = sum(inner_list)
            element_sum += inner_list_sum
        SS_f_OD_dy_b_t_delta_OD.append(element_sum)

    #print("eeeeeeeeee\n", SS_f_OD_dy_b_t_delta_OD) 
    value1 = (SS_f_OD_dy_b_t_delta_OD[0] + SS_f_OD_dy_b_t_delta_OD[1] + SS_f_OD_dy_b_t_delta_OD[2]) /3
    value2 = (SS_f_OD_dy_b_t_delta_OD[3] + SS_f_OD_dy_b_t_delta_OD[4] + SS_f_OD_dy_b_t_delta_OD[5]) /3
    value3 = (SS_f_OD_dy_b_t_delta_OD[6] + SS_f_OD_dy_b_t_delta_OD[7] + SS_f_OD_dy_b_t_delta_OD[8]) /3
    value4 = (SS_f_OD_dy_b_t_delta_OD[9] + SS_f_OD_dy_b_t_delta_OD[10] + SS_f_OD_dy_b_t_delta_OD[11]) /3
    value5 = (SS_f_OD_dy_b_t_delta_OD[12] + SS_f_OD_dy_b_t_delta_OD[13] + SS_f_OD_dy_b_t_delta_OD[14]) /3
    value6 = (SS_f_OD_dy_b_t_delta_OD[15] + SS_f_OD_dy_b_t_delta_OD[16] + SS_f_OD_dy_b_t_delta_OD[17]) /3
    value7 = (SS_f_OD_dy_b_t_delta_OD[18] + SS_f_OD_dy_b_t_delta_OD[19] + SS_f_OD_dy_b_t_delta_OD[20]) /3
    value8 = (SS_f_OD_dy_b_t_delta_OD[21] + SS_f_OD_dy_b_t_delta_OD[22] + SS_f_OD_dy_b_t_delta_OD[23]) /3
    SS_f_OD_dy_b_t_delta_OD_ = [(value1+value5)/2, (value2+value6)/2, (value3+value7)/2, (value4+value8)/2]
    #print(SS_f_OD_dy_b_t_delta_OD_)
    sum45 = sum(SS_f_OD_dy_b_t_delta_OD_)
    #print(sum45/4)
    return sum45/4
#______________________________________________________________________________________
#46
#στην ουσια πρεπει τουλ 1 σταθμος φορτισης
def check_inequality_46(x_stn_Ch):
    N_B = calculate_N_B(x_stn_Ch)
    if N_B < 1:
        return False
    else:
        return True

#______________________________________________________________________________________
#47
#OF4_ = 1/OF4(x_stn_Ch)    #apo maximize se minimize
#OF = [OF1, OF2, OF3, OF4_]
#print(OF4_)
#print(OF)

#_____________________________________________
#48
def OF_norm(OF, OF_min, OF_max):
    OF_norm = []
    for i in range(len(OF)):
        OF_norm.append((OF[i] - OF_min[i]) / (OF_max[i] - OF_min[i]))
    return OF_norm
#OF_norm = OF_norm(OF)

#print(OF)
#print(OF_norm)
#__________________________________________________________________
#50
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
x1=-5
x2=1
x3=3
X = [x1,x2,x3]
sig_X = sigmoid(sum(X)/4) #dia 4 ola einai minimize

#print(sig_X)

#__________________________________________________________________
#49
def P_l(X):
    return 1 if sigmoid(sum(X)/4) > np.random.randint(2) else 0
#print(P_l)

#__________________________________________________________________
#51
def mi_func(fitness_values_ofthebest, OF_min, OF_max):
    mi_f_list = []
    for i in range(len(fitness_values_ofthebest)):
        if(fitness_values_ofthebest[i] > OF_max[i]):
            mi_f_list.append(0)
        if(fitness_values_ofthebest[i] < OF_min[i]):
            mi_f_list.append(1)
        if(OF_min[i] <= fitness_values_ofthebest[i] <= OF_max[i]):
            mi_f_list.append((OF_max[i] - fitness_values_ofthebest[i])/(OF_max[i] - OF_min[i]))
    return mi_f_list

#______________________________________________________________________________________________
#52
def mi_f(mi_f_list):
    mi_f = 1
    for i in range(len(mi_f_list)):
        mi_f *= mi_f_list[i]
    return mi_f

#______________________________________________________

#fitness function
# Συνάρτηση fitness
def calculate_fitness(individual):
    fitness_values = []
    OF_values = []
    OF1_values =[]
    OF2_values = []
    OF3_values = []
    OF4_values = []
    P_loss_seasons_values = []
    fores=0
    for i in range(len(individual)):
        x_stn_Ch = individual[i][0]
        x_stn_PV = x_stn_Ch
        y_stn_Ch = individual[i][1]
        y_stn_PV = individual[i][2]
        #print(x_stn_Ch)
        #print(x_stn_PV)
        #print(y_stn_Ch)
        #print(y_stn_PV)
        # Εξίσωση OF1
        if not all(elem==0 for elem in x_stn_Ch):
            # Εξίσωση OF1
            OF1_ = OF1(x_stn_Ch,x_stn_PV,y_stn_Ch,y_stn_PV) # Υπολογισμός του πρώτου παράγοντα OF1
            
            # Εξίσωση OF2
            OF2_ = OF2_minV_maxV_(x_stn_Ch)[0] # Υπολογισμός του παράγοντα OF2
            P_loss_seasons = OF2_minV_maxV_(x_stn_Ch)[3]

            # Εξίσωση OF3
            OF3_ = OF3(x_stn_Ch) # Υπολογισμός του παράγοντα OF3
            
            # Εξίσωση OF4
            OF4_ = OF4(x_stn_Ch) # Υπολογισμός του παράγοντα OF4 - !!!=maximize to minimize
            
            # Υπολογισμός της fitness ως γενικό άθροισμα των εξισώσεων
            OF = [OF1_, OF2_, OF3_, OF4_]
            OF_values.append(OF)
            OF1_values.append(OF1_)
            OF2_values.append(OF2_)
            OF3_values.append(OF3_)
            OF4_values.append(OF4_)
            P_loss_seasons_values.append(P_loss_seasons)

            #fitness = OF_norm[0] + OF_norm[1] + OF_norm[2] + OF_norm[3]
        else:
            continue
    
    OF_min = [min(OF1_values), min(OF2_values), min(OF3_values), min(OF4_values)]
    OF_max = [max(OF1_values), max(OF2_values), max(OF3_values), max(OF4_values)]
    #print(OF_values)
    #print(OF_min)
    #print(OF_max)
    for i in range(len(OF_values)):
        #print(OF_values[i])
        fitness = OF_norm(OF_values[i], OF_min, OF_max)
        fitness_values.append(fitness)
    return fitness_values, OF_values, P_loss_seasons_values, OF_min, OF_max


##############################################################################################################################################################################################################################
##############################################################################################################################################################################################################################
##############################################################################################################################################################################################################################
##############################################################################################################################################################################################################################

solution = [[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 10.0, 0.0, 0.0], [0.0, 0.0, 183.5, 0.0, 0.0, 0.0, 160.4, 0.0, 0.0, 220.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 9, 0.0, 0.0, 0.0, 14, 0.0, 13.0, 0.0, 0.0, 10], [0.0, 0.0, 135.8, 0.0, 0.0, 0.0, 149.0, 0.0, 140.6, 0.0, 0.0, 83.1]],
            [[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0], [0.0, 9.0, 0.0, 12.0, 0.0, 0.0, 15.0, 0.0, 7.0, 0.0, 0.0, 9], [0.0, 62.4, 0.0, 146.8, 0.0, 0.0, 217.5, 0.0, 150.4, 0.0, 0.0, 224.3]],
            [[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 6, 0.0, 0.0, 0.0, 8.0, 0.0, 15.0, 0.0, 0.0, 13.0, 0.0], [0.0, 149.2, 0.0, 0.0, 0.0, 175.6, 0.0, 217.3, 0.0, 0.0, 215.9, 0.0]],
            [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [11.0, 0.0, 6, 0.0, 0.0, 0.0, 11.0, 0.0, 8.0, 0.0, 8.0, 0.0], [156.5, 0.0, 71.6, 0.0, 0.0, 0.0, 135.9, 0.0, 177.5, 0.0, 75.9, 0.0]],
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 6.0, 0.0, 9, 0.0, 8.0, 0.0], [0.0, 0.0, 183.5, 0.0, 0.0, 0.0, 214.3, 0.0, 154.6, 0.0, 144.8, 0.0]],
            [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 8, 0.0, 0.0, 13.0, 0.0, 0.0, 9.0, 0.0, 10.0, 0.0], [0.0, 0.0, 190.5, 0.0, 0.0, 214.8, 0.0, 0.0, 212.5, 0.0, 147.0, 0.0]],
            [[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0], [0.0, 13.0, 0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 9.0, 0.0, 0.0, 11], [0.0, 179.6, 0.0, 137.1, 0.0, 0.0, 202.8, 0.0, 214.8, 0.0, 0.0, 195.0]],
            [[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], [10.0, 0.0, 7, 0.0, 0.0, 14.0, 0.0, 0.0, 14, 0.0, 0.0, 11], [154.9, 0.0, 210.5, 0.0, 0.0, 160.5, 0.0, 0.0, 221.9, 0.0, 0.0, 179.4]],
            [[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 9, 0.0, 0.0, 0.0, 6.0, 0.0, 14.0, 0.0, 0.0, 15.0, 0.0], [0.0, 170.8, 0.0, 0.0, 0.0, 228.8, 0.0, 128.0, 0.0, 0.0, 224.4, 0.0]],
            [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [13, 0.0, 12, 0.0, 0.0, 0.0, 12, 0.0, 10.0, 0.0, 12, 0.0], [149.3, 0.0, 162.6, 0.0, 0.0, 0.0, 194.0, 0.0, 121.9, 0.0, 183.5, 0.0]],
            [[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 14, 0.0, 0.0, 0.0, 8.0, 0.0, 12, 0.0, 0.0, 13.0, 0.0], [0.0, 216.2, 0.0, 0.0, 0.0, 201.5, 0.0, 114.1, 0.0, 0.0, 108.6, 0.0]],
            [[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], [7.0, 0.0, 12.0, 0.0, 0.0, 6.0, 0.0, 13, 0.0, 0.0, 10.0, 0.0], [123.7, 0.0, 170.2, 0.0, 0.0, 195.2, 0.0, 79.2, 0.0, 0.0, 152.4, 0.0]],
            [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0], [7.0, 0.0, 9, 0.0, 0.0, 0.0, 10, 0.0, 15, 0.0, 0.0, 12], [167.0, 0.0, 108.8, 0.0, 0.0, 0.0, 196.2, 0.0, 133.4, 0.0, 0.0, 205.3]],
            [[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 12.0, 0.0, 15, 0.0, 0.0, 8, 0.0, 11, 0.0, 8.0, 0.0], [0.0, 187.9, 0.0, 157.0, 0.0, 0.0, 228.8, 0.0, 157.4, 0.0, 229.6, 0.0]],
            [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 8.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0], [0.0, 0.0, 122.3, 0.0, 0.0, 187.0, 0.0, 0.0, 0.0, 223.7, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 13, 0.0, 0.0, 0.0, 8.0, 0.0, 10.0, 0.0, 12, 0.0], [0.0, 0.0, 185.4, 0.0, 0.0, 0.0, 164.7, 0.0, 147.8, 0.0, 214.7, 0.0]],
            [[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 15.0, 0.0, 14.0, 0.0, 0.0, 12.0, 15, 0.0, 0.0, 6, 0.0], [0.0, 179.2, 0.0, 125.4, 0.0, 0.0, 219.9, 57.2, 0.0, 0.0, 97.6, 0.0]],
            [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 14, 0.0, 0.0, 15, 0.0, 8.0, 0.0, 0.0, 0.0, 11], [0.0, 0.0, 138.1, 0.0, 0.0, 222.4, 0.0, 147.6, 0.0, 0.0, 0.0, 149.8]],
            [[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 6.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 12.0, 0.0, 7, 0.0], [0.0, 209.7, 0.0, 0.0, 0.0, 204.4, 0.0, 0.0, 155.5, 0.0, 188.8, 0.0]],
            [[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 7.0, 0.0, 0.0, 0.0, 6.0, 0.0, 13, 0.0, 0.0, 6, 0.0], [0.0, 231.2, 0.0, 0.0, 0.0, 230.2, 0.0, 82.6, 0.0, 0.0, 136.3, 0.0]],
            [[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [6.0, 0.0, 13.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 11, 0.0, 0.0], [154.1, 0.0, 79.4, 0.0, 0.0, 231.5, 0.0, 0.0, 0.0, 212.4, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 10, 0.0, 0.0, 13, 0.0, 7, 0.0, 0.0, 0.0, 9.0], [0.0, 0.0, 236.2, 0.0, 0.0, 225.8, 0.0, 177.1, 0.0, 0.0, 0.0, 179.2]],
            [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [8.0, 0.0, 7.0, 0.0, 0.0, 0.0, 12.0, 0.0, 9, 0.0, 6.0, 0.0], [191.2, 0.0, 219.4, 0.0, 0.0, 0.0, 62.7, 0.0, 148.2, 0.0, 220.3, 0.0]],
            [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 8.0, 0.0, 0.0, 7, 0.0, 8.0, 0.0, 0.0, 0.0, 11], [0.0, 0.0, 105.4, 0.0, 0.0, 197.3, 0.0, 141.8, 0.0, 0.0, 0.0, 202.5]],
            [[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [11.0, 0.0, 7.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 9, 0.0, 0.0], [77.8, 0.0, 215.7, 0.0, 0.0, 210.1, 0.0, 0.0, 0.0, 119.1, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 6, 0.0, 0.0, 8.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 12.0], [0.0, 237.0, 0.0, 0.0, 187.9, 0.0, 0.0, 144.3, 0.0, 0.0, 0.0, 211.8]],
            [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 11.0, 0.0, 0.0, 9.0, 0.0, 14, 0.0, 0.0, 8, 0.0], [0.0, 0.0, 203.4, 0.0, 0.0, 190.8, 0.0, 161.1, 0.0, 0.0, 224.3, 0.0]],
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 7, 0.0, 0.0, 10, 0.0, 0.0], [0.0, 0.0, 134.8, 0.0, 0.0, 0.0, 223.5, 0.0, 0.0, 201.7, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 13, 0.0, 9.0, 0.0, 8.0, 0.0], [0.0, 0.0, 205.9, 0.0, 0.0, 0.0, 61.9, 0.0, 205.4, 0.0, 56.1, 0.0]]]


fit_norm = calculate_fitness(solution)[0]
fit = calculate_fitness(solution)[1]
p_loss_seasons = calculate_fitness(solution)[2]

print("\n\nPloss winter, spring, summer, automn\n", p_loss_seasons)
print("\n\nOF1, OF2, OF3, OF4 normalized values\n", fit_norm)
print("\n\nOF1, OF2, OF3, OF4 values\n", fit)

OF1_values_norm = []
OF2_values_norm = []
OF3_values_norm = []
OF4_values_norm = []
for i in range(len(fit_norm)):
    OF1_values_norm.append(fit_norm[i][0])
    OF2_values_norm.append(fit_norm[i][1])
    OF3_values_norm.append(fit_norm[i][2])
    OF4_values_norm.append(fit_norm[i][3])

OF_min_norm = [min(OF1_values_norm), min(OF2_values_norm), min(OF3_values_norm), min(OF4_values_norm)]
OF_max_norm = [max(OF1_values_norm), max(OF2_values_norm), max(OF3_values_norm), max(OF4_values_norm)]

mi_f_values = []
for i in range(len(fit_norm)):
    #print(fit_norm[i])
    mi_func_list = mi_func(fit_norm[i], OF_min_norm, OF_max_norm)
    mi_f_values.append(mi_f(mi_func_list))

print("\n\nmi_f values\n", mi_f_values)

def sort_solution(solution, mi_f_values):
    combined = list(zip(solution, mi_f_values))
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    #print(sorted_combined)
    formatted_sorted_solution = []
    for item in sorted_combined:
        formatted_sorted_solution.append({'mi_f': item[1], 'solution': item[0]})
    
    return formatted_sorted_solution

sorted_solution_by_mi_f = sort_solution(solution, mi_f_values)

print("\n\nSorted solution by mi_f\n")
for item in sorted_solution_by_mi_f:
    print(f'mi_f: {item["mi_f"]}, solution: {item["solution"]}\n')
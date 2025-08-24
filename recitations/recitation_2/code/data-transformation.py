"""
Created on Mon Aug 18 11:26:00 2025
# Combine the previously generated datasets from the R
# library `mlogit` for choice analysis
@adapted-from: https://github.com/cjsyzwsh/ASU-DNN/blob/master/code/0_combine_R_datasets.py
"""
 
import pandas as pd
import pickle

# List of dataset names (without file extensions)
data_list = [
    'Beef_long',
    'Car_wide',
    'Catsup_wide',
    'Cracker_wide',
    'Electricity_wide',
    'Fishing_wide',
    'HC_wide',
    'Heating_wide',
    'Ketchup_wide',
    'MobilePhones_long',
    'Mode_wide',
    'ModeCanada_long',
    'Telephone_long',
    'TollRoad_wide',
    'Train_wide',
    'Tuna_wide'
]

# Dictionary to hold all loaded datasets
data_dic = {}

# Load each CSV file into the dictionary
for name in data_list:
    file_path = f"../data/mlogit_{name}.csv"
    # Read the CSV file; assumes the first column is the index
    data_dic[name] = pd.read_csv(file_path, index_col=0)

# Save the combined dictionary as a pickle file for later use
output_pickle = '../data/mlogit_choice_data.pickle'
with open(output_pickle, 'wb') as data:
    pickle.dump(data_dic, data, protocol=pickle.HIGHEST_PROTOCOL)

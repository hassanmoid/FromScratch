import pandas as pd
import numpy as np
import sys
import os


#Global variables
base_data_loc = "../data/regression"
data_name = sys.argv[1].lower()
category_col_list = sys.argv[2].split(",")

def read_data(file_loc):
    df = pd.read_csv(file_loc)
    return df

def generate_file_location(name,base_data_loc=base_data_loc):
    return os.path.join(base_data_loc, name)


data_loc_dict = {'fish':generate_file_location("fish.csv"), 'insurance':generate_file_location("insurance.csv"),'real estate':generate_file_location("real_estate.csv"), 'physical activity':generate_file_location("physical_activity_obesity.csv")}

df_inp = read_data(data_loc_dict[data_name])
print(df_inp.shape)
print(df_inp.head())

#dropping the category columns from input data for simplicity
df_inp.drop(category_col_list,axis=1,inplace=True)

print("DF after dropping categorical columns")
print(df_inp.head())
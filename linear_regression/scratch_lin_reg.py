import pandas as pd
import numpy as np
import sys

def read_data(file_loc):
    df = pd.read_csv(file_loc)
    return df

data_loc_dict = {'fish':'../data/lin_reg/fish.csv', 'insurance':'../data/lin_reg/insurance.csv','real estate':'../data/lin_reg/real_estate.csv', 'physical activity':'../data/lin_reg/physical_activity_obesity.csv'}

data_name = sys.argv[1].lower()


df_inp = read_data(data_loc_dict[data_name])

print(df_inp.shape)

print(df_inp.head())
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
np.random.seed(0)

from src.regression_module import MultipleLinearRegression
from src.utils import utils

base_data_loc = "../data/regression"

# Function to retrieve the data location using name of data
def generate_file_location(name,base_data_loc=base_data_loc):
    return os.path.join(base_data_loc, name)

data_loc_dict = {'fish':generate_file_location("fish.csv"), 'insurance':generate_file_location("insurance.csv"),'real estate':generate_file_location("real_estate.csv"), 'physical activity':generate_file_location("physical_activity_obesity.csv")}

#User input
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', dest="data_name", required=True)
parser.add_argument('--category_cols', dest="category_cols", required=True)
parser.add_argument('--target_col', dest="target_col", required=True)

args = parser.parse_args()
args_dict = vars(args)
print("Args dict: ", args_dict)
data_name = args_dict["data_name"]
category_col_list = args_dict["category_cols"].split(",")
target_col_name = args_dict["target_col"]
print("category_col_list is - ", category_col_list)
print(f"target column for this data is: ", target_col_name)

#Generating an object from utils module
utils_obj = utils(data_name, data_loc_dict, category_col_list, target_col_name)

#Reading data
df_inp = utils_obj.read_data()
print(df_inp.shape)
print(df_inp.head())

#dropping the category columns from input data for simplicity and moving the target column to last column of dataframe
df_final = utils_obj.drop_category_cols(df_inp)
print(df_final.head())

#Splitting the data into development and validation and separating the X and y for development and validation data
x_dev, y_dev, x_val, y_val = utils_obj.dev_val_split(df_final, 0.7)
print(f"shape of dev data - independent variable - {x_dev.shape}, dependent variable - {y_dev.shape}")
print(f"shape of val data - independent variable - {x_val.shape}, dependent variable - {y_val.shape}")

#Standardization (bringing the data on a scale of [0,1] using min-max scaling for smooth cost function curve)
standardize_x_dev, standardize_x_val = utils_obj.preprocessing_dev_val(x_dev, x_val,"standard")
print(f"shape of dev data - independent variable - {standardize_x_dev.shape}, dependent variable - {y_dev.shape}")
print(f"shape of val data - independent variable - {standardize_x_val.shape}, dependent variable - {y_val.shape}")
print("checking the preprocessing of standardization")
print("For dev:")
print(np.mean(standardize_x_dev,axis=0))
print(np.std(standardize_x_dev,axis=0))
print("For val:")
print(np.mean(standardize_x_val,axis=0))
print(np.std(standardize_x_val,axis=0))
print("\n")

#Normalization (Mean centering (mu~0) with unit standard deviation for smooth cost function curve)
normal_x_dev, normal_x_val = utils_obj.preprocessing_dev_val(x_dev, x_val,"normal")
print(f"shape of dev data - independent variable - {normal_x_dev.shape}, dependent variable - {y_dev.shape}")
print(f"shape of val data - independent variable - {normal_x_val.shape}, dependent variable - {y_val.shape}")
print("checking the preprocessing of normalization")
print("For dev:")
print(np.min(normal_x_dev,axis=0))
print(np.max(normal_x_dev,axis=0))
print("For val:")
print(np.min(normal_x_val,axis=0))
print(np.max(normal_x_val,axis=0))

'''
Steps to minimize the cost function of linear regression model
1. Initialize weights and bias_term as random
2. Forward propagate the input to output(prediction) using linear function and weights, bias term from step 1.
3. Compute cost using the prediction from step2.
4. Using the predictions from step2 backward propagate the error using gradient descent updating weights and bias term.
5. Repeat steps 2 top steps 4 for n number of iterations
'''

np.random.seed(42)
num_iterations = 500
learning_rate = 0.01

rows, columns = standardize_x_dev.shape
weights = np.random.rand(columns)
bias_term = np.random.rand()
lr = MultipleLinearRegression(learning_rate, num_iterations, standardize_x_dev, y_dev)

cost_list = []
for i in range(num_iterations):
    y_pred = lr.forward_propagation(weights, bias_term)
    cost = lr.compute_cost(y_dev, y_pred)
    cost_list.append(cost)
    weights, bias_term = lr.backward_propagation(y_pred, standardize_x_dev, y_dev, weights, bias_term)


df_cost = pd.DataFrame(data=cost_list, columns=["cost"]).reset_index()
df_cost.columns = ["epochs", "cost"]
df_cost["epochs"] = df_cost["epochs"] + 1
print(df_cost.tail())

#Plotting the cost as it goes down with increase in iterations
plot1 = df_cost.plot(x="epochs",y="cost",title="Epochs vs cost",figsize=(10,10),colormap="summer", marker=".")
plot1.get_figure().savefig("plots/EpochsVsCost_standardization.png")

'''
Printing the peformance of the linear regression model using metrics such as:
1. r_square
2. mean square error
3. mean absolute error
4. symmetric mean absolute percentage error
5. median absolute error
'''
y_dev_pred = lr.predict(standardize_x_dev, weights, bias_term)
y_val_pred = lr.predict(standardize_x_val, weights, bias_term)

print("training data metrics:")
r_square, mse, mae, smape, med_ae = utils_obj.linear_regression_metrics(y_dev, y_dev_pred)
print("r-square:", r_square)
print("mean square error:", mse)
print("mean absolute error:", mae)
print("SMAPE", smape)
print("median absolute error:", med_ae)

print("\n")

print("test data metrics:")
r_square, mse, mae, smape, med_ae = utils_obj.linear_regression_metrics(y_val, y_val_pred)
print("r-square:", r_square)
print("mean square error:", mse)
print("mean absolute error:", mae)
print("SMAPE", smape)
print("median absolute error:", med_ae)

print("======================================================================================================================================")
print("\n Running Linear Regression when the preprocessing is normalization")
#minimize the cost function of linear regression model
np.random.seed(42)
num_iterations = 2000
learning_rate = 0.01

rows, columns = normal_x_dev.shape
weights = np.random.rand(columns)
bias_term = np.random.rand()
lr = MultipleLinearRegression(learning_rate, num_iterations, normal_x_dev, y_dev)

cost_list = []
for i in range(num_iterations):
    y_pred = lr.forward_propagation(weights, bias_term)
    cost = lr.compute_cost(y_dev, y_pred)
    cost_list.append(cost)
    weights, bias_term = lr.backward_propagation(y_pred, normal_x_dev, y_dev, weights, bias_term)


df_cost = pd.DataFrame(data=cost_list, columns=["cost"]).reset_index()
df_cost.columns = ["epochs", "cost"]
df_cost["epochs"] = df_cost["epochs"] + 1
df_cost.tail()

#Plotting the cost as it goes down with increase in iterations
plot1 = df_cost.plot(x="epochs",y="cost",title="Epochs vs cost",figsize=(10,10),colormap="summer", marker=".")
plot1.get_figure().savefig("plots/EpochsVsCost_normalization.png")

#Printing the peformance of the linear regression model
y_dev_pred = lr.predict(normal_x_dev, weights, bias_term)
y_val_pred = lr.predict(normal_x_val, weights, bias_term)

print("training metrics:")
r_square, mse, mae, smape, med_ae = utils_obj.linear_regression_metrics(y_dev, y_dev_pred)
print("r-square:", r_square)
print("mean square error:", mse)
print("mean absolute error:", mae)
print("SMAPE", smape)
print("median absolute error:", med_ae)

print("\n")

print("test data metrics:")
r_square, mse, mae, smape, med_ae = utils_obj.linear_regression_metrics(y_val, y_val_pred)
print("r-square:", r_square)
print("mean square error:", mse)
print("mean absolute error:", mae)
print("SMAPE", smape)
print("median absolute error:", med_ae)


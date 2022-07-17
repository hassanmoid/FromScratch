import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,median_absolute_error

class utils:

    
    def __init__(self, name, data_loc_dict, category_col_list, target_col_name) -> None:
        self.file_name = name
        self.category_col_list = category_col_list
        self.data_loc_dict = data_loc_dict
        self.target_col_name = target_col_name
    
    def read_data(self):
        df = pd.read_csv(self.data_loc_dict[self.file_name])
        
        return df

    
    def drop_category_cols(self, df):
        if len(self.category_col_list) > 0 and len(self.category_col_list[0])>0:
            print(f"DF after dropping {self.category_col_list} categorical columns")
            df_inp1 = df.drop(self.category_col_list,axis=1)

        df_target = df_inp1[self.target_col_name]
        df_final = df_inp1.drop([self.target_col_name],axis=1)
        df_final[self.target_col_name] = df_target
        
        return df_final

    
    def dev_val_split(self, df, split_ratio):
        dev = []
        val = []
        np.random.seed(0)
        inp_values = df.values
        for  i in range(len(inp_values)):
            if np.random.rand() < split_ratio:
                dev.append(inp_values[i])
            else:
                val.append(inp_values[i])
        dev = np.array(dev)
        val = np.array(val)

        print(f"dev data shape {dev.shape}")
        print(f"val data shape {val.shape}")

        x_dev = dev[:,0:-1]
        y_dev = dev[:,-1]
        x_val = val[:,0:-1]
        y_val = val[:,-1]

        return x_dev, y_dev, x_val, y_val


    def preprocessing_dev_val(self, dev_x, val_x, preprocessing_type):
        if preprocessing_type =="standard":
            print("Standardization preprocessing initiated:")
            mean_list = np.mean(dev_x, axis=0)
            std_list = np.std(dev_x,axis=0)
            standard_dev = (dev_x - mean_list) / std_list
            standard_val = (val_x - mean_list) / std_list
            return standard_dev, standard_val
        else:
            print("Normaliztion preprocessing initiated:")
            min_list = np.min(dev_x,axis=0)
            max_list = np.max(dev_x,axis=0)
            normal_dev = (dev_x - min_list) / (max_list - min_list)
            normal_val = (val_x - min_list) / (max_list - min_list)
            
            return normal_dev, normal_val

    
    def smape_error(self,y_true, y_pred):
        
        return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

    
    def linear_regression_metrics(self,y_true, y_pred):
        r_square = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        smape = self.smape_error(y_true, y_pred)
        med_ae = median_absolute_error(y_true, y_pred)

        return r_square, mse, mae, smape, med_ae
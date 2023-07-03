import time
import concurrent.futures

import pandas as pd
import numpy as np
import os
import datetime
import seaborn as sns
import scipy as sp
from sklearn.metrics import r2_score


class ReduceGrid:
    def __init__(self,parameter_dict:dict,\
                 X_train,\
                 y_train,\
                 X_test,\
                 y_test,\
                 model_obj:object):
        self.parameter_dict = parameter_dict
        self.X_train = X_train 
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_obj = model_obj
        self.performance = {'test_r2':[],'train_r2':[],'params':[]}
        
    def parameters_calc(self):
        parameter_values_list = [v for k,v in self.parameter_dict.items()]
        parameter_permutations = list(itertools.product(*parameter_values_list))
        return parameter_permutations,len(parameter_permutations)
    
    def create_parameters(self,single_permutation):
        params = {k:single_permutation[v] for k,v in zip(self.parameter_dict.keys(),range(0,len(self.parameter_dict.keys())))}
        return params
    
    def train_model_obj(self,single_permutation):
        params = self.create_parameters(single_permutation)
        reg = self.model_obj(**params)
        reg.fit(self.X_train,self.y_train)
        test_preds = reg.predict(np.array(self.X_test))
        train_preds = reg.predict(np.array(self.X_train))
        return train_preds,test_preds
    
    def assess_model(self,single_permutation):
        train_preds, test_preds = self.train_model_obj(single_permutation)
        test_rsquared = r2_score(self.y_test,test_preds)
        train_rsquared = r2_score(self.y_train,train_preds)
        self.performance['test_r2'].append(test_rsquared)
        self.performance['train_r2'].append(train_rsquared)
        self.performance['params'].append(i)
        
    def sequential_grid(self):
        parameter_permutations,parameter_amount = self.parameters_calc()
        for i in parameter_permutations:
            self.assess_model(i)
        return self.performance
                
    def parallelize_grid(self):
        parameter_permutations,parameter_amount = self.parameters_calc()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self.assess_model,parameter_permutations)
        return self.performance
        
        
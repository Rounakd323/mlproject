import os 
import sys 

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.metrics import r2_score


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb")as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report={}
        for i in range (len(list(models))):
            model=list(models.values())[i]
            model.fit(x_train,y_train)

            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)

            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)
            # model_train_mae,model_train_mse,model_train_rmse,model_train_r2=evaluate_model(y_train,y_train_pred)
            # model_test_mae,model_test_mse,model_test_rmse,model_test_r2=evaluate_model(y_test,y_test_pred)
            report[list(models.keys())[i]]=test_model_score
            # print(list(models.keys())[i])
            # model_list.append(list(models.keys())[i])
        return report

    except Exception as e:
        pass
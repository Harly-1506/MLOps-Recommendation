import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models, optimizer, loss):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            '''
            for sklean and finding best param for models !!!!
            # para=param[list(models.keys())[i]]
            # gs = GridSearchCV(model,para,cv=3)
            # gs.fit(X_train,y_train)
            # model.set_params(**gs.best_params_)
            '''
            # optimizer.build(model.trainable_variables)
            model.compile(optimizer=optimizer, loss=loss)
            # model.fit(X_train,y_train)
            model.fit([X_train['user_id'], X_train['product_id']], X_train['rating'],validation_split = 0.2,
#           validation_data = [[test_df['user_id'], test_df['user_id']], test_df['rating']], 
          epochs=1, batch_size=48, verbose=1)
            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict([X_train['user_id'], X_train['product_id']])
            y_test_pred = model.predict([X_test['user_id'], X_test['product_id']])

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
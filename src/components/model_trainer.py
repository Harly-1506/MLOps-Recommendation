import os
import sys
from dataclasses import dataclass
from src.models import *

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras 
import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            print(X_train.shape)
            models = {
                "CB_model":CB_model(),
                "CF_model": CF_model(X_train, y_train)
            }

            # init param for models
            initial_learning_rate = 0.01
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=100,
                decay_rate=0.96,
                staircase=True)
            adam = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
            loss = tf.losses.MeanAbsoluteError()
            

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models, optimizer = adam, loss = loss)
            
            ## To get best model score from dict
            best_model_score = min(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # if best_model_score < 1:
            #     raise CustomException("No best model found")
            # logging.info(f"Best found model on both training and testing dataset")
            best_model.save("artifacts/model_data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict([X_test[:, 0], X_test[:, 1]])

            r2_square = mean_squared_error(y_test, predicted)
            return r2_square
            

        except Exception as e:
            raise CustomException(e,sys)
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
import tensorflow as tf
import numpy as np

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=("artifacts/testmodel")
            preprocessor_path=("artifacts\proprocessor.pkl")
            print("Before Loading")
            # model=load_object(file_path=model_path)
            model = tf.keras.models.load_model(model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict([data_scaled[:,0],data_scaled[:,1]])
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

    def recommend_products(self,model,user_id, pred, num_recommendations = 5):

        all_products = np.unique(train['product_id'])
        
        user_input = np.array([user_id] * len(all_products))
        product_input = all_products
        predicted_ratings = model.predict([user_input, product_input])
        
        recommended_indices = np.argsort(predicted_ratings, axis=0)[-num_recommendations:][::-1]
        
        recommended_products = all_products[recommended_indices]
        
        return recommended_products

        pass

class CustomData:
    def __init__(  self,
        user_id: str,
        product_id: str,):

        self.user_id = user_id

        self.product_id = product_id


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "user_id": [self.user_id],
                "product_id": [self.product_id],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
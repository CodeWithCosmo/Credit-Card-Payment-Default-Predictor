import os
import sys
import dill
from py_dotenv import dotenv
from pymongo import MongoClient

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging as lg
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = list(params.values())[i]

            gs = GridSearchCV(model,param,cv=5) 
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)          

            y_test_pred = model.predict(X_test)

            test_model_score = accuracy_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys) 
    
        
def read_mongo():
    try:
            lg.info('Connecting to MongoDB Cloud')
            dotenv.read_dotenv('.env')
            client = os.getenv('client')
            lg.info('Connection successful')
            
            database = os.getenv('database')
            collection = os.getenv('collection')            

            client = MongoClient(client)
            db = client[database]
            collection = db[collection]
            cursor = collection.find({}) 
            data = list(cursor)            
            return data
    
    except Exception as e:
        raise CustomException(e,sys)
        
    finally:
        client.close()
def write_mongo(data):
    try:
            lg.info('Connecting to MongoDB Cloud')
            dotenv.read_dotenv('.env')
            client = os.getenv('client')
            lg.info('Connection successful')
            
            database = os.getenv('database')
            collection = os.getenv('collection')            

            client = MongoClient(client)
            db = client[database]
            collection = db[collection]
            lg.info('Inserting data into MongoDB Cloud')            
            data['_id'] = range(1, len(data) + 1)
            data = data.to_dict(orient='records')
            collection.insert_many(data)
            lg.info('Data insertion successful')                       
    
    except Exception as e:
        raise CustomException(e,sys)
    
    finally:
        client.close() 
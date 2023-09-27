import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from src.logger import logging as lg
from src.exception import CustomException
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join('artifacts','model.pkl')

class CustomModel:
    def __init__(self, preprocessing_object, trained_model_object):
        self.preprocessing_object = preprocessing_object

        self.trained_model_object = trained_model_object

    def predict(self, X):
        transformed_feature = self.preprocessing_object.transform(X)

        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],train_array[:,-1],
                test_array[:,:-1],test_array[:,-1]
                )

            models = {
                "decision_tree_classifier": DecisionTreeClassifier(),
                "random_forest_classifier": RandomForestClassifier(),
                "gradient_boosting_classifier": GradientBoostingClassifier(),
                "logistic_regression": LogisticRegression(),
                "xgboost_classifier": XGBClassifier(),
                "catboost_classifier": CatBoostClassifier(verbose=False),
                "adaboost_classifier": AdaBoostClassifier(),
                "knn_classifier": KNeighborsClassifier(),
                "svm_classifier": SVC(),
                "extra_trees_classifier": ExtraTreesClassifier()
            }

            params={
                "decision_tree_classifier": {
                    # 'criterion':['gini','entropy','log_loss'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2', None],
                },
                "random_forest_classifier":{
                    # 'criterion':['gini','entropy','log_loss'],                 
                    # 'max_features':['sqrt','log2', None],
                    # 'n_estimators': [8,16,32]
                },
                "gradient_boosting_classifier":{
                    # 'loss':['log_loss','exponential'],
                    # 'learning_rate':[.1,.01,.001],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['sqrt','log2', None],
                    # 'n_estimators': [8,16,32,64]
                },
                "logistic_regression":{
                    # 'penalty':['l1','l2','elasticnet',None],
                    # 'solver':['newton-cg','lbfgs','liblinear','sag','saga'],
                    # 'max_iter':[100,500,1000]
                },          
                "xgboost_classifier":{
                    # 'learning_rate':[.1,.01,.001],
                    # 'n_estimators': [8,16,32,64]
                },
                "catboost_classifier":{
                    # 'depth': [6,8,10],
                    # 'learning_rate': [.1,.01,.001],
                    # # 'iterations': [30, 50, 100]
                },
                "adaboost_classifier":{
                    # 'learning_rate':[.1,.01,.001],
                    # 'algorithm':['SAMME','SAMME.R'],
                    # 'n_estimators': [8,16,32,64]
                },
                "knn_classifier" : {
                    # "n_neighbors": [1, 3, 5],
                    # "weights": ["uniform", "distance"],
                    # "algorithm": ["auto", "kd_tree", "ball_tree"]
                },
                "svm_classifier":{
                    # # "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
                    # "degree": [1, 2, 3],
                    # "gamma": ["scale", "auto"],
                    
                },
                "extra_trees_classifier":{
                    # "criterion":['gini','entropy','log_loss'],
                    # "n_estimators": [8,16,32,64]
                }
            }       

            lg.info('Hyperparameter tuning initiated')
            lg.info('Initiating model trainer and model evaluation')
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models, params)
            lg.info('Hyperparameter tuning completed')
            lg.info('Model training and evaluation completed')

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.8:
                raise Exception('No best model found')
            lg.info('Best model found')

            lg.info('Saving best model')
            save_object(self.model_trainer_config.trained_model_path,best_model)
            lg.info('Best model saved')
            
            prediction = best_model.predict(X_test)
            accuracy = accuracy_score(y_test,prediction)

            return accuracy            
        
        except Exception as e:
            raise CustomException(e, sys)
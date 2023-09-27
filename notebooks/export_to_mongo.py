import sys
import pandas as pd
from src.exception import CustomException
from src.utils import write_mongo

try:
    data = pd.read_csv('https://raw.githubusercontent.com/sunnysavita10/credit_card_pw_hindi/main/creditCardFraud_28011964_120214.csv')
    write_mongo(data)

except Exception as e:
    raise CustomException(e, sys)
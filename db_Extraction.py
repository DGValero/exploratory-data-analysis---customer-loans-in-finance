''' 
Script to extract the loan and payment data stored in an AWS RDS database.
The methods below extract the data from this database using confidential credentials.
The data is saved locally as a .csv file to increase performance during the posterior analyis. 
'''
#%%
import yaml
import pandas as pd
from sqlalchemy import create_engine


class RDSDatabaseConnector:
    #Method that initialises a SQLAlchemy engine from the credentials provided
    def __init__(self, credentials_dict):
        # DEFINE THE DATABASE CREDENTIALS
        user = credentials_dict['RDS_USER'] #'root'
        password = credentials_dict['RDS_PASSWORD'] #'password'
        host = credentials_dict['RDS_HOST'] #'127.0.0.1'
        port = credentials_dict['RDS_PORT'] #5432
        database = credentials_dict['RDS_DATABASE'] #'postgres'

        #Start engine
        engine = create_engine("postgresql+psycopg2://{0}:{1}@{2}:{3}/{4}".format(user, password, host, port, database))
        
        #Check table names:        
        # inspector = inspect(engine)
        # table_names = inspector.get_table_names()
        # print(table_names)
        
        #Extract data from database:
        with engine.execution_options(isolation_level='AUTOCOMMIT').connect() as conn:
            loan_payments = pd.read_sql_table('loan_payments', engine)
            #loan_payments = pd.read_sql_query('''SELECT * FROM loan_payments LIMIT 20''', engine).set_index('id')
          
        #Save the data to csv:
        save_df_to_csv(loan_payments)

def get_credentials():
    with open('credentials.yaml', mode='r') as file:
        mydict = yaml.safe_load(file)
    return mydict

#Function to save the data as .csv
def save_df_to_csv(df):
    df.to_csv('loan_payments.csv', index=False) # index=False is used to avoid saving the index column.

#Function to open the .csv file obtained from the RDS
def open_csv_as_df():
    df = pd.read_csv('loan_payments.csv')
    return df

if __name__ == '__main__':
    #RDSDatabaseConnector(get_credentials())
    df = open_csv_as_df()
    print(df.head())

# %%

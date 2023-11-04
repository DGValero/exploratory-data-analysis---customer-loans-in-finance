''' 
Script to extract the data from the cloud 
The loan payments data is stored in an AWS RDS database. 
In this script I create the methods which will allow me to extract the data from this database. 
'''
#%%
import yaml
import pandas as pd

class RDSDatabaseConnector:
    #Method that initialises a SQLAlchemy engine from the credentials provided

    #Method that extracts data from database
    #The data is stored in a table called loan_payments. 

    pass

def get_credentials():
    with open('credentials.yaml', mode='r') as file:
        mydict = yaml.safe_load(file)
    return mydict

#Function to save the data as .csv
#sample_df.to_csv('sample.csv', index=False) # index=False is used to avoid saving the index column.

if __name__ == '__main__':
    RDSDatabaseConnector

# %%

'''
Script containing methods to modify a dataframe.
These methods include: formatting data types (to integer, string and datetime),
removing suffixes, mapping from a yaml file and obtaining the skewness of data.
'''
import pandas as pd
import yaml


class DataFrameTransform:
    '''
    Parameters:
    ----------
    df: dataframe
        Dataframe that contains the data to be transformed by the method
    list_columns: list
        List containing the names of the columns where we want to apply the method
    column: string
        Name of the column where we want to apply the method
    suffix: string
        Text (suffix) that we want to remove from the data
    yaml_file: string
        Name of the yaml file containing the dictionary for the mapping we want to apply.
        Filename is to include file extension. Usage example: 'employment_length_dict.yaml'

    Methods:
    -------
    format_to_integer(df, list_columns)
        Method to format the columns in list_columns to 'int64' data type

    format_to_string(df, list_columns)
        Method to format the columns in list_columns to 'string' data type

    format_to_datetime(df, list_columns)
        Method to format the columns in list_columns to 'datetime' data type
        The original data must be stored as 'Month-Year' (eg. Jan-21), or in
        other words stored as '%b-%Y', for the method to work sucessfully
    
    format_remove_suffix(df, column, suffix)
        Method to remove the suffix on rows of a column.
        The output is converted to a numeric data type
    
    format_to_mapping(df, column, yaml_file)
        Method to apply a mapping dictionary to a column. The dictionary must
        be stored as a yaml file in the location indicated in yaml_file.

    get_skewness(df, list_columns)
        Method to identify a list of columns that contain skewed data.
        The method will filter the list list_columns based on a skewness threshold of 5,
        and it will print out the list of columns that are above the skewness threshold.
    '''

    def format_to_integer(self, df, list_columns):
        '''
        Method to format the columns in list_columns to 'int64' data type
        '''
        for column in list_columns:
            df[column] = df[column].astype('int64', errors='ignore')

    def format_to_string(self, df, list_columns):
        '''
        Method to format the columns in list_columns to 'string' data type
        '''
        for column in list_columns:
            df[column] = df[column].astype('string')
    
    def format_to_datetime(self, df, list_columns):
        '''
        Method to format the columns in list_columns to 'datetime' data type
        The original data must be stored as 'Month-Year' (eg. Jan-21), or in
        other words stored as '%b-%Y', for the method to work sucessfully
        '''
        for column in list_columns:
            df[column] = pd.to_datetime(df[column], format='%b-%Y')

    def format_remove_suffix(self, df, column, suffix):
        '''
        Method to remove the suffix on rows of a column.
        The output is converted to a numeric data type
        '''
        df[column] = df[column].str.replace(suffix, '', regex=False)
        df[column]  = pd.to_numeric(df[column])

    def format_to_mapping(self, df, column, yaml_file):
        '''
        Method to apply a mapping dictionary stored in a yaml file to a column
        '''
        with open(yaml_file, mode='r') as file:
            mydict = yaml.safe_load(file)
        df[column] = df[column].replace(mydict)

    def get_skewness(self, df, list_columns):
        '''
        Method to identify a list of columns that contain skewed data.
        The method will filter the list list_columns based on a skewness threshold of 5,
        and it will print out the list of columns that are above the skewness threshold.
        '''
        list_skewed_columns = []
        for each_column in list_columns:
            if abs(df[each_column].skew()) > 5: # This is the value I defined as my skew threshold 
                print(f"Population {each_column} is skewed with a skew value of {df[each_column].skew()}")
                list_skewed_columns.append(each_column)
        print(list_skewed_columns)
        return list_skewed_columns

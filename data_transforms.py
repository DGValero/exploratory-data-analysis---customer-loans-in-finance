'''
Script containing methods to transform data.
These methods include: removing columns, imputing values, transform skewed 
columns to a normal distribution and removing outliers.
'''
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import yeojohnson


class DataTransform:
    '''
    Parameters:
    ----------
    df: dataframe
        Dataframe that contains the data to be transformed by the method
    list_columns: list
        List containing the names of the columns where we want to apply the method
    n: integer
        Number of rows
    column_name: string
        Name of the column where we want to apply the method

    Methods:
    -------
    remove_columns(df, list_columns)
        Removes the columns especified in a list (list_columns) from the dataframe df.

    impute_median_values(df, list_columns)
        Fills empty rows of data with the column median for each column especified in the list: list_columns.

    log_transform(df, list_columns)
        Applies the natural logarithm for each column especified in the list: list_columns.
        This helps analysing data that follows an exponenctial distribution as the log transformation
        may convert the data distribution so it is closer to a normal distribution. And therefore,
        some statistical techniques can be used.

    yeojohnson_transform(df, list_columns)
        Applies the yeojohnson function to the data with the aim of transforming 
        the data to a normal distribution.

    remove_n_smallest(df, list_columns, n)
        Removes the n rows with the lowest values for each column especified in the list: list_columns.
        This method was designed to remove outliers from the dataset.
    '''

    def remove_columns(self, df, list_columns):
        '''
        Removes the columns specified in list_columns
        '''
        df = df.drop(columns=list_columns)
        return df
    
    def impute_median_values(self, df, list_columns):
        '''
        Method to impute missing values using the median value for each column.
        '''
        for each_column in list_columns:
            df[each_column] = df[each_column].fillna(df[each_column].median())

    def log_transform(self, df, list_columns):
        '''
        Applies the natural log to the data with the aim of transforming the data to a normal distribution.
        '''
        for each_column in list_columns:
            df[each_column] = df[each_column].map(lambda i: np.log(i) if i > 0 else 0)
        return df

    def yeojohnson_transform(self, df, list_columns):
        '''
        Applies the yeojohnson function to the data with the aim of transforming the data to a normal distribution.
        '''
        for each_column in list_columns:
            yeojohnson_population = stats.yeojohnson(df[each_column])
            yeojohnson_population= pd.Series(yeojohnson_population[0])
            df[each_column] = yeojohnson_population
        return df
    
    def remove_n_smallest(self, df, column_name, n):
        '''
        Removes the n rows with the lowest values for each column with the aim to remove lower end outliers.
        '''
        sorted_df = df.sort_values(by=column_name)
        rows_to_remove = sorted_df.head(n).index

        updated_df = df.drop(rows_to_remove)

        return updated_df
    
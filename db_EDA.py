#%%
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import numpy as np
import plotly.express as px

pd.set_option('display.max_columns', None)

#CONVERT COLUMNS TO THE CORRECT FORMAT:
class DataTransform:
    def __init__(self, df):
        df.funded_amount = df.funded_amount.astype('int64', errors='ignore')
        df.funded_amount_inv = df.funded_amount_inv.astype('int64', errors='ignore')
        df.grade = df.grade.astype('string')
        df.sub_grade = df.sub_grade.astype('string')
        df.issue_date = pd.to_datetime(df.issue_date, format='%b-%Y')
        df.earliest_credit_line = pd.to_datetime(df.earliest_credit_line, format='%b-%Y')
        df.last_payment_date = pd.to_datetime(df.last_payment_date, format='%b-%Y')
        df.next_payment_date = pd.to_datetime(df.next_payment_date, format='%b-%Y')
        df.last_credit_pull_date = pd.to_datetime(df.last_credit_pull_date, format='%b-%Y')
        df.application_type = df.application_type.astype('string')
        #creating a mapping dictionary for employment_length
        # 0 will mean: <1 year
        # 10 will mean: +10 years
        employment_length_mapping = {
            '< 1 year': 0,
            '1 year' : 1,
            '2 years' : 2,
            '3 years' : 3,
            '4 years' : 4,
            '5 years' : 5,
            '6 years' : 6,
            '7 years' : 7,
            '8 years' : 8,
            '9 years' : 9,
            '10+ years' : 10
        }

        #Replacing the employment years in the column
        df['employment_length'] = df['employment_length'].replace(employment_length_mapping)
        #print("\nDataframe after cleaning 'employment_length' Column:")
        #print(df['employment_length'])

        #Replacing the suffix "months" in the term column
        df['term'] = df['term'].str.replace(' months', '', regex=False)
        df.term = pd.to_numeric(df.term)

        #Replacing 'payment_plan' column with booleans:
        df['payment_plan'] = df['payment_plan'].replace({r'n': False, r'y': True})


#This class contains methods that generate useful information about the DataFrame
class DataFrameInfo:
    def __init__(self):
        pass

    #Describe all columns in the DataFrame to check their data types:
    def column_data_types(self, df):
        print(df.dtypes)

    #Extract statistical values: median, standard deviation and mean from the columns and the DataFrame
    def statistical_values(self, df):
        print(df.describe())
    
    #Print out the shape of the DataFrame:
    def dataframe_shape(self, df):
        print(df.shape)

    #Generate a count/percentage count of NULL values in each column:
    def count_nulls(self, df):     
        print(df.isnull().sum()/len(df)*100) #or print(df.isnull().mean()* 100)

    def get_skewness(self, list_columns):
        list_skewed_columns = []
        for each_column in list_columns:
            if abs(df[each_column].skew()) > 1: #This is the value I defined as my skew threshold 
                    #print(f"Population {each_column} is skewed with a skew value of {df[each_column].skew()}")
                    list_skewed_columns.append(each_column)
        return list_skewed_columns
    
#CLASS TO VISUALISE INSIGHTS FROM THE DATA
class Plotter:
    def __init__(self):
        pass

    #Plot histograms 
    def histogram_plots(self,list_columns):
        for each_column in list_columns:
            if df[each_column].dtype == 'float64' or df[each_column].dtype == 'int64':
                print(each_column)
                #Show histogram:
                df[each_column].hist(bins=50)

    #Quantile-Quantile plots
    def qq_plots(self,list_columns):
        for each_column in list_columns:
            if df[each_column].dtype == 'float64' or df[each_column].dtype == 'int64':
                print(each_column)
                #Show qq plot:
                qq_plot = qqplot(df[each_column] , scale=1 ,line='q')
                pyplot.show()

    def visualise_null_values(self,df):
        msno.matrix(df)

    #Plot histograms with seaborn
    def seaborn_histograms(self,numeric_features):
        sns.set(font_scale=0.7)
        f = pd.melt(df, value_vars=numeric_features)
        g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False)
        g = g.map(sns.histplot, "value", kde=True)

    #Plot box-and-whiskers
    def box_whisker_plots(self, list_columns):
        for each_column in list_columns:
            fig=px.box(df, y=str(each_column), width=600, height=500)
            fig.show()

    #Plot correlation matrix
    def correlation_plot(self, dataframe):
        sns.heatmap(dataframe.corr(), square=True, annot=False, cmap='coolwarm')
        plt.yticks(rotation=0)
        plt.title('Correlation Matrix of all Numerical Variables')
        plt.show()

#CLASS TO PERFORM EXPLORATORY DATA ANALYSIS TRANSFORMATIONS
class DataFrameTransform:
    def __init__(self):
        pass

    
    def remove_columns(self, df, list_columns):
        df = df.drop(columns=list_columns)
        return df
    
    #Method to impute missing values:
    def impute_values(self, list_columns):
        for each_column in list_columns:
            df[each_column] = df[each_column].fillna(df[each_column].median())

    #Method to transform skewed data to normal distribution:
    def transform_skewed(self, list_columns):
        for each_column in list_columns:
            df[each_column] = df[each_column].map(lambda i: np.log(i) if i > 0 else 0)

    #Method to remove oulier rows from the dataset:
    def remove_n_smallest(self, df, list_columns):
        for each_column in list_columns:
            n_smallest = df.nsmallest(74, each_column)
            df=df[~df.isin(n_smallest)].dropna(how='all')
        return df
    
#Function to obtain list of columns from the df with numeric features
def get_list_with_numeric_features():
    numeric_features = []
    for column in df:
        if df[column].dtype == 'float64' or df[column].dtype == 'int64':
            numeric_features.append(column)

    categorical_features = [col for col in df.columns if col not in numeric_features]
    numeric_features.remove("id")
    numeric_features.remove("member_id") 
    numeric_features.remove("collections_12_mths_ex_med")
    numeric_features.remove("policy_code")

    return numeric_features, categorical_features

if __name__ == '__main__':
    df = pd.read_csv('loan_payments.csv')
    DataTransform(df) #transform to correct data types

    #Drop the columns where percentage of missing data is too high to impute values:
    list_columns_missing_data =['mths_since_last_delinq','mths_since_last_record','next_payment_date','mths_since_last_major_derog']
    df = DataFrameTransform().remove_columns(df, list_columns_missing_data)
    #DataFrameInfo().count_nulls(df)

    list_columns_with_nulls = ['funded_amount','term','int_rate','employment_length','last_payment_date','last_credit_pull_date','collections_12_mths_ex_med']
    #Plotter().histogram_plots(list_columns_with_nulls)
    #Plotter().visualise_null_values(df)
       
    DataFrameTransform().impute_values(list_columns_with_nulls)

    #Plotter().visualise_null_values(df)
    #DataFrameInfo().count_nulls(df)
    #print(df.dtypes)

    #IDENTIFYING AND CORRECTING SKEWNESS====================================
    numeric_features, categorical_features = get_list_with_numeric_features()
    #print(numeric_features)
    #print('\n')
    #print(categorical_features)
    #Plotter().seaborn_histograms(numeric_features)

    list_skewed_columns = DataFrameInfo().get_skewness(numeric_features)
    #Visualise the data to analyse the skew:
    #Plotter().qq_plots(list_skewed_columns)

    DataFrameTransform().transform_skewed(list_skewed_columns)

    #Visualise the data to check the results of the transformation have improved the skewness of the data:
    print('========================================================')
    #print('AFTER CORRECTING SKEWNESS:')
    #Plotter().qq_plots(list_skewed_columns)
    #list_skewed_columns = DataFrameInfo().get_skewness(numeric_features)
    #Plotter().seaborn_histograms(numeric_features)
    
    #Save a separate copy of your DataFrame to compare results later on:
    df_copy = df

    #REMOVE OUTLIERS FROM THE DATA=========================================
    #Visualise box-and-whisker plots to identify outliers
    #Plotter().box_whisker_plots(numeric_features)
    #From the box whisker plots we can see that total_payment, total_payment_inv, total_rec_prncp and total_rec_int
    #have points near 0 that appear to be outliers. To remove these:
    list_outliers_columns = ['total_payment', 'total_payment_inv', 'total_rec_prncp', 'total_rec_int']
    df = DataFrameTransform().remove_n_smallest(df, list_outliers_columns)
    #Check that outliers have been removed:
    #Plotter().box_whisker_plots( list_outliers_columns)

    #DROPPING OVERLY CORRELATED COLUMNS====================================
    # create correlation matrix for the data:
    #First select the numeric columns of the dataframe: df[numeric_features]
    Plotter().correlation_plot(df[numeric_features])
    #As we want to predict loan_amount and funded_amount, funded_amount_inv and instalment
    #are all strongly collinear, we will drop those columns:
    list_collinear_columns = ['funded_amount', 'funded_amount_inv', 'instalment']
    df = DataFrameTransform().remove_columns(df, list_collinear_columns)
  


# %%

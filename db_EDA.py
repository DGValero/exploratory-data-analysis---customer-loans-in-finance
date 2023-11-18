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

        #Creating mapping dictionary for grade:
        # grade_mapping = {
        #     'A' : '1',
        #     'B' : '2',
        #     'C' : '3',
        #     'D' : '4',
        #     'E' : '5',
        #     'F' : '6',
        #     'G' : '7'
        # }
        # df['grade'] = df['grade'].replace(grade_mapping)
        # df.grade = pd.to_numeric(df.grade)

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
            if abs(df[each_column].skew()) > 5: #This is the value I defined as my skew threshold 
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
            n_smallest = df.nsmallest(10, each_column)
            df=df[~df.isin(n_smallest)].dropna(how='all')
        return df
    
#Function to obtain list of columns from the df with numeric features
def get_list_with_numeric_features(df):
    numeric_features = []
    for column in df:
        if df[column].dtype == 'float64' or df[column].dtype == 'int64' or df[column].dtype == 'Int64':
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
    numeric_features, categorical_features = get_list_with_numeric_features(df)
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
    #df_copy = df

    #REMOVE OUTLIERS FROM THE DATA=========================================
    #Visualise box-and-whisker plots to identify outliers
    #Plotter().box_whisker_plots(numeric_features)
    #From the box whisker plots we can see that total_payment, total_payment_inv, total_rec_prncp and total_rec_int
    #have points near 0 that appear to be outliers. To remove these:
    list_outliers_columns = ['total_payment', 'total_payment_inv', 'total_rec_prncp', 'total_rec_int']
    #df = DataFrameTransform().remove_n_smallest(df, list_outliers_columns)
    #Check that outliers have been removed:
    #Plotter().box_whisker_plots( list_outliers_columns)

    #DROPPING OVERLY CORRELATED COLUMNS====================================
    # create correlation matrix for the data:
    #First select the numeric columns of the dataframe: df[numeric_features]
    ##Plotter().correlation_plot(df[numeric_features])
    #As we want to predict loan_amount and funded_amount, funded_amount_inv and instalment
    #are all strongly collinear, we will drop those columns:
    #list_collinear_columns = ['funded_amount', 'funded_amount_inv', 'instalment']
    list_collinear_columns = ['total_payment_inv','total_rec_prncp']
    df = DataFrameTransform().remove_columns(df, list_collinear_columns)

    #CURRENT STATE OF THE LOANS============================================
    #Percentage of the loans that are recovered against the investor funding
    #This means number of loans that have £0 remaining outstanding:
    percentage_loans_recoverd_inv = len(df[df['out_prncp_inv']==0]) / len(df) *100
    print(f'The percentage of loans recovered is: {round(percentage_loans_recoverd_inv,2)} %')

    #Percentage of the total would be recovered up to 6 months' in the future
    #If we substract 6months of payments to the outstanding amount, then any value below 0 would mean the loan has been paid
    #Visualise the data.
    df['out_prncp_in_six_months'] = df['out_prncp'] - df['instalment']*6
    percentage_loans_recoverd_in_six_m = len(df[df['out_prncp_in_six_months']<=0]) / len(df) *100
    print(f'Estimated percentage of loans that will be recovered in 6 monhts: {round(percentage_loans_recoverd_in_six_m,2)} %')

    plt.bar(['Nowadays','in 6 months'],[percentage_loans_recoverd_inv,percentage_loans_recoverd_in_six_m])
    plt.ylabel("Percentage loans recovered (%)")
    plt.ylim(50, 100)
    plt.show()

    #CALCULATING LOSS=====================================================
    #The company wants to check what percentage of loans have been a loss to the company. 
    # Loans marked as Charged Off in the loan_status column represent a loss to the company.
    # Calculate the percentage of charged off loans historically and amount that was paid 
    # towards these loans before being charged off.
    percentage_loans_charged_off = len(df[df['loan_status']=='Charged Off']) / len(df) *100
    print(f'The percentage of loans that have been a loss is: {round(percentage_loans_charged_off,2)} %')

    mask = df['loan_status']=='Charged Off'
    total_paymnt_before_charged_off = df['total_payment'][mask].sum()
    print(f'Total paid towards loans before being charged off: £' + '{:10,.2f}'.format(total_paymnt_before_charged_off))

    #CALCULATING THE PROJECTED LOSS========================================
    #Based on the interest rate of the loan and the loans term, you can calculate how much 
    #revenue the loan would have generated for the company.
    df['theoretical_revenue'] = df['instalment']*df['term'] - df['total_payment']
    theoretical_revenue = df['theoretical_revenue'][mask].sum()

    mask = df['loan_status']!='Charged Off' #filter rows that are not the charged off ones to calculate actual revenue
    percentage_revenue_lost = theoretical_revenue / df['theoretical_revenue'][mask].sum() *100

    print(f'Revenue that the charged off loans would have generated for the company: £' + '{:10,.2f}'.format(theoretical_revenue))
    print(f'Estimated percentage of revenue lost: {round(percentage_revenue_lost,2)} %')

    #POSSIBLE LOSS=========================================================
    #There are customers who are currently behind with their loan payments this subset of customers represent a risk to company revenue. 
    #Percentage of users in this bracket that currently represent as a percentage of all loans:
    percentage_loans_late = (len(df[df['loan_status']=='Late (31-120 days)']) + len(df[df['loan_status']=='Late (16-30 days)'])) / len(df) *100
    print(f'The percentage of loans at risk of loss is: {round(percentage_loans_late,2)} %')

    #Projected loss of these loans if the customer were to finish the loans term:
    theoretical_loss = df['theoretical_revenue'][df['loan_status']=='Late (31-120 days)'].sum() + df['theoretical_revenue'][df['loan_status']=='Late (16-30 days)'].sum()
    print(f'Projected loss if the late payment customers were to finish the loans term: £' + '{:10,.2f}'.format(theoretical_loss))

    #If customers converted to Charged Off, the percentage of total revenue that these customers and the 
    #customers who have already defaulted on their loan is:
    percentage_potential_loss = theoretical_loss / df['theoretical_revenue'][mask].sum() *100
    print(f'Estimated percentage of revenue lost if late customers converted to Charged Off: {round(percentage_revenue_lost + percentage_potential_loss,2)} %')
    print('===============================================================')

    #INDICATORS OF LOSS====================================================
    #In this section I will be analysing the data to visualise the possible indicators that a 
    #customer will not be able to pay the loan. 
    #To help identify which columns will be of interest, I will create a subset of users who 
    #have already stopped paying and customers who are currently behind on payments and plot
    #another correlation matrix to work out which columns are of interest. 
    #(df['loan_status']=='Charged Off') | 
    df_loss = df.loc[(df['loan_status']=='Late (16-30 days)') | (df['loan_status']=='Late (31-120 days)')]
    #print(df_loss.head(15))

    numeric_features_loss, categorical_features = get_list_with_numeric_features(df_loss)
    Plotter().correlation_plot(df_loss[numeric_features_loss])
    # look our for correlation with delinq_2yr as this indicates late payments

    sns.countplot(df_loss['purpose'])
    plt.title('Loan purpose for late payment customers (count)')
    plt.show()
    # From this graph we can see a clear indication that the loan purpose 'debt_consolidation' and 'credit_card' 
    # can lead to late payments 
    sns.countplot(df_loss['grade'])
    plt.title('Grade for late payment customers (count)')
    plt.show()
    #We can conclude that customers with grades A and G may indicate good customers and that the loan will be paid on time

    df_loss = df.loc[df['loan_status']=='Charged Off']
    numeric_features_loss, categorical_features = get_list_with_numeric_features(df_loss)
    Plotter().correlation_plot(df_loss[numeric_features_loss])
    # look our for correlation with delinq_2yr as this indicates late payments
    sns.countplot(df_loss['purpose'])
    plt.title('Loan purpose for Charged Off customers (count)')
    plt.show()

    #From this graph we can see a clear indication that the loan purpose 'debt_consolidation' 
    #can lead to late payments 

    sns.countplot(df_loss['grade'])
    plt.title('Grade for Charged Off customers (count)')
    plt.show()

    #We can conclude that customers with grades A may indicate good customers and that the loan will be paid on time

    #print(df_loss.describe())

    #The stats also show that annual_inc income has a very low standard deviation
    #This may indicate that people with a similar annual income to 55.7K may be prone to become Charged Off
    mean_annual_income_loss = np.exp(df_loss['annual_inc'].mean())
    print(f'Average income for Charged Off clients = {mean_annual_income_loss}')

    #In summary customers with late loan payments that took the loan with the purpose of debt consolidation 
    #are likely to become Charged Off customers (debt consolidation as an indicator).
# %%

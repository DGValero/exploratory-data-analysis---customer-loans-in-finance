'''
Script containing plotting methods to visualise insights from data.
'''
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import plotly.express as px


class Plotter:
    '''
    Parameters:
    ----------
    df: dataframe
        Dataframe that contains the data to be transformed by the method
    column: string
        Name of the column to apply the method
    list_columns: list
        List containing the names of the columns where we want to apply the method
    numeric_columns: list
        List containing the names of columns with numeric data types
    title: string
        Title of the graph

    Methods:
    -------
    histogram_plots(df, list_columns)
        Plots a histogram for each column in list_columns

    qq_plot_single(df, column)
        Plots a qq graph for the data column specified
    
    qq_plots(df, list_columns)
        Plots multiple qq graphs, one for each column in list_columns

    visualise_null_values(df)
        Plots a graph to visualise null values in a dataframe, df.
    
    seaborn_histograms(df, numeric_columns)
        Plots a histogram for each column in numeric_columns using the seaborn library.
        Plots are grouped in a single visualisation.
    
    box_whisker_single(df, column)
        Plots a box-whisker graph for the data column specified
    
    box_whisker_plots(df, list_columns)
        Plots multiple box-whisker graphs, one for each column in list_columns
    
    correlation_plot(df)
        Plots a graph to visualise correlation vlues for a dataframe, df.

    count_plot(df,, title)
        Count plot for categorical variables
    '''

    def histogram_plots(self, df, list_columns):
        '''
        Plot histograms 
        '''
        for each_column in list_columns:
            if df[each_column].dtype == 'float64' or df[each_column].dtype == 'int64':
                print(each_column)
                #Show histogram:
                df[each_column].hist(bins=50)

    def qq_plot_single(self, df, column):
        '''
        Quantile-Quantile plots
        '''
        if df[column].dtype == 'float64' or df[column].dtype == 'int64':
            print(f"{column} :")
            #Show qq plot:
            qq_plot = qqplot(df[column] , scale=1 ,line='q')
            plt.show()

    def qq_plots(self, df, list_columns):
        '''
        Quantile-Quantile plots
        '''
        #sub-select the columns specified:
        df = df[list_columns]

        # Set the number of rows and columns in the grid
        num_rows = 2
        num_cols = len(df.columns) // num_rows + (len(df.columns) % num_rows > 0)

        # Create a grid of subplots
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 5))

        # Flatten the axes
        axes = axes.flatten()

        # Plot quantile-quantile plots for each column
        for i, column in enumerate(df.columns):
            sm.qqplot(df[column], line='s', ax=axes[i])
            axes[i].set_title(f'Q-Q Plot - {column}')

        # Remove empty subplots
        for j in range(len(df.columns), len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

    def visualise_null_values(self, df):
        '''
        Visualisation showing null values in a dataframe
        '''
        msno.matrix(df)

    def seaborn_histograms(self, df, numeric_columns):
        '''
        Plot histograms with seaborn
        '''
        sns.set(font_scale=0.7) 
        data_frame = pd.melt(df, value_vars=numeric_columns)
        facet_grid = sns.FacetGrid(data_frame, col="variable",  col_wrap=3, sharex=False, sharey=False)
        facet_grid = facet_grid.map(sns.histplot, "value", kde=True)

    def box_whisker_single(self, df, column):
        '''
        Plot a single box-and-whiskers
        '''
        fig=px.box(df, y=str(column), width=600, height=400)
        fig.show()

    def box_whisker_plots(self, df, list_columns):
        '''
        Plot multiple box-and-whiskers graphs
        '''
        #sub-select the columns specified:
        df = df[list_columns]
        
        # Set the number of rows and columns in the grid
        num_rows = 5
        num_cols = len(df.columns) // num_rows + (len(df.columns) % num_rows > 0)

        # Create a grid of subplots
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10))

        # Flatten the axes
        axes = axes.flatten()

        # Plot box-and-whisker plots for each column
        for i, column in enumerate(df.columns):
            sns.boxplot(x=df[column], ax=axes[i])
            #axes[i].set_title(f'Boxplot - {column}')

        # Remove empty subplots
        for j in range(len(df.columns), len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

    def correlation_plot(self, df):
        '''
        Plot correlation matrix
        '''
        sns.heatmap(df.corr(), square=True, annot=False, cmap='coolwarm')
        plt.yticks(rotation=0)
        plt.title('Correlation Matrix of all Numerical Variables')
        plt.show()

    def count_plot(self, df, title):
        '''
        Count plot for categorical variables
        '''
        sns.countplot(df)
        plt.title(title)
        plt.show()

    def discrete_probability_distribution(self, df, column):
        # Calculate value counts and convert to probabilities
        probs = df[column].value_counts(normalize=True)

        # Create bar plot
        dpd=sns.barplot(y=probs.values, x=probs.index)

        plt.xlabel('Values')
        plt.ylabel('Probability')
        plt.title('Discrete Probability Distribution')
        plt.show()

    # ADDITIONAL PLOTTING METHODS (NOT USED IN THE ANALYSIS)===================================
    # =========================================================================================
    def box_whisker_plots_Matplotlib(self, df, list_columns):
        '''
        Plot box-and-whiskers
        '''
        #sub-select the columns specified:
        df = df[list_columns]

        # Plotting box-and-whisker plots for each column
        fig, ax = plt.subplots()
        df.boxplot(ax=ax)

        # Customizing the plot (optional)
        ax.set_title('Box-and-Whisker Plots for Each Column')
        ax.set_ylabel('Values')
        ax.set_xticklabels(df.columns, rotation=45, ha='right')

        # Display the plot
        plt.show()

    def box_whisker_plots_Seaborn(self, df, list_columns):
        '''
        Plot box-and-whiskers
        '''
        #sub-select the columns specified:
        df = df[list_columns]

        # Melt the DataFrame to long format for easy plotting
        df_long = pd.melt(df, var_name='Columns', value_name='Values')

        # Plotting box-and-whisker plots for each column
        sns.set(style="whitegrid")  # Optional: Set a different style
        plt.figure(figsize=(10, 6))  # Optional: Set the figure size

        # Using seaborn's boxplot function
        ax = sns.boxplot(x='Columns', y='Values', data=df_long)

        # Customizing the plot (optional)
        ax.set_title('Box-and-Whisker Plots for Each Column')
        ax.set_ylabel('Values')
        ax.set_xlabel('Columns')

        # Display the plot
        plt.show()
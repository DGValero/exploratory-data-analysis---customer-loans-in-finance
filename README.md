# Exploratory Data Analysis - Customer Loans in Finance
### **Table of contents:**

 - [A description of the project](#item-one)
    - [Aim of the project](#the-aim-of-the-project)
    - [What it does](#what-it-does)

 - [Installation instructions](#item-two)

 - [Usage instructions](#usage-instructions)

 - [File structure of the project](#item-three)

 - [Programming Language & Licensing](#programming-language)

<a id="item-one"></a>

## A description of the project
### The aim of the project
The aim of the project is to simulate a real world Exploratory Data Analysis on finance data, in this case customer loans, where the data needs to be downloaded from a database, it is cleaned, transformed and analysed in Python to be able to find out the state of the loans, find performance indicators and report back to senior leaders.

### What it does
#### Data Extraction
As it would happen in a real case scenario, the data was stored in an AWS database. To analyse the data, first I had to access it. 

To do so, I created a script **db_Extraction.py** to connect to the RDS database and download the data as csv file. To do this I followed a few steps:
1. I saved the connection credentials in a .yaml file, as I do not want my credentials being pushed to GitHub for security reasons. To load the credentials.yaml file I used the python package `PyYAML`.
2. I created a python method **RDSDatabaseConnector** that initialises a `SQLAlchemy` engine using the credetials passed from the yaml file. 
3. The method above extracts data from the RDS database and returns it as a `Pandas` dataframe.
4. Finally, I created a function to save the dataframe to a csv file for improved performance. A copy of the csv file is also saved in the repository as **loan_payments.csv**.

#### Exploratory Data Analysis (EDA)
Next, I created a python notebook to analyse and document the data analysis, in a file named **db_Analysis.ipynb**. This is the main file of the project where the exploratory data analyis is undertaken.

For readability and in order to create a set of methods (tools) that I can reuse for future projects I created 3 modules that are imported into the **db_Analysis.ipynb** python notebook:
- **dataframe.py**: this is a python module I created to perform changes on dataframes. It includes methods to formatting data types,removing suffixes, mapping data and obtaining the skewness of data.
    - **employment_length_dict.yaml**: this is a yaml file I created and it contains a dictionary to map the employment lenght data, so that the data can be converted to a categorical type.
- **data_transforms.py**: this is a python module I created to perform data transformations. It includes methods for removing columns, imputing values, transform skewed data, and removing outliers.
- **plot.py**: this is a python module I created to aid in the creation of graphs during my data analysis. It contains methods to create histograms, q-q charts, count plots, amongts other plotting methods.

I split the analysis undertaken in the python notebook in different sections, this is explained the paragraphs below.

#### 1. Loading and cleaning the data, including formatting data and imputing missing values.

I loaded the data into a dataframe using the `Pandas` method *read_csv()* and familiarised myself with the data using the *head()* and *info()* methods, identifying:

- Columns that could be represented better numerically
- Identifying catergorical columns
- Dates that are in the wrong format
- Null values

For the columns that needed to be coverted the correct format, I created a  *DataFrameTransform()* class within the **dataframe.py** module to handle these conversions. 

An important EDA task is to impute or remove missing values from the dataset. Missing values can occur due to a variety of reasons such as data entry errors or incomplete information. I calculated the percentage of missing data in the columns:
- For columns with more than 10% missing data: I decided to drop these columns as they were not relevant to my analysis later on.
- For columns with less than 1%: I imputed the data using statistical concepts and assumptions.

#### 2. Identifying and correcting skewness

Skewed data can lead to biased models and inaccurate results, so it's important to address this issue before proceeding with any analysis. 

First, I identified the skewed columns in the data making use of histograms and the `Pandas` *skew()* method.

Once the skewed columns were identified, I performed different transformations on these columns to determine which transformation results in the biggest reduction in skew. In my case, the statistical `Yeo-Jonhson` transformation produced better results. I implemented these transformation methods in my **data_transforms.py** module.

Finally I visualised the data to check that the results of the transformation have improved the skewness of the data. 

#### 3. Removing outliers from the data

Removing outliers from the dataset will improve the quality and accuracy of the analysis as outliers can distort the analysis results.

To identify outliers I used box-whisker plots and this is a great visual way to represent inter-quartile ranges.

Once identified, I used a method to remove the outliers from the dataset. Again, I built this method in my **data_transforms.py** module.

With the outliers removed, I re-visualised my data with my *Plotter* class to check that the outliers have been correctly removed. 

#### 4. Dropping overly correlated columns

Highly correlated columns in a dataset can lead to multicollinearity issues, which can affect the accuracy and interpretability of models built on the data.

First, I computed the correlation matrix for the dataset and plotted it.

I identified the columns that are highly correlated but I decided to keep all these columns as they could be useful for my analysis later on and the data does not need to be used by machine learning algorithms, only predictive models would be applied.

#### 5. Current state of the loans, where I analyse the data draw conclusions

    WORK IN PROGRESS

<a id="item-two"></a>

## Installation instructions
First initialise git bash. Then you will need to navigate to the folder directory where you would like to save the files. For example in Windows:
> cd C:/Users/user/my_project

To get a copy of this repository, you can do so with the following command:
> git clone https://github.com/DGValero/exploratory-data-analysis---customer-loans-in-finance.git

## Usage instructions
Open the **db_Analysis.ipynb** to explore the data analysis undertaken on the customer loan finance database saved as a csv file: **loan_payments.csv** 

<a id="item-three"></a>

## File structure of the project
├── **db_Analysis.ipynb** - this is the main file of the EDA project. 

│├── **dataframe.py** - module with methods to makes changes to dataframes

││├──**employment_length_dict.yaml** - dictionary to map the employment lenght data

│├── **data_transforms.py** - module with methods to transform data

│├── **plot.py** - module with plotting methods

├── **db_Extraction.py** - script to download the data stored in the AWS RDS.

├── **loan_data_dict.md** - dictionary containing a description of the data 

├── **loan_payments.csv** - local copy of the database 

## Programming language
Python 3.6

## License information
Apache License 2.0

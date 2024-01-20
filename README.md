
# Exploratory Data Analysis - Customer Loans in Finance


## A description of the project
### The aim of the project
The aim of the project is to simulate a real world Exploratory Data Analysis on finance data, in this case customer loans. 

### What it does
As it would happen in a real case, the data was stored in an AWS database. To analyse the data, first I created a script **db_Extraction.py** to connect to the database and I used the credentials stored in a .yaml file. Once the connection was stablished, the script downloads the data and stores it locally as a .csv file for improved performance. A copy of this file is also saved in the repository as **loan_payments.csv**.

Next, I created a python notebook to analyse and document the data analysis, in a file named **db_Analysis.ipynb**. 

For readability and in order to create a set of functions or tools that I can reuse for future projects I created 3 modules that are imported into the python notebook:
- **dataframe.py**
- **data_transforms.py**
- **plot.py**

These modules are explained in more detail in the section 'File Structure' below.

The analysis undertaken in the python notebook is split in different sections:
1. Loading and cleaning the data, including formatting data and imputing missing values.
2. Identifying and correcting skewness
3. Removing outliers from the data
4. Dropping overly correlated columns
5. Current state of the loans, where I analyse the data draw conclusions


## Installation instructions
First initialise git bash. Then you will need to navigate to the folder directory where you would like to save the files. For example in Windows:
> cd C:/Users/user/my_project

To get a copy of this repository, you can do so with the following command:
> git clone https://github.com/DGValero/exploratory-data-analysis---customer-loans-in-finance.git

## Usage instructions
Open the **db_Analysis.ipynb** to explore the data analysis undertaken on the customer loan finance database saved as a csv file: **loan_payments.csv** 

## File structure of the project
**db_Analysis.ipynb** this is the main file of the project. It is a python notebook that contains the data analysis undertaken on a customer loan finance database.

├── **dataframe.py** this is a python module I created to perform changes on dataframes. It includes methods to formatting data types,removing suffixes, mapping data and obtaining the skewness of data.

│├──**employment_length_dict.yaml** this is a yaml file I created and it contains a dictionary to map the employment lenght data, so that the data can be analysed.

├── **data_transforms.py** this is a python module I created to perform data transformations. It includes methods for removing columns, imputing values, transform skewed data, and removing outliers.

├── **plot.py** this is a python module I created to aid in the creation of graphs during my data analysis. It contains methods to create histograms, q-q charts, count plots, amongts other plotting methods.

**db_Extraction.py** this is the script that I used to extract the finance data that was stored in an AWS RDS.

**credentials.yaml** this file is not uploaded to the repository. This file contains credential information to access the RDS and is ignored by Git due to containing sensitive information such as passwords.

**loan_data_dict.md** dictionary containing a description of each data column of the database.

**loan_payments.csv** this file is a local copy of the database to increase performance when undertaking the data analysis. This file was downloaded using the **db_Extraction.py** script.

## Programming language
Python 3.6

## License information
Apache License 2.0

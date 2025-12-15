# import data manipulatation libraries
import numpy as np 
import pandas as pd

# Import data visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Import Data warning libraries
import warnings 
warnings.filterwarnings('ignore')

#Import scikrit learn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler


filepath = 'https://raw.githubusercontent.com/shivamsingh-itds/StudentPerformanceModelAnalysis/refs/heads/main/data/Student_Performance.csv'

# Step 1 : Data Ingestion 
def data_ingestion():
     return pd.read_csv(filepath)

# step 2 : Data Exploration
def data_exploration(df):
    from collections import OrderedDict
    numerical_col = df.select_dtypes(exclude = "object").columns
    categorical_col = df.select_dtypes(include = "object").columns

    # Checking Stats: Numerical Columns
    num_stats = []
    cat_stats = []
    data_info = []

    for i in numerical_col:

        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        IQR = Q3 - Q1
        LWR = Q1 - 1.5*IQR
        UWR = Q3 + 1.5*IQR

        outlier_count = len(df[(df[i] < LWR) | (df[i] > UWR)])
        outlier_percentage = (outlier_count / len(df)) * 100

        numericalstats = OrderedDict({
            "Feature":i,
            "Mean":df[i].mean(),
            "Median":df[i].median(),
            "Minimum":df[i].min(),
            "Maximum":df[i].max(),
            "Q1":Q1,
            "Q3":Q3,
            "IQR":IQR,
            "LWR":LWR,
            "UWR":UWR,
            "Outlier Count":outlier_count,
            "Outlier Percentage":outlier_percentage,
            "Standard Deviation":df[i].std(),
            "Variance":df[i].var(),
            "Skewness":df[i].skew(),
            "Kurtosis":df[i].kurtosis()
            })
        num_stats.append(numericalstats)
        numerical_stats_report = pd.DataFrame(num_stats)

    # Checking for Categorical columns
    for i in categorical_col:
        cat_stats1 = OrderedDict({
            "Feature":i,
            "Unique Values":df[i].nunique(),
            "Value Counts":df[i].value_counts().to_dict(),
            "Mode":df[i].mode()[0]
        })
        cat_stats.append(cat_stats1)
        categorical_stats_report = pd.DataFrame(cat_stats)

    # Checking datasetinformation
    for i in df.columns:
        data_info1 = OrderedDict({
            "Feature":i,
            "Data Type":df[i].dtype,
            "Missing_Values":df[i].isnull().sum(),
            "Unique_Values":df[i].nunique(),
            "Value_Counts":df[i].value_counts().to_dict()
            })
        data_info.append(data_info1)
        data_info_report = pd.DataFrame(data_info)

    return numerical_stats_report,categorical_stats_report,data_info_report


    

    # return num_stats,cat_stats,data_info
    pass

# step 3 : Data preprocessiong
def data_preprocessing(df):
    X = df.drop(columns = 'Performance Index',axis = 1)
    y = df['Performance Index']

  # Split the Dataset into train and test
    X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                  test_size = 0.3,
                                                  random_state = 10)

    for i in df.select_dtypes(include = "object").columns:
        le = LabelEncoder()
        X_train[i] = le.fit_transform(X_train[i])  # Seen Data
        X_test[i] = le.transform(X_test[i])        # Unseen Data

    # Using Normalization Technique
      
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)          # Seen Data
    X_test = sc.transform(X_test)                # Unseen Data
    return X_train,X_test,y_train,y_test


# step 4 : Model Building
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
def data_model(df):
    
    model_comparison = []

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor()
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2score = r2_score(y_test, y_pred)

        model_comparison.append({
            "Model Name": model_name,
            "R2 Score": r2score
        })

    model_comparison = pd.DataFrame(model_comparison)
    return model_comparison


# function calling 
df = data_ingestion()

num_stats, cat_stats , data_info = data_exploration(df)

X_train ,X_test , y_train ,y_test = data_preprocessing(df)

model_comparsion = data_model(df)


# Testing

print(df)
print(num_stats)
print(cat_stats)
print(data_info)
print(X_train)
print(model_comparsion)

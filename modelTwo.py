from analyze import outliers
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from regressors import stats


#---------------
# We are going to improve out model making some changes to ur features, this is called 
# feature engineering, we can transform, combine, or create new features based on the originals. 
# The goal is to have variables that have a better relationship with the target variable.
#
#For our new model we are going to add three new variables: 
# 1.- Age squared to give more weight to larger values. 
# 2.- A boolean variable indicating whether the person is overweight. 
# 3.-a logical operation indicating whether the person is overweight and a smoker. 
# --------------


def run():
    #LOAD DATA 
    df = pd.read_csv('./insurance.csv')

    #ONE HOT
    df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True, dtype='int64')

    #REMOVE OUTLIERS
    df_second = df.copy()
    df_second = df_second[df_second['charges'] < 50000]
    min_bmi, max_bmi = outliers(df_second, 'bmi')
    df = df[(df['bmi'] < max_bmi) & (df['bmi'] > min_bmi)]

    #FEATURE ENGINEERING
    df_second['age2'] = df['age']**2
    df_second['overweight'] = (df['bmi'] > 25).astype(int)
    df_second['overweight*smoker'] = df_second['overweight'] * df_second['smoker_yes'] 
    
    #SPLIT FEATURES
    columns = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes','region_northwest', 
               'region_southeast', 'region_southwest', 'age2','overweight', 'overweight*smoker']
    X = df[columns].values
    Y = df['charges'].values.reshape(-1, 1)
    
    #SPLIT TEST & TRAIN DATA
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.25)
    
    #SCALE DATA
    x_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler.fit(Y)
    
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)
    Y_train = x_scaler.transform(Y_train)
    Y_test = x_scaler.transform(Y_test)
    
    #MODEL
    model = LinearRegression()
    model.fit(X_train, Y_train) #Train second model
    

if __name__ == '__main__':
    run()
    
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

#LOAD DATA 
df = pd.read_csv('./insurance.csv')

#ONE HOT
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True, dtype='int64')

#FEATURE ENGINEERING
df_second = df.copy()
df_second['age2'] = df['age']**2
df_second['overweight'] = (df['bmi'] > 25).astype(int)
df_second['overweight*smoker'] = df_second['overweight'] * df_second['smoker_yes'] 
print(df_second.head())

def run():
    pass

if __name__ == '__main__':
    run()
    
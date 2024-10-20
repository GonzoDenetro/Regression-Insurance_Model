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
    
    #FEATURE ENGINEERING
    df_second['age2'] = df['age']**2
    df_second['overweight'] = (df['bmi'] > 25).astype(int)
    df_second['overweight*smoker'] = df_second['overweight'] * df_second['smoker_yes'] 
    
    
    #SPLIT FEATURES
    columns = ['bmi', 'children', 'age2','overweight', 'overweight*smoker']
    X = df_second[columns].values
    Y = df_second['charges'].values.reshape(-1, 1)
    
        
    #SPLIT TEST & TRAIN DATA
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.25)
    
    #SCALE DATA
    x_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(Y)
    
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)
    Y_train = y_scaler.transform(Y_train)
    Y_test = y_scaler.transform(Y_test)
    
    #Look if we have NaN values
    x_nan = pd.isna(X_train).any()
    print(x_nan)
    
    
    #MODEL
    model = LinearRegression(fit_intercept=False) # We dont add the intercept as parameter
    model.fit(X_train, Y_train) #Train model
    
    #PREDICTION
    y_pred = model.predict(X_test)
    
    #EVALUATE
    r2 = metrics.r2_score(Y_test, y_pred)
    print(r2)


    #MODEL STATS
    Y_test = Y_test.reshape(-1)
    model.coef_ = model.coef_.reshape(-1)

    print("=========Summary========")
    stats.summary(model, X_test, Y_test, columns)

if __name__ == '__main__':
    run()
    
#INSIGHTS
# After looking to the model stats, with the p-value we can see tha,
# the features with p-value greater of 5% can be discard,
# the features with a p-value lower than 5% are more significative

#The features that are gonna be remove are intecept, age and sex
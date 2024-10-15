from analyze import outliers
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def run():
    #LOAD DATA
    df = pd.read_csv('./insurance.csv')
    
    #ONE HOT
    df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True, dtype='int64')
    
    #REMOVE OUTLIERS
    df = df[df['charges'] < 50000]
    min_bmi, max_bmi = outliers(df, 'bmi')
    df = df[(df['bmi'] < max_bmi) & (df['bmi'] > min_bmi)]
    
    #SPLIT FEATURES
    X = df[['age', 'bmi', 'children', 'sex_male', 'smoker_yes']].values.T
    Y = df['charges'].values.T
    
    #SPLIT TRAIN & TEST SET    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=104, test_size=0.25)
    
    #SCALE DATA
    x_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(Y)
    
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)
    Y_train = y_scaler.transform(Y_train)
    Y_test = y_scaler.transform(Y_test)

if __name__ == '__main__':
    run()
    
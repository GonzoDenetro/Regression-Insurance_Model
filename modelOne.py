from analyze import outliers
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from regressors import stats


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
    X_columns = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes']
    X = df[X_columns].values
    Y = df['charges'].values.reshape(-1, 1)
    
    #SPLIT TRAIN & TEST SET    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.25)
    
    #SCALE DATA
    x_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(Y)
    
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)
    Y_train = y_scaler.transform(Y_train)
    Y_test = y_scaler.transform(Y_test)
    
    #MODEL
    model = LinearRegression()
    model.fit(X_train, Y_train) #Train model
    
    coef = model.coef_.T #Model parameters
    intercept = model.intercept_[0] #Bias term
    y_pred = model.predict(X_test)
    
    print(f'Model Parameters: {intercept}, {coef}')
    
    #EVALUATE MODEL
    mse = metrics.mean_squared_error(Y_test, y_pred) #Mean Squared Error
    r2 = metrics.r2_score(Y_test, y_pred) #R^2
    
    print(f'Mean Square Error: {mse.round(4)}')
    print("R^2", r2)
    
    #MODEL STATS
    Y_test = Y_test.reshape(-1)
    model.coef_ = model.coef_.reshape(-1)
    model.intercept_ = model.intercept_[0]
    
    print("=========Summary========")
    stats.summary(model, X_test, Y_test, X_columns)   

if __name__ == '__main__':
    run()
    
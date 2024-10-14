import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np 

#----------
# Before start working with the model, first we need to analyze our data,
# look the relations between features, their distribution. So we can get the most
# relevant features
# ----------

#LOAD DATA 
df = pd.read_csv('./insurance.csv')

print(df.head())
print("--"*50)
print(df.describe())

#ONE HOT
df_oneht = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True, dtype='int64')
sns.scatterplot(x=df_oneht['age'], y=df_oneht['charges'], hue=df_oneht['smoker_yes'])

#VISUALUZE FRECUENCY 
fig, axs = plt.subplots(2, 2, figsize=(12, 10)) #Create a a canvas with 2 rows and 2 columns
axs = axs.flatten() #We make axs of 1D
features = ['charges', 'age', 'bmi']

#Frequency of data
for i, feature in enumerate(features):
    axs[i].hist(df_oneht[feature], bins=40,  color='skyblue', edgecolor='black')
    axs[i].set_title(feature)

# VISUALIZE RELATIONS
sns.pairplot(df, height=2.5)
plt.tight_layout()
plt.show()

#BMI OUTLIERS
Q1_bmi = df_oneht['bmi'].quantile(0.25)
Q2_bmi = df_oneht['bmi'].median()
Q3_bmi = df_oneht['bmi'].quantile(0.75)
iqr = Q3_bmi - Q1_bmi

min_lim = Q1_bmi -(1.5*iqr)
max_lim = Q3_bmi + (1.5*iqr)
print("--"*50)
print(f'Rango para detección de outliers: {min_lim}, {max_lim}')
print("--"*50)

bmi_df = df_oneht[df_oneht['bmi'] < max_lim]

#CHARGES OUTLIERS
charges_df = df_oneht[df_oneht['charges'] < 50000]

#CORRELATION
corr = df_oneht[['age', 'bmi', 'charges']].corr()
print(corr)
print("--"*50)
corr_matrix = np.corrcoef(df_oneht[['age', 'bmi', 'charges']].values.T)
print(corr_matrix)


#INSIGHTS:
# the data of BMI feature has a normal distribution
# the data of charges has a asimetric distribution to the right
# We can a positive relation between charges & age 


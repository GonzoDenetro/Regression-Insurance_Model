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

#VISUALUZE FRECUENCY 
fig, axs = plt.subplots(2, 2, figsize=(12, 10)) #Create a a canvas with 2 rows and 2 columns
axs = axs.flatten() #We make axs of 1D
features = ['charges', 'age', 'bmi']

#Frequency of data
for i, feature in enumerate(features):
    axs[i].hist(df[feature], bins=40,  color='skyblue', edgecolor='black')
    axs[i].set_title(feature)

# VISUALIZE RELATIONS
sns.pairplot(df, height=2.5)
plt.tight_layout()
plt.show()



#INSIGHTS:
# the data of BMI feature has a normal distribution
# the data of charges has a asimetric distribution to the right
# We can a positive relation between charges & age 


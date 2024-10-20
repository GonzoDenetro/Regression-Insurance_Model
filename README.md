# Regression-Insurance_Model

This repository contains a linear regression model for predicting insurance premiums based on a dataset that includes information on age, body mass index (BMI), number of children, smoking status, and more.

The dataset used to train the model has the following structure and columns:

age: Age of the insured.
sex: Sex of the insured (female or male).
bmi: Body mass index of the insured.
children: Number of dependent children.
smoker: Indicates whether the insured is a smoker (yes or no).
region: Region of residence of the insured.
charges: Insurance charges (target variable).

### Data Example

| age |   sex   |   bmi   | children | smoker |   region   |   charges   |
|-----|---------|---------|----------|--------|------------|-------------|
| 19  | female  |  27.9   |    0     |  yes   | southwest  |  16884.924  |
| 18  | male    |  33.77  |    1     |  no    | southeast  |  1725.5523  |
| 28  | male    |  33     |    3     |  no    | southeast  |  4449.462   |
| 33  | male    |  22.705 |    0     |  no    | northwest  |  21984.4706 |
| 32  | male    |  28.88  |    0     |  no    | northwest  |  3866.8552  |
| 31  | female  |  25.74  |    0     |  no    | southeast  |  3756.6216  |
| 46  | female  |  33.44  |    1     |  no    | southeast  |  8240.5896  |

## Requirements

The libraries used for this project are located in the `requirements.txt` file. To install them, run the following command:

```bash
pip install -r requirements.txt

### Data Analysis (`analyze.py`)

The `analyze.py` file is responsible for analyzing data from the `insurance.csv` dataset to identify relationships between variables and better understand their distribution. Below are the main functions and findings of this script:

#### Key Features
- **Outlier Detection:** An `outliers()` function is implemented that uses the interquartile range (IQR) to identify outliers in the analyzed features. This allows filtering the data to obtain a cleaner and more representative set.

#### Findings
1. **BMI Distribution:**
- The data for the BMI (Body Mass Index) feature has a normal distribution. This indicates that most values ​​cluster around the mean, which is typical for physical measurements such as BMI.

2. **Distribution of Charges:**
- The data on charges show a skewed distribution to the right. This means that there is a higher concentration of low values ​​and some very high values, suggesting that some individuals have significantly higher expenses than average.

3. **Relationship between Age and Charges:**
- A positive relationship was identified between charges and age. This indicates that, in general, as individuals' age increases, their health insurance expenses also tend to increase.

#### Visualizations
- Scatter plots and histograms are used to visualize the frequency and relationships between key characteristics (age, BMI, and charges).
- The correlation matrix allows us to observe how these variables relate to each other.

These initial analyses are essential to select the most relevant characteristics before building the regression model.

### Linear Regression Model (`modelOne.py`)

The `model.py` file implements a linear regression model to predict health insurance charges from various individual characteristics. The main steps and results obtained are detailed below.

#### Process

1. **Data Loading:**
- The `insurance.csv` dataset is loaded.

2. **One-Hot Encoding:**
- One-hot encoding is applied to categorical variables (`sex`, `smoker`, `region`) to convert them to numeric variables.

3. **Outlier Removal:**
- Outliers in the `charges` column are removed, setting a threshold of 50,000. Outliers in BMI are also filtered out using the `outliers()` function defined in `analyze.py`.

4. **Data Splitting:**
- The features (`X`) and the target (`Y`) are split. The selected features are: `age`, `bmi`, `children`, `sex_male` and `smoker_yes`.
- The data is split into training and test sets (75% for training and 25% for testing) using `train_test_split`.

5. **Data Scaling:**
- The features and target are normalized using `StandardScaler` to improve the accuracy of the model.

6. **Model Training:**
- A linear regression model is trained on the training data.

7. **Model Evaluation:**
- The mean square error (MSE) and coefficient of determination (R²) are calculated to evaluate the performance of the model on the test data. The model achieved an R² of approximately **72%**, indicating that 72% of the variation in charges can be explained by the selected features.

8. **Model Results:**
- The model parameters are printed, including the bias term and coefficients for each feature.

#### Key Results
- **R²:** 0.72
- **Mean Square Error (MSE):** The specific value is printed, indicating the model's accuracy in predictions.

This first model provides a solid foundation for future improvements, such as including interactions between variables or testing more complex models.

### Final Model (`modelTwo.py`)

The `final_model.py` file implements an improved linear regression model using a feature engineering process. This technique seeks to transform, combine, or create new variables from the original ones to optimize the relationship with the target variable, in this case, health insurance charges.

#### Process

1. **Data Loading:**
- The `insurance.csv` dataset is loaded.

2. **One-Hot Encoding:**
- One-hot encoding is applied to categorical variables (`sex`, `smoker`, `region`).

3. **Outlier Removal:**
- Outliers in the `charges` column are filtered out, setting a threshold of 50,000.

4. **Feature Engineering:**
- New features are created to improve the relationship with the jobs:
- **Age squared (`age2`):** Age is squared to give greater weight to higher values.
- **Overweight (`overweight`):** A boolean variable is generated that indicates whether the person has a BMI greater than 25.
- **Interaction between Overweight and Smoking (`overweight*smoker`):** A variable is created that indicates whether a person is both overweight and a smoker.

5. **Data Splitting:**
- Features (`X`) and target (`Y`) are separated using the new variables.
- Data is split into training and test sets (75% for training and 25% for testing).

6. **Data Scaling:**
- Features and target are normalized using `StandardScaler`.

7. **Model Training:**
- A linear regression model is trained without including the bias term.

8. **Model Evaluation:**
- The coefficient of determination (R²) is calculated to evaluate the performance of the model on the test data.

9. **Model Results:**
- It is printed that the R² of the model has increased to 78%, indicating an improvement in the predictive ability compared to the previous model, which had an R² of 72%.
- In addition, information about the significance of the features is printed.

#### Key Results
- It is observed that some features have a p-value greater than 5%, indicating that they are not significant for the model. It is recommended to discard the following variables:
- Intercept
- Age
- Sex

This feature engineering approach has allowed to improve the performance of the model, providing more accurate predictions on health insurance charges.
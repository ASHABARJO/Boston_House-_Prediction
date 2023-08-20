# Features
# The dataset contains 506 rows and 13 columns.

# The dataset has no null or missing values.

# Many machine learning papers use this dataset to address regression problems.

# The dataset contains the following 13 characteristics:
# Characteristic	Description
# CRIM	This is the average per person crime rate by town.
# ZN	This is the extent of private land zoned for lots over 25,000 square feet.
# INDUS	This is the extent of non-retail business sections of land per town.
# CHAS	It is considered to be 1 if tract bounds river, otherwise it’s always 0.
# NOX	This refers to the Nitric Oxide concentration.
# RM	This is the average number of rooms per residence.
# AGE	This is the extent of proprietor-involved units worked before 1940.
# DIS	This is the weighted distance to five Boston business focuses.
# RAD	This is the index of access to radial highways.
# TAX	This is the property tax rate (full-value) per $10,000.
# PRATIO	This tells the student-teacher ratio.
# B	B is calculated by 1000(Bk-0.63)^2. Bk denotes the proportion of black people by town.
# LSTAT	This tells us the percent lower status of the population.



# Step 1: Set Up the Environment
!pip install scikit-learn

# Step 2: Load and Explore the Dataset
from sklearn.datasets import load_boston
import pandas as pd
# Load the dataset
boston=load_boston()
data=pd.DataFrame(boston.data,columns=boston.feature_names)
data['target']=boston.target
# Display the first few row of the dataset
print(data.head())

Step 3: Split Data into Training and Testing Sets

from sklearn.model_selection import train_test_split
# Split the data into feature (X) and target (Y)
X=data.drop('target',axis=1)
y=data['target']
# split the data into training and testing sets 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

Step 4: Train a Regression Model

from sklearn.linear_model import LinearRegression
# Create a linear regression model
model=LinearRegression()
# Train the model on the training data
model.fit(X_train,y_train)

Step 5: Evaluate the Model

from sklearn.metrics import mean_squared_error,r2_score
# Make prediction on the test data
y_pred=model.predict(X_test)

# Calculate the mean squared error and R-squared score
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print("Mean Squared error ", mse)
print("R-squared score",r2)


Step 6: Visualize the Results (Optional)

import matplotlib.pyplot as plt


# Scatter plot with predicted prices on both axes
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Line')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.legend()
plt.show()







# In this plot, the blue dots represent the actual scatter plot of predicted prices on both axes. The red dashed line represents the ideal line where the predicted prices are perfectly aligned with the actual prices.


# it would indicate that the predicted values are very close to the actual values. This alignment along a diagonal line indicates a strong correlation between the predicted and actual values, suggesting that your regression model is performing well and making accurate predictions.

# In other words, if you see a linear relationship in the scatter plot, it means that the model is doing a good job of capturing the underlying patterns in the data. This is the ideal scenario for a regression model, as it means the model's predictions are close to the true values.


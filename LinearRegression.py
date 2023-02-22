import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# Here reading .csv to dataframes
houseData = pd.read_csv("data_set/house_prices.csv")
size = houseData['sqft_living']
price = houseData['price']

# As in pandas .csv will read into dataframes but machine learning handle arrays not dataframes
# So for this we were using numpy.
# Here if we use size or price directly we will get indexes of the values so, to remove the index we were suing reshape
# reshape (4,3) means 4 arrays each with 3 elements (columns)
# i.e reshape(-1,1) means 1 column and -1 means python will figure out the value
x = np.array(size).reshape(-1, 1)
y = np.array(price).reshape(-1, 1)

# we use Linear Regression + fit() to train the given model
# here we want to define linear relationship between price and sizes
# when we use fit method then sklearn will train the model using the Gradient Descent
model = LinearRegression()
model.fit(x, y)

# After training the model with the help of fit function we can evaluate MSE and R values
regression_model_mse = mean_squared_error(x,y)
print("MSE: ", math.sqrt(regression_model_mse))
print("R squared value: ", model.score(x,y))

# we can get b values after the model fit
# b0--> intercept between the linear regression model line and as far as the price access (y-intercept the value of y when x=0)
print(model.coef_[0])
# b1 --> slope of the give linear regression (slope or how y changes per unit increase in x)
print(model.intercept_[0])

# Visualize the data-set with the fitted model
plt.scatter(x, y, color='green')
plt.plot(x, model.predict(x), color='black')
plt.title("Linear Regression")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()

# Finally predicting prices based on size
print("Prediction by the model: ", model.predict([[2000]]))


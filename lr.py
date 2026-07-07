#importing numpy as np  
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
class LinearRegression:
    # initiating the class with learning rate and number of iterations
    def __init__(self,learning_rate, n_iterations):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        
    def fit(self, X, Y):
        # number of training examples and number of features
        self.m, self.n = X.shape #number of rows and columns
        # initializing weights and bias to zero
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        #implementing gradient descent
        for i in range(self.n_iterations):
            self.update_weights()

    def update_weights(self):
        Y_predicted = self.predict(self.X)
        #calculate gradients
        dw = -(2*(self.X.T).dot(self.Y-Y_predicted)) / self.m
        db = -(2*np.sum(self.Y-Y_predicted)) / self.m
        #update weights and bias
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def predict(self, X):
        return X.dot(self.w) + self.b
salary_data = pd.read_csv('salary_data.csv')
#printing the first 5 rows of the dataset
print(salary_data.head())
#last 5 rows of the dataset
print(salary_data.tail())
#printing the shape of the dataset
print(salary_data.shape)
#checking for null values in the dataset
print(salary_data.isnull().sum())
#splitting the dataset into features and target variable
X = salary_data.iloc[:, :-1].values# all rows and all columns except the last column
y = salary_data.iloc[:, -1].values # all rows and only the last column
print(X)
print(y)
#splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)  
model=LinearRegression(learning_rate=0.02, n_iterations=1000)    
model.fit(X_train, y_train)
#printing the parameters values (weights and bias) 
print("Weights:", model.w[0])
print("Bias:", model.b)
test_predictions = model.predict(X_test)
print("Predictions on test set:", test_predictions)
#visualizing the results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, test_predictions, color='red', label='Predicted')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience')
plt.legend()
plt.show()
print("R2 Score:", r2_score(y_test, test_predictions))
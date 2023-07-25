import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('/content/Netflix.csv')
print("NETFLIX STOCKS PRICE DATASET IS:\n",data)

# Preprocess data
data = data.dropna()  # Remove any missing values
data = data.reset_index(drop=True)  # Reset index after dropping rows
data['Date'] = pd.to_datetime(data['Date'])  # Convert date column to datetime
print("DATASET AFTER PREPROCESSING IS AS FOLLOWS:\n",data)

# Feature engineering  

#'Open' price refers to the price of the first trade that occurs for a particular stock on a given day.
#'High' price refers to the highest price at which a stock was traded during the day.
#'Low' price refers to the lowest price at which a stock was traded during the day.
#'Close' price refers to the last price at which a stock was traded at the end of the day.
#'Volume' refers to the total number of shares that were traded during the day.

#Additional Columns are added so that All the trends affecting the stock price should lead to better accuracy of prediction
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Weekday'] = data['Date'].dt.weekday
X = data[['Year', 'Month', 'Day', 'Weekday', 'Open', 'High', 'Low', 'Volume']]
y = data['Close']

# Plot the Close prices
data['Open'].plot(figsize=(16,8))
plt.title('Netflix Stock Prices (Close)')
plt.xlabel('Date')
plt.ylabel('ClosePrice (USD)')
plt.show()

# Split data into training and testing sets
train_size = int(len(data) * 0.8)  # Use 80% of data for training
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Decision Tree
def Decision_tree():
  print("ANALYSING THE PREDICTIONS USING DECISION TREE!")
  dt_model = DecisionTreeRegressor(max_depth=5)
  dt_model.fit(X_train, y_train)
  dt_pred = dt_model.predict(X_test)
  dt_mse = mean_squared_error(y_test, dt_pred)
  dt_rmse = sqrt(dt_mse)
  dt_r2 = r2_score(y_test, dt_pred)

  #Analysing the result 
  print("\nMean Squared Error For Decision Tree: ", dt_mse)
  print("\nRoot Mean Squared Error for Decision Tree:",dt_rmse)
  print("\nAccuracy Score for Decision Tree: ", dt_r2)

  #Plot the Decision Tree
  print("\nDECISION TREE IS AS FOLLOWS:")
  import matplotlib.pyplot as plt
  from sklearn.tree import plot_tree
  plt.figure(figsize=(20,10))
  plot_tree(dt_model, feature_names=X.columns, filled=True)
  plt.show()

  # Plot scatter plot for decision tree
  print("SCATTER PLOT FOR DECISION TREE IS AS:")
  plt.scatter(y_test, dt_pred, color='blue')
  plt.title('Decision Tree: Actual vs Predicted')
  plt.xlabel('Actual Stock Price')
  plt.ylabel('Predicted Stock Price')
  plt.show()

# Linear Regression
def Linear_Regression():
  print("ANALYSING THE PREDICTIONS USING LINEAR REGRESSION!")
  lr_model = LinearRegression()
  lr_model.fit(X_train, y_train)
  lr_pred = lr_model.predict(X_test)
  lr_mse = mean_squared_error(y_test, lr_pred)
  lr_rmse = (sqrt(lr_mse))
  lr_r2 = r2_score(y_test, lr_pred)

  # Analysing the results
  print("\nMean Squared Error For Linear Regression:", lr_mse)
  print("\nRoot Mean Squared Error for Linear Regression:", lr_rmse)
  print("\nAccuracy Score for Linear Regression: ", lr_r2)

  #PLOT THE LINEAR REGRESSION
  import matplotlib.pyplot as plt

  # Plotting the actual data
  plt.scatter(X_test['Open'], y_test, color='Red',label='Actual')
  # plt.plot(X_test['Open'], y_test, color='Green')
  # Plotting the linear regression line
  plt.scatter(X_test['Open'], lr_pred, color='Blue',label='Predicted')

  # Setting the plot title, x-label, and y-label
  plt.title("Linear Regression")
  plt.xlabel("Open Price")
  plt.ylabel("Closing Price")

  # Displaying the plot
  plt.show()
  print("\nIt is Prominently observed that the Accuracy Score is improved as compared to Decision tree!")
  
#print("THIS IS THE HEAT MAP REPRESENTING THE CORELATION BETWEEN ALL THE FEATURES PREDCITING THE STOCK PRICE IN FUTURE:")

def main():
  while True:
     print("\nPlease select an option from the menu:")
     print("1. Decision tree")
     print("2. Linear Regression")
     print("3. Exit")
     choice = input("Enter your choice (1 or 2 or 3): ")
     if choice == "1":
       Decision_tree()
     elif choice == "2":
       Linear_Regression() 
     elif choice == "3":
       print("\nHope We Helped you!")
       break
     else:
       print("Invalid choice! Please enter valid choice.")

main()


# Compute the correlation matrix
corr = data.corr()
top_corr_features = corr.index[abs(corr["Close"]) > 0.5]

# plot heatmap
print("THIS IS THE HEAT MAP REPRESENTING THE CORELATION BETWEEN ALL THE FEATURES PREDCITING THE STOCK PRICE IN FUTURE:")
plt.figure(figsize=(10,8))
sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="YlGnBu")
plt.show()

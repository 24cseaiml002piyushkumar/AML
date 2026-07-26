import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df = pd.read_excel("/content/cgpaPRED.xls")
print("Dataset loaded successfully")

print("first 5 rows:")
print(df.head())

print("\n Dataset shape:")
print(df.shape)

print("Column names:")
print(df.columns)

print("\n Informtions:")
print(df.info())

print("Missing values:")
print(df.isnull().sum())

label = "SEM 5"
print("Target variable:",label)

corr = df.corr(numeric_only = True)
print(corr)

print(corr["SEM 5"].sort_values(ascending = False))

plt.figure(figsize = (8,6))
sns.heatmap(
    corr,
    annot = True,
    cmap = "coolwarm"
)
plt.title("Correlation Map")
plt.show

plt.figure(figsize = (8,6))
sns.boxplot(data = df)
plt.title("BoxPlot")
plt.show()

X = df[["SEM 4"]]
y = df["SEM 5"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = LinearRegression()
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Simple Linear Regression:")
print("MAE: ",mean_absolute_error(y_test,pred))
print("MSE: ",mean_squared_error(y_test,pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_test,pred)))
print("R2 Score: ",r2_score(y_test,pred))

plt.figure(figsize = (8,6))
plt.scatter(X_test,y_test,color = 'blue')
plt.plot(X_test,pred,color = "red")
plt.xlabel("SEM 4")
plt.ylabel("SEMM 5")
plt.title("Simple Linear Regression")
plt.show()

X = df[["SEM 1","SEM 2","SEM 3","SEM 4"]]
y = df["SEM 5"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size = 0.20,
    random_state =42
)

mlr = LinearRegression()
mlr.fit(X_train,y_train)

prediction = mlr.predict(X_test)

print("Multiple Linear Regression:")
print("MAE :", mean_absolute_error(y_test,prediction))
print("MSE :", mean_squared_error(y_test,prediction))
print("RMSE :", np.sqrt(mean_squared_error(y_test,prediction)))
print("R2 Score :", r2_score(y_test,prediction))

coef  = pd.DataFrame({
    "Features":X.columns,
    "Coefficient": mlr.coef_
})
print(coef)

plt.figure(figsize = (8,6))
plt.scatter(y_test,prediction)
plt.plot(
    [y_test.min(),y_test.max()],
    [y_test.min(),y_test.max()],
    'r--'
)
plt.xlabel("Actual Value")
plt.ylabel("Predicted value")
plt.title("Actual vs Predicted")
plt.show()

print("Intercept: ",mlr.intercept_)

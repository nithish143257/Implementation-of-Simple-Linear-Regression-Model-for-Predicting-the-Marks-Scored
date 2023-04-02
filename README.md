# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import Pandas as pd & Import numpy as np
2. Calulating The y_pred & y_test
3. Find the graph for Training set & Test Set
4. Find the values of MSE,MSA,RMSE

### Program:


Program to implement the simple linear regression model for predicting the marks scored.
```
/*
Developed by: NITHISH KUMAR P
RegisterNumber: 212221040115
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores (1).csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="black")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_train,regressor.predict(X_train),color="green")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE= ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)


## Output:

### df.head():

![MODEL](head.png)

### df.tail():

![MODEL](tail.png)

### Array of  value X :

![MODEL](array.png)

### Array of value Y :

![MODEL](side.png)

### Values of Y Prediction :

![MODEL](point.png)

### Array values of Y test :

![MODEL](small.png)

### Training Set Graph :

![MODEL](training.png)

### Test  Set Graph :

![MODEL](testing.png)

### Values of MSE,MAE,RMSE :

![MODEL](mse.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

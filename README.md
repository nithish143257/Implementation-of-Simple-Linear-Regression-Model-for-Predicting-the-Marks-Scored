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
df=pd.read_csv('student_scores.csv')
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
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="green")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MSE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

## Output:

### df.head():

![image](https://user-images.githubusercontent.com/113762839/229332799-a59de621-50eb-463f-80ef-6ed8057019a9.png)

### df.tail():

![image](https://user-images.githubusercontent.com/113762839/229332855-be9ef753-c2e1-4200-8bb8-f6195d3b2d0d.png)

### Array of  value X :

![image](https://user-images.githubusercontent.com/113762839/229332890-605347d1-6085-4d59-8461-f290cc40357c.png)

### Array of value Y :

![image](https://user-images.githubusercontent.com/113762839/229332911-a94d5a0a-579f-42e6-94f2-a6ab601f9272.png)

### Values of Y Prediction :

![image](https://user-images.githubusercontent.com/113762839/229332924-3cad0dfc-3dff-4a6a-8920-7b4bdf0c38e0.png)

### Array values of Y test :

![image](https://user-images.githubusercontent.com/113762839/229332937-998cded1-bd90-44af-8add-e7f7e5350cac.png)

### Training Set Graph :

![image](https://user-images.githubusercontent.com/113762839/229332964-8caf1dbb-f83f-406c-81fc-8d6f2cd6765a.png)

### Test  Set Graph :

![image](https://user-images.githubusercontent.com/113762839/229332990-6e06bd6e-717c-46d2-bc71-a35c1176a5e8.png)

### Values of MSE,MAE,RMSE :

![image](https://user-images.githubusercontent.com/113762839/229333003-cb4c7f68-b906-4912-9cde-0314da1c3a9e.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

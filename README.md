# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Vanisha Ramesh
RegisterNumber:  212222040174
*/
```
```
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#removes the specified row or column
data1.head()

data1.isnull().sum() #to check for null values

data1.duplicated().sum() #to check for duplicate values

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")#A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)#accuracy score=(TP+TN)/(TP+FN+TN+FP),True+ve/
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
1.Placement Data

![image](https://github.com/Vanisha0609/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104009/afa4c8e7-837b-4cae-98fb-2276dd5f0c54)

2.Salary Data

![image](https://github.com/Vanisha0609/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104009/89fb5c8d-6a65-4270-b5ce-807c99315afe)

3.Checking the null() function

![image](https://github.com/Vanisha0609/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104009/ac1b0e91-590f-4a28-b295-46c8743363c9)

4.Data Duplicate

![image](https://github.com/Vanisha0609/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104009/bc595233-0b2b-4de0-9522-3e0b2bfb051c)

5.Print Data

![image](https://github.com/Vanisha0609/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104009/bce96665-5dd3-47fa-9b9d-7219024e401d)

6.Data-status

![image](https://github.com/Vanisha0609/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104009/7cc9e1e5-6d49-40c9-a306-fcb95edb5c89)

7.y_prediction array

![image](https://github.com/Vanisha0609/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104009/bd7fb1c3-1683-4cbd-a7f6-4ffa07735cc0)

8.Accuracy Value

![image](https://github.com/Vanisha0609/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104009/1c78b050-c7a7-458a-b969-42c3bffea20a)

9.Confusion Array

![image](https://github.com/Vanisha0609/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104009/ba8c7017-1f89-40e2-b2a7-1d61814c00e3)

10.Classification Report

![image](https://github.com/Vanisha0609/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104009/b71a4ee3-acd6-4d50-84cb-3628a1589201)

11.Prediction of LR

![image](https://github.com/Vanisha0609/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119104009/195860b8-8eac-4574-b28a-1dc4a543f373)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

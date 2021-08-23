# importing libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data collection -> kaggle,0 for healthy and 1 for PD(PARKINSONS DISEASE).
# loading data using pandas DataFrame
PD_dataset = pd.read_csv('parkinsons.csv')

#print first 5 row of parkinsons dataset
print(PD_dataset.head())
#print last 5 row of parkinsons dataset
print(PD_dataset.tail())
#number of rows and columns in the dataset
print(PD_dataset.shape)
#Getting more information
print(PD_dataset.info())
# checking missing value
print(PD_dataset.isnull().sum())
# Getting statistical measure
print(PD_dataset.describe())
#Destribution of target variable
print(PD_dataset['status'].value_counts())
#grouping the data based on the target variable
print(PD_dataset.groupby('status').mean())

# Data Pre-processing

# separating the features & Target

x=PD_dataset.drop(columns=['name','status'],axis=1)
y=PD_dataset['status']
print(x,y)

#Spliting Data to training and test data

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2,random_state=2)
print(x.shape,x_train.shape,x_test.shape)

#model training
#svm-support vector model

model =svm.SVC(kernel='linear')
model.fit(x_train,y_train)

# Model Accuracy Score

x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train,x_train_prediction)
print("Accuracy score of training data :", training_data_accuracy)


x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test,x_test_prediction)
print("Accuracy score of test data :", test_data_accuracy)

#Prediction process

#this input for 1 ->take it from dataset
"""#input_data = (119.99200,157.30200,74.99700,0.00784,0.00007,0.00370,0.00554,
              0.01109,0.04374,0.42600,0.02182,0.03130,0.02971,0.06545,0.02211,21.03300,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654)"""

# this input for 0
input_data =(237.22600,247.32600,225.22700,0.00298,0.00001,0.00169,0.00182,0.00507,0.01752,0.16400,0.01035,
             0.01024,0.01133,0.03104,0.00740,22.73600,0.305062,0.654172,-7.310550,0.098648,2.416838,0.095032)

#changing input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)

print(prediction)

if prediction[0] == 1:
    print("The person has Parkinsons Disease")
else:
    print("The person does not have Parkinsons Disease")
    
    

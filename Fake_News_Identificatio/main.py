# importing libraries
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')

#printing stopwords in english
print(stopwords.words('english'))

#loading the dataset to a pandas DataFrame

dataset = pd.read_csv('train.csv')
print(dataset.shape)
print(dataset.head())

#counting number of missing value in the dataset
dataset.isnull().sum()
#replacing the null values with empty string

dataset =dataset.fillna('')

# merging the author name and news title
dataset['content'] = dataset['author'] + ' '+ dataset['title']

print(dataset['content'])
# separating the data & label

x = dataset.drop(columns='label', axis=1)
y = dataset['label']
print(x,y)

#stemming -->  It is the process of reducing a word to its Root word

port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')                   ]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

dataset['content'] = dataset['content'].apply(stemming)
print(dataset['content'])

#separating the data and label

x=dataset['content'].values
y=dataset['label'].values

print(x,y)

# converting the textual data to numerical data

vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)
print(x)

#splitting the dataset to train and test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, stratify = y,random_state = 2)

#Trainig The Model
model = LogisticRegression()
model.fit(x_train,y_train)

#Find Accuracy score
# accuracy score on the training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

#Making Prediction
x_new = x_test[8]

prediction = model.predict(x_new)
print(prediction)

if prediction[0] == 0:
    print('This news is Real')
else:
    print('This new is Fake')

#checking with y_test value
print(y_test[8])
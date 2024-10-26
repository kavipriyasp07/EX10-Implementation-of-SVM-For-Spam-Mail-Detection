# EX 10 Implementation of SVM For Spam Mail Detection
## DATE:
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: Collect and preprocess the email dataset, converting text data into numerical features, often using techniques like TF-IDF.

2.Data Splitting: Split the dataset into training and testing sets to train and evaluate the model.

3.Model Training: Use a Support Vector Machine (SVM) classifier on the training set to learn to distinguish between spam and non-spam emails.

4.Model Evaluation: Test the trained model on the testing set and assess its accuracy in predicting spam versus non-spam emails.. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection.
Developed by:kavipriya s.p
RegisterNumber: 2305002011 
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
df=pd.read_csv('/content/spamEX10.csv',encoding='ISO-8859-1')
df.head()
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(df['v2'])
y=df['v1']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=svm.SVC(kernel='linear')
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,predictions))
print("Classification Report:")
print(classification_report(y_test,predictions))
def predict_message(message):
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)
    return prediction[0]
new_message="Free prixze money winner"
result=predict_message(new_message)
print(f"The message: '{new_message}' is classified as: {result}")
```

## Output:
![Screenshot 2024-10-26 122702](https://github.com/user-attachments/assets/f2096800-d194-41f1-83e6-b745ac78b8bb)
![Screenshot 2024-10-26 122711](https://github.com/user-attachments/assets/97b08069-70ea-4ca0-a793-db0ac177691f)
![Screenshot 2024-10-26 122724](https://github.com/user-attachments/assets/3b044dbc-3c84-481e-b212-99313dec83b9)
![Screenshot 2024-10-26 122732](https://github.com/user-attachments/assets/a5220edc-c41e-491d-b0ca-4a394fc9bc9a)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

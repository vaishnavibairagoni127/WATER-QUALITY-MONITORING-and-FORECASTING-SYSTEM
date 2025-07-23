from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
from django.core.files.storage import FileSystemStorage
import pymysql
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.layers import Dense, Dropout

import keras.layers
from sklearn.ensemble import RandomForestClassifier


global X, Y, dataset, X_train, X_test, y_train, y_test
global algorithms, accuracy, f1, precision, recall, classifier


def ProcessData(request):
    if request.method == 'GET':
        global X, Y, dataset, X_train, X_test, y_train, y_test
        dataset = pd.read_csv("Dataset/ml.csv")
        dataset.fillna(0, inplace = True)
        label = dataset.groupby('labels').size()
        columns = dataset.columns
        temp = dataset.values
        dataset = dataset.values
        X = dataset[:,2:dataset.shape[1]-1]
        Y = dataset[:,dataset.shape[1]-1]
        Y = Y.astype(int)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(columns)):
            output += "<th>"+font+columns[i]+"</th>"            
        output += "</tr>"
        for i in range(len(temp)):
            output += "<tr>"
            for j in range(0,temp.shape[1]):
                output += '<td><font size="" color="black">'+str(temp[i,j])+'</td>'
            output += "</tr>"    
        context= {'data': output}
        label.plot(kind="bar")
        plt.title("Water Quality Graph, 0 (Good quality) & 1 (Poor Quality)")
        plt.show()
        return render(request, 'UserScreen.html', context)
        

def TrainRF(request):
    global X, Y
    global algorithms, accuracy, fscore, precision, recall, classifier
    if request.method == 'GET':
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        cls = RandomForestClassifier()
        cls.fit(X, Y)
        classifier = cls
        predict = cls.predict(X_test)
        p = precision_score(y_test, predict,average='macro') * 100
        r = recall_score(y_test, predict,average='macro') * 100
        f = f1_score(y_test, predict,average='macro') * 100
        a = accuracy_score(y_test,predict)*100
        algorithms.append("Random Forest")
        accuracy.append(a)
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        arr = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th>"+font+arr[i]+"</th>"
        output += "</tr>"
        for i in range(len(algorithms)):
            output +="<tr><td>"+font+str(algorithms[i])+"</td><td>"+font+str(accuracy[i])+"</td><td>"+font+str(precision[i])+"</td><td>"+font+str(recall[i])+"</td><td>"+font+str(fscore[i])+"</td></tr>"
        context= {'data': output}
        return render(request, 'UserScreen.html', context)

def TrainLSTM(request):
    if request.method == 'GET':
        global X, Y
        global algorithms, accuracy, fscore, precision, recall
        algorithms = []
        accuracy = []
        fscore = []
        precision = []
        recall = []     
        X1 = np.reshape(X, (X.shape[0], X.shape[1], 1))
        Y1 = to_categorical(Y)
        print(X1.shape)
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.2)
        if request.method == 'GET':
            lstm_model = Sequential()
            lstm_model.add(keras.layers.LSTM(100,input_shape=(X_train.shape[1], X_train.shape[2])))
            lstm_model.add(Dropout(0.5))
            lstm_model.add(Dense(100, activation='relu'))
            lstm_model.add(Dense(y_train.shape[1], activation='softmax'))
            lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            lstm_model.fit(X1, Y1, epochs=40, batch_size=32, validation_data=(X_test, y_test))             
            print(lstm_model.summary())#printing model summary
            predict = lstm_model.predict(X_test)
            predict = np.argmax(predict, axis=1)
            testY = np.argmax(y_test, axis=1)
            p = precision_score(testY, predict,average='macro') * 100
            r = recall_score(testY, predict,average='macro') * 100
            f = f1_score(testY, predict,average='macro') * 100
            a = accuracy_score(testY,predict)*100
            algorithms.append("LSTM")
            accuracy.append(a)
            precision.append(p)
            recall.append(r)
            fscore.append(f)
            arr = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
            output = '<table border=1 align=center width=100%>'
            font = '<font size="" color="black">'
            output += "<tr>"
            for i in range(len(arr)):
                output += "<th>"+font+arr[i]+"</th>"
            output += "</tr>"
            for i in range(len(algorithms)):
                output +="<tr><td>"+font+str(algorithms[i])+"</td><td>"+font+str(accuracy[i])+"</td><td>"+font+str(precision[i])+"</td><td>"+font+str(recall[i])+"</td><td>"+font+str(fscore[i])+"</td></tr>"
            context= {'data': output}
            return render(request, 'UserScreen.html', context)

def Predict(request):
    if request.method == 'GET':
       return render(request, 'Predict.html', {})

def PredictAction(request):
    if request.method == 'POST':
        global classifier
        testFile = request.POST.get('t1', False)
        test = pd.read_csv("Dataset/testData.csv")
        test.fillna(0, inplace = True)
        test = test.values
        X = test[:,2:dataset.shape[1]-1]
        predict = classifier.predict(X)
        print(predict)
        arr = ['Test Data', 'Water Quality Forecasting Result']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th>"+font+arr[i]+"</th>"
        output += "</tr>"
        labels = ['Good Quality', 'Poor Quality'] 
        for i in range(len(predict)):
            output +="<tr><td>"+font+str(test[i])+"</td><td>"+font+str(labels[predict[i]])+"</td></tr>"
        context= {'data': output}    
        return render(request, 'UserScreen.html', context) 


def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})  

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Signup(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})


def UserLoginAction(request):
    global uname
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'Waterquality',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    uname = username
                    index = 1
                    break		
        if index == 1:
            context= {'data':'welcome '+uname}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed. Please retry'}
            return render(request, 'UserLogin.html', context)        

def SignupAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        gender = request.POST.get('t4', False)
        email = request.POST.get('t5', False)
        address = request.POST.get('t6', False)
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'Waterquality',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break
        if output == 'none':
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'Waterquality',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO signup(username,password,contact_no,gender,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+gender+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = 'Signup Process Completed'
        context= {'data':output}
        return render(request, 'Signup.html', context)
      



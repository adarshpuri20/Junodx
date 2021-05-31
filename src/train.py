# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:59:54 2021

@author: adars
"""

#Import the dependency for script

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import tensorflow as tf
import pickle



def data_preprocessing(df):
    '''The Function Normalizes the data for each column.
        Input is the training data frame, only keeps important features based on analysis
        and the output Data frame'''
    df['pred02'].iloc[np.where(df['pred02']<1)]=df['pred02'].iloc[np.where(df['pred02']<1)]*100
    df=df.drop(['ID','class01','class02','pred01','pred02','pred07'],axis=1)
    df=df.dropna()

    return df

# Path of Training data 
path='../Data/20210412_trainingdata.csv'

#Dataframe of Training Data
df=pd.read_csv(path)
print('\nNumber of NUll values in the data',df.isnull().sum())

#Drops the Nan Values
df=df.dropna()

#Splitting class01 and class02 in two variables
class01=df['class01']
class02=df['class02']

#one hot encoding each classes
class01=tf.keras.utils.to_categorical(
    class01, num_classes=4, dtype='float32')
class02=tf.keras.utils.to_categorical(
    class02, num_classes=4, dtype='float32')

#Concatenates the one hot encoded vectors
encoded_class=np.concatenate((class01,class02),axis=1)
print('\nShape of Encoded Class vector: ',encoded_class.shape)

#Pre Process Data
df_train=data_preprocessing(df)

#Train:validation split with 80:20
X_train, X_valid, y_train, y_valid = train_test_split(df_train.iloc[:,:5],encoded_class,
                                                    test_size=0.20, random_state=42)




#Random forest Classifier
rf = RandomForestClassifier(n_estimators=149, oob_score=True, random_state=42)

#fitting on train data
rf.fit(X_train, y_train)

#save the trained model weight parameters
filename = 'randomforest_weights.sav'
pickle.dump(rf, open(filename, 'wb'))

#threshold for no_call
threshold=0.9

#prediction on validation set
predicted = rf.predict(X_valid)

#custom script for accuracy and no call rate
correct,incorrect=0,0
no_call=len(predicted)
max_prob=[]
prob=rf.predict_proba(X_valid)
for i in range(len(prob)):
    max_prob.append(np.max(prob[i],axis=1))
    
for i in range(len(predicted)):
    if ((max_prob[0][i]+max_prob[1][i]+max_prob[2][i]+ max_prob[3][i]+max_prob[4][i]+max_prob[5][i]+max_prob[6][i] +max_prob[7][i])/8> threshold):
        if all(predicted[i]==y_valid[i]):
            correct+=1
        else:
            incorrect+=1
        
    else:
        no_call-=1

accuracy=correct/no_call
no_call_rate=(len(predicted)-no_call )/len(predicted)

print(f'\nOut-of-bag score estimate: {rf.oob_score_:.3}')
print(f'\nMean accuracy score: {accuracy:.3}')
print(f'\nNo Call Rate: {no_call_rate:.3}')

#each class prediction
predicted_class01,predicted_class02=np.argmax(predicted[:,:4],axis=1),np.argmax(predicted[:,4:],axis=1)

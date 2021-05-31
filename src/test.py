# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:46:37 2021

@author: adars
"""
import pandas as pd
import numpy as np
import pickle
from train import data_preprocessing as dpp
from train import threshold

#load the Trained Model
filename = './randomforest_weights.sav'
randomforest = pickle.load(open(filename, 'rb'))

# Path of Test data 
path='../Data/20210412_testdata.csv'

#Dataframe of Test Data
dataframe=pd.read_csv(path)
print('\nNumber of NUll values in the data',dataframe.isnull().sum())

#preprocess test data
df_test=dpp(dataframe)


#predcition on test data as encoded vectors
test_predicted = randomforest.predict(df_test.iloc[:,:])

#Custom Function to threshold to get no call
max_prob=[]
count=len(test_predicted)
final_pred=pd.DataFrame(columns=['class01_prediction','class02_prediction'])
prob_test=randomforest.predict_proba(df_test.iloc[:,:])
for i in range(len(prob_test)):
    max_prob.append(np.max(prob_test[i],axis=1))

for i in range(len(test_predicted)):
    if ((max_prob[0][i]+max_prob[1][i]+max_prob[2][i]+ max_prob[3][i]+max_prob[4][i]+max_prob[5][i]+max_prob[6][i] +max_prob[7][i])/8> threshold):
        final_pred=final_pred.append({'class01_prediction':np.argmax(test_predicted[i,:4]),
                                      'class02_prediction':np.argmax(test_predicted[i,4:])},ignore_index=True)
        
    else:
        final_pred=final_pred.append({'class01_prediction':'no-call',
                                      'class02_prediction':'no-call'},ignore_index=True)
        count-=1

no_call_rate_test=(len(test_predicted)-count )/len(test_predicted)    
print(f'\nNo Call Rate On test: {no_call_rate_test:.3}')

#Final Predictions saved in csv file in Data folder
final_pred.to_csv('../Data/test_predictions.csv')

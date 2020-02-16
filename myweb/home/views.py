from django.shortcuts import render

from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn import metrics   
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import pandas_profiling
from sklearn.ensemble import VotingClassifier

import pandas as pd 
import numpy as np
import seaborn as sns 
import requests
import json
import os
import csv


from bs4 import BeautifulSoup
from html_table_parser import parser_functions as parser
import requests

def write(request):
   
   return render(request, 'write.html')




url='http://www.syncsignal.co.kr/HunetGaia/RQCSVData.do?Name=전남부표1'
response = requests.get(url)
html = response.text 
soup = BeautifulSoup(html, 'html.parser')
table=soup.findAll('table')
p = parser.make2d(table[0])

if p[1][5] == 0.0:
    x6 = 1
else:
    x6=0


x1 = float(p[1][1])
x2 = float(p[1][2])
x3 = float(p[1][3])
x4 = float(p[1][4])


dff4 =pd.read_csv('home/적발생데이터.csv', encoding = 'cp949')
dff= dff4[['DATE','수온', '염분', '용존산소량', 'pH','발생']]


#dff.loc[len(dff)+1] = ['2019-09-01', 25, 32, 6, 7.11, 0]
dff.loc[len(dff)+1] = ['2019-09-01', x1, x2, x3, x4, x6]
dff = dff.set_index('DATE')

def MajorityVote_fit():
    
    bbc_dt = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(max_features=4, max_depth=5),
                                    sampling_strategy='auto',
                                    replacement=False,
                                    random_state=21)


    bbc_dt.fit(x_train, y_train)


    bbc_dt_file = 'bbc_dt.sav'
    joblib.dump(bbc_dt, open(bbc_dt_file,'wb'))

    preds_dt = bbc_dt.predict(x_test)

    bbc_rf = BalancedBaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=20, max_depth=6),
                                    sampling_strategy='auto',
                                    replacement=False,
                                    random_state=21)
    
   

    # Train the classifier.
    bbc_rf.fit(x_train, y_train)

    # save the model to disk
    bbc_rf_file = 'bbc_rf.sav'
    joblib.dump(bbc_rf, open(bbc_rf_file,'wb'))

    preds_rf = bbc_rf.predict(x_test)

    

    bbc_svm = BalancedBaggingClassifier(base_estimator=SVC(

                            C=1, cache_size=200, class_weight=None, coef0=1.0,
                            decision_function_shape='ovr', degree=2, gamma='auto', kernel='rbf',
                            max_iter=-1, probability=False, random_state=21, shrinking=True,
                            tol=0.001, verbose=False),

                                    sampling_strategy='auto',
                                    replacement=False,
                                    random_state=21)


    # Train the classifier.
    bbc_svm.fit(x_train, y_train)
    
    
    # save the model to disk
    bbc_svm_file = 'bbc_svm.sav'
    joblib.dump(bbc_svm, open(bbc_svm_file,'wb'))
    


    preds_svm = bbc_svm.predict(x_test)
    preds_svm = bbc_svm.predict(x_test)

    bbc_gnb = BalancedBaggingClassifier(base_estimator=GaussianNB(),

                                    sampling_strategy='auto',
                                    replacement=False,
                                    random_state=21)


    bbc_gnb.fit(x_train, y_train)
    preds_gnb = bbc_gnb.predict(x_test)


    bbc_gnb_file = 'bbc_gnb.sav'
    joblib.dump(bbc_gnb, open(bbc_gnb_file,'wb'))
    preds_gnb = bbc_gnb.predict(x_test)

    bbc_gbt = BalancedBaggingClassifier(base_estimator=GradientBoostingClassifier(n_estimators=150,
                                                                                  max_depth=3,
                                                                                  learning_rate = 2,
                                                                                  random_state=21),
                                    sampling_strategy='auto',
                                    replacement=False,
                                    random_state=21)


    bbc_gbt.fit(x_train, y_train)


    bbc_gbt_file = 'bbc_gbt.sav'
    joblib.dump(bbc_gbt, open(bbc_gbt_file,'wb'))
    

    preds_gbt = bbc_gbt.predict(x_test)
    

    preds_mjv = []
    
    for i in range(len(preds_dt)):
        preds_mjv.append(find_majority([preds_dt[i],preds_svm[i],preds_gbt[i], ]))
  
    앙상블성능 = metrics.accuracy_score(y_test, preds_mjv)

    
    return preds_mjv 


def find_majority(votes):
    vote_count = Counter(votes)
    top_two = vote_count.most_common(2)
    
    if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
        # It is a tie
        return 0
    
    return top_two[0][0]


x = dff[['수온', '염분', '용존산소량', 'pH']]
y = dff['발생']


upto = '2014-01-01'

num = 19  
test_from_1 = '20{}-01-01'.format(num)  
test_from_2 = '20{}-12-31'.format(num)

x = x[x.index >= upto]
y = y[y.index >= upto]

x_train = x[(x.index <test_from_1 ) | (x.index >test_from_2)]
y_train = y[(x.index <test_from_1 ) | (x.index >test_from_2)]

x_test = x[(x.index >= test_from_1) & (x.index <=test_from_2 )]
y_test = y[(y.index >= test_from_1) & (y.index  <=test_from_2 )]

preds_mjv_fit = MajorityVote_fit()
예측비교_test = pd.DataFrame(y_test)
예측비교_test['예측'] = preds_mjv_fit

Y_pred_test = preds_mjv_fit

예측비교_test = 예측비교_test.merge(dff ,left_index=True, right_index=True, how='left')

유의인자_범위 = 예측비교_test[(예측비교_test.발생_x ==1) & (예측비교_test.예측 ==1)].describe()
유의인자_범위 = 유의인자_범위[['수온','염분', '용존산소량', 'pH']]
유의인자_범위_main = 유의인자_범위[(유의인자_범위.index =='mean')|(유의인자_범위.index =='min') |(유의인자_범위.index =='max')]

bbc_dt = joblib.load('bbc_dt.sav')
bbc_svm = joblib.load('bbc_svm.sav')
bbc_gbt = joblib.load('bbc_gbt.sav')
bbc_gnb = joblib.load('bbc_gnb.sav')
bbc_rf = joblib.load('bbc_rf.sav')

ensemble = VotingClassifier(estimators=[('DT', bbc_dt), ('SVM', bbc_svm), ('GBT', bbc_gbt) ,('GNB', bbc_gnb),('RF', bbc_rf),], voting='soft')

probas = [c.fit(x_train, y_train).predict_proba(x_test) for c in (bbc_svm, bbc_gbt,bbc_dt, bbc_gnb,bbc_rf, ensemble)]


유의인자_범위_main['현재 적조발생확률']= probas[-1][-1][1]


def index(request):
	
	return render(request, 'index.html', {'message': probas[-1][-1][1]})

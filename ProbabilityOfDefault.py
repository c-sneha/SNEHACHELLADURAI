# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:36:13 2019

@author: Sneha
"""

##importing libraries and reading the dataset, removed addr_state,zip_code,,desc
import pandas as pd
import numpy as np
            
#%%
# loading the file, using text file.

cred=pd.read_csv('E:/project/Project_G4/XYZCorp_LendingData.txt', sep='\t'
                , header = 0, low_memory = False)
#%%
print(cred)         
#%%
##Checking column names 
cred.columns
#%%Checking maximum options and missing values
pd.set_option('display.max_columns',None)
cred.head()
cred.isnull().sum()
#%%Handling categorical variables
## term and payment plan into numeric

cred['term'] = pd.factorize(cred.term)[0]
print(cred.term)
cred['pymnt_plan'] = pd.factorize(cred.pymnt_plan)[0]
print(cred.pymnt_plan)
#%%Handling categorical variables
colname = ['grade','sub_grade']
print (colname)


print(cred['grade'])
#%%

from sklearn import preprocessing

le={}
for x in colname:
    le[x]=preprocessing.LabelEncoder()
    
for x in colname:
    cred[x]=le[x].fit_transform(cred[x])
#%%
print(cred.grade) 
print(cred.sub_grade)

#%%
df=cred.emp_length
print(cred['emp_length'])
#%% Changing emp_length to three broad categories and then making it numeric
df = df.replace(to_replace=r'< 1 year', value='low', regex=True)
df = df.replace(to_replace=r'1 year', value='low', regex=True)
df = df.replace(to_replace=r'2 years', value='medium', regex=True)
df = df.replace(to_replace=r'3 years', value='medium', regex=True)
df = df.replace(to_replace=r'4 years', value='medium', regex=True)
df = df.replace(to_replace=r'5 years', value='medium', regex=True)
df = df.replace(to_replace=r'6 years', value='medium', regex=True)
df = df.replace(to_replace=r'7 years', value='medium', regex=True)
df = df.replace(to_replace=r'8 years', value='medium', regex=True)
df = df.replace(to_replace=r'9 years', value='medium', regex=True)
#%%
df = df.replace(to_replace=r'10+ years', value='high')
#%%
cred['emp_length']=df
#%%
print(df)
#%% Chwcking the levels in emp_length
cred.groupby(['emp_length']).size() 

#%% Handling categorical variables- home_ownership 
cred.home_ownership.isnull().sum()
cred.groupby(['home_ownership']).size() #ANY MORTGAGE NONE OTHER OWN RENT 
ho=cred.home_ownership
h1 = ho.replace(to_replace=r'ANY', value='OTHER', regex=True) # any value conatin only 3 obs
type(h1)
cred['home_ownership']=h1
print(cred['home_ownership'])
#%%
cred.groupby(['home_ownership']).size() 
#%% Handling categorical variables
colname = ['home_ownership']
print (colname)


from sklearn import preprocessing

le={}
for x in colname:
    le[x]=preprocessing.LabelEncoder()
    
for x in colname:
    cred[x]=le[x].fit_transform(cred[x])
#%%
print(cred.home_ownership)
 #%%
print(cred.application_type)
cred.application_type.isnull().sum()
cred.groupby(['application_type']).size()
 #%%
#### ---->> drop the JOINT  value here and check the over all accuracy

cred['application_type'] = pd.factorize(cred.application_type)[0]
print(cred.application_type)
 #%%
cred.groupby(['initial_list_status']).size()
cred.initial_list_status.isnull().sum()
cred['initial_list_status'] = pd.factorize(cred.initial_list_status)[0]
print(cred.initial_list_status)
#%%
cred.groupby(['verification_status']).size()
cred['verification_status'] = pd.factorize(cred.verification_status)[0]
print(cred.verification_status)
#%%
cred.isnull().sum()
#%%
print(cred.emp_length)

#%%
#cred.drop('emp_length_New', axis=1, inplace=True)
#%%
cred.emp_length.isnull().sum()
#%%
cred.shape
#%%
cred.groupby(['collections_12_mths_ex_med']).size ()
#%%
for value in['collections_12_mths_ex_med']:
    cred[value].fillna(cred[value].mode()[0],
                   inplace=True)
#%%
cred.collections_12_mths_ex_med.isnull().sum()
#%%
cred.groupby(['emp_length']).size ()
#%%
for value in['emp_length']:
    cred[value].fillna(cred[value].mode()[0],
                   inplace=True)
#%%    
cred.emp_length.isnull().sum()
#%%
cred_new=cred.drop(['open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 
                    'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 
                    'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 
                    'inq_last_12m','mths_since_last_record','desc','zip_code','addr_state'],axis=1)
#%%
cred_new.mths_since_last_delinq.mode()
cred_new.groupby(['mths_since_last_delinq']).size ()
#%%
cred_new.mths_since_last_delinq.isnull().sum()
#%%
for value in['mths_since_last_delinq']:
    cred_new[value].fillna(cred_new[value].mode()[0],
                   inplace=True)
#%%
 
#%%
cred_new.isnull().sum() 
cred_new.emp_length.isnull().sum()
#%%
cred_new['emp_length'] = pd.factorize(cred_new.emp_length)[0]
print(cred_new.emp_length)
#%%
print(cred_new.revol_util)
#%%
cred_new.revol_util.dtype
cred_new.revol_util.mean()

#%%replacing NAs in revol_util with mean
cred_new['revol_util'].fillna((cred_new['revol_util'].mean()), inplace=True)
#%%
cred_new.isnull().sum()
#%%
cred_new.head
#%%
##cred_new=cred(['annual_inc_joint','dti_joint','verification_status_joint'],axis=1)
#%%
cred_new.shape
#%%
cred_new1=cred_new.drop(['annual_inc_joint','dti_joint','verification_status_joint'],axis=1)
#%%
cred_new1.isnull().sum()
#%%
cred_new1.groupby(['tot_coll_amt']).size ()
#%%
boxplot = cred_new1.boxplot(column=['annual_inc', 'verification_status',
       'dti', 'delinq_2yrs','inq_last_6mths', 'mths_since_last_delinq', 'open_acc',
       'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
       'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
       'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
       'collection_recovery_fee', 'last_pymnt_amnt','collections_12_mths_ex_med','mths_since_last_major_derog','application_type',
       'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim'])
#%%
    

boxplot=cred_new1.boxplot('annual_inc')# yes
boxplot=cred_new1.boxplot('verification_status')#no
boxplot=cred_new1.boxplot('dti') #yes
boxplot=cred_new1.boxplot('delinq_2yrs')#yes
boxplot=cred_new1.boxplot('inq_last_6mths')    #yess
boxplot=cred_new1.boxplot('mths_since_last_delinq')#yes
boxplot=cred_new1.boxplot('open_acc')    #yes
boxplot=cred_new1.boxplot('revol_bal')#yes
boxplot=cred_new1.boxplot('revol_util')   #yes 
boxplot=cred_new1.boxplot('total_acc')#yes
boxplot=cred_new1.boxplot('initial_list_status')  #  
boxplot=cred_new1.boxplot('out_prncp')#yes
boxplot=cred_new1.boxplot('out_prncp_inv') #yes
boxplot=cred_new1.boxplot('total_pymnt')#yes
boxplot=cred_new1.boxplot('total_pymnt_inv') #yes
boxplot=cred_new1.boxplot('total_rec_prncp')#yes
boxplot=cred_new1.boxplot('total_rec_int')#yes
boxplot=cred_new1.boxplot('total_rec_late_fee')#yes
boxplot=cred_new1.boxplot('recoveries')#yes
boxplot=cred_new1.boxplot('collection_recovery_fee')#yes
boxplot=cred_new1.boxplot('last_pymnt_amnt')#yes
boxplot=cred_new1.boxplot('collections_12_mths_ex_med')#yes
boxplot=cred_new1.boxplot('mths_since_last_major_derog')#yes
#boxplot=cred_new1.boxplot('application_type')
boxplot=cred_new1.boxplot('acc_now_delinq')#yes
boxplot=cred_new1.boxplot('tot_coll_amt')#yes
boxplot=cred_new1.boxplot('tot_cur_bal')#yes
boxplot=cred_new1.boxplot('total_rev_hi_lim')#yes




 #%%
cred_new2=cred_new1.drop(['last_pymnt_d','next_pymnt_d','mths_since_last_major_derog','last_credit_pull_d'],axis=1)
#%%
cred_new2.isnull().sum()
print(cred_new2.tot_coll_amt,cred_new2.tot_cur_bal,cred_new2.total_rev_hi_lim)
cred_new2.tot_coll_amt.mode()
cred_new2.tot_cur_bal.mode()
cred_new.total_rev_hi_lim.mode()
#%%
for value in['tot_coll_amt']:
    cred_new2[value].fillna(cred_new2[value].mode()[0],inplace=True)
    
    #%%
for value in['tot_cur_bal']:
    cred_new2[value].fillna(cred_new2[value].mode()[0],
                   inplace=True) 
#%%  
cred_new.total_rev_hi_lim.mean()
cred_new2['total_rev_hi_lim'].fillna((cred_new2['total_rev_hi_lim'].mean()), inplace=True)   
#%%
cred_new2.isnull().sum()
#%%
print(cred_new2) 
#%%
##X= cred_new2.values[:,:-1]
##Y=cred_new2.values[:,-1]
#%%print(cred_new2)
#%%
cred_new2.to_csv('E:/project/Project_G4/cred_new2.csv')
#%% Sorting the data according to date
cred_new3=cred_new2.sort_values(by='issue_d')
#%%
print(cred_new3)
#%%
print(cred_new3.issue_d)
#%%
#%%
#%% Dropping issue date and earliest cr line for finding K-best features
cred_new4=cred_new3.drop(['earliest_cr_line','issue_d','emp_title','title'
                          ,],axis=1)
#%%
cred_new4['purpose'] = pd.factorize(cred_new4.purpose)[0]
print(cred_new4.purpose)
#%%
cred_new4.to_csv('cred_new4.csv',index=False, header= True)

#%% Splitting into X and Y for feature selection
X= cred_new4.values[:,:-1]
Y= cred_new4.values[:,-1]

dtype(cred_new4['application_type'])
#%%
## adding this line because it is giving me error in 
#rfe = RFE(classifier, 25)



#%%
# RFE recursive feature elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
classifier=(LogisticRegression())
rfe = RFE(classifier, 25)
model_rfe = rfe.fit(X, Y)
print("Num Features: ",model_rfe.n_features_)
print("Selected Features: ") 
print(list(zip(colname, model_rfe.support_)))
print("Feature Ranking: ", model_rfe.ranking_)         
#%%
cred_new5=cred_new3.drop(['earliest_cr_line','emp_title','title'],axis=1)

print (cred_new5.head(0))
#%%
cred_new5['purpose'] = pd.factorize(cred_new5.purpose)[0]
#%%
# now sorting the df with issue date
cred_new5=cred_new5.sort_values(by='issue_d')
#%%
#creating df with only 25 selected variables as given by RFE + issue date to split
print (cred_new5.head(0))
#%%
cred_new6=cred_new5.drop(['id', 'member_id','loan_amnt','term','installment',
                          'annual_inc','pymnt_plan','delinq_2yrs','inq_last_6mths',
                          'pub_rec', 'revol_bal','initial_list_status','collections_12_mths_ex_med',
                          'application_type', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal'
                          , 'total_rev_hi_lim'],axis=1)
#%%
print (cred_new6.head(0))
#%%
#converting issuedate to date time
cred_new6.issue_d=pd.to_datetime(cred_new6.issue_d)
col_name='issue_d'
print (cred_new6[col_name].dtype)
#%%
#split by issue date and delete issue date
split_date = "2015-06-01"

train = cred_new6.loc[cred_new6['issue_d'] <= split_date]
train = train.drop(['issue_d'],axis=1)

test = cred_new6.loc[cred_new6['issue_d'] > split_date]
test = test.drop(['issue_d'],axis=1)
#%%
# Creating X train and Ytrain and Xtest and Ytest
X_train= train.values[:,:-1]
Y_train= train.values[:,-1]

X_test= test.values[:,:-1]
Y_test= test.values[:,-1]

#%%
Y_train=Y_train.astype(int)
Y_test=Y_test.astype(int)

#%%

#%% Importing sklearn for Logistic regression model
from sklearn.linear_model import LogisticRegression
##create a model
classifier=LogisticRegression()
##fitting training data to the model
classifier.fit(X_train, Y_train)

Y_pred=classifier.predict(X_test)
print(list(zip(Y_test, Y_pred)))
#%% Checking the confusion matrix and classification report
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
cfm=confusion_matrix(Y_test, Y_pred)
print(cfm)

print("Classification report: ")
print(classification_report(Y_test, Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)

#%% Checking roc curve and predicted probabilities
import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = classifier.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
#%%Checking roc curve
import matplotlib.pyplot as  plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#%%
print(Y_pred)
#%%
# store the predicted probabilities
y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)

#%%
 
y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value > 0.40:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)
#%%

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,y_pred_class)
print(cfm)
acc=accuracy_score(Y_test, y_pred_class)
print("Accuracy of the model: ",acc)
print(classification_report(Y_test, y_pred_class))
#%%
#adjusting the thershold

for a in np.arange(0,1,0.1):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", 
          cfm[1,0]," , type 1 error:", cfm[0,1])
#%%
    # **Running model using cross validation**


#Using cross validation

classifier=(LogisticRegression())

#performing kfold_cross_validation
from sklearn.model_selection import KFold
kfold_cv=KFold(n_splits=10)
print(kfold_cv)

from sklearn.model_selection import cross_val_score
#running the model using scoring metric as accuracy
kfold_cv_result=cross_val_score(estimator=classifier,X=X_train,
                                                 y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())


for train_value, test_value in kfold_cv.split(X_train):
    classifier.fit(X_train[train_value], Y_train[train_value]).predict(X_train[test_value])

    
Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusion_matrix=confusion_matrix(Y_test,Y_pred)
print(confusion_matrix)
print()


print("Classification report: ")

print(classification_report(Y_test,Y_pred))

accuracy_score=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",accuracy_score)
#%%
"""predicting using the Gradient_Boosting_Classifier
Default trees is 100"""
from sklearn.ensemble import GradientBoostingClassifier

model_GradientBoosting=GradientBoostingClassifier(random_state=10)

#fit the model on the data and predict the values
model_GradientBoosting.fit(X_train,Y_train)

Y_pred=model_GradientBoosting.predict(X_test)

 

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
#%%
# predicting using Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
model_DecisionTree= DecisionTreeClassifier()
# fit the model on the 
model_DecisionTree.fit(X_train,Y_train)

Ypred=model_DecisionTree.predict(X_test)
print (Ypred)
print (list(zip(Y_test,Ypred)))

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print(confusion_matrix(Y_test,Ypred))
print(accuracy_score(Y_test,Ypred))
print(classification_report(Y_test,Ypred))
#%%
# predicting using random forest

from sklearn.ensemble import RandomForestClassifier

model_RandomForest=RandomForestClassifier(201,random_state=10)
""" default is 10"""
#fit the model on the data and predict the values
model_RandomForest.fit(X_train,Y_train)

Y_pred=model_RandomForest.predict(X_test)

 

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
#%%
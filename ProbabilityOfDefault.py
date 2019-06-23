# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:38:46 2019

@author: Sachin
"""

import pandas as pd



#%%
df1=pd.read_csv('E:/Manoj Malpani/DSP Data/Python/XYZCorp_LendingData.txt', sep='\t',
                infer_datetime_format = True , na_values = 'NaN', header = 0,
                low_memory = False)
#%%
pd.set_option('display.max_columns', None)
df1.isnull().sum()
df1.columns.values.tolist()

#%%
df1.columns[df1.isnull().any()].tolist()
#%%
df1=df1.drop(['desc',
 'mths_since_last_delinq',
 'mths_since_last_record',
 'revol_util',
 'mths_since_last_major_derog',
 'annual_inc_joint',
 'dti_joint',
 'verification_status_joint',
 'open_acc_6m',
 'open_il_6m',
 'open_il_12m',
 'open_il_24m',
 'mths_since_rcnt_il',
 'total_bal_il',
 'il_util',
 'open_rv_12m',
 'open_rv_24m',
 'max_bal_bc',
 'all_util',
 'inq_fi',
 'total_cu_tl',
 'inq_last_12m'],axis = 1)

#%%
df1.isnull().sum()
#%%
print(df1.next_pymnt_d.value_counts())
print(df1.last_pymnt_d.value_counts()) # 8862 mv
#%%
df1 = df1.drop(['last_pymnt_d','next_pymnt_d'],axis=1)
#%%
df1.isnull().sum()
print(df1.emp_title.value_counts())
print(df1.earliest_cr_line.value_counts())









































import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder
test_df=pd.read_csv('UNSW_NB15_testing-set.csv')
tra_df = pd.read_csv('UNSW_NB15_training-set.csv')

train_df = pd.concat([test_df,tra_df])

lb_make = LabelEncoder()
train_df[['proto']]= lb_make.fit_transform(train_df[['proto']])

train_df[['attack_cat']]= lb_make.fit_transform(train_df[['attack_cat']])
train_df[['service']]= lb_make.fit_transform(train_df[['service']])
train_df[['state']]= lb_make.fit_transform(train_df[['state']])

print(train_df.info())
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_df = sc.fit_transform(train_df)
my_df = pd.DataFrame(train_df)
my_df.to_csv("Dataset_Transformed.csv")
#train_df = pd.DataFrame(train_df)
#print(train_df.info())
#preprocessing(train_df.head())

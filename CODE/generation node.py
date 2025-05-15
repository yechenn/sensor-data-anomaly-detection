# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:29:06 2022

@author: YT
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats


import csv


m='input your number'
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

f=open('path')
df=pd.DataFrame(pd.read_csv(f))
df=df.replace(np.nan,0.1)

l=len(df['num1'])




f1=open('path')
df1=pd.DataFrame(pd.read_csv(f1))
maxx=df1['max']
minn=df1['min']




row=[]
for i in range(m):
    row.append(df['num%d'%i])

new_row=[]
for j in range(l):
    node = np.zeros((m,1))
    for i in range(m):
        if (maxx[i]-minn[i])!=0:
            node[i]=(row[i][j]-minn[i])/(maxx[i]-minn[i])
        else:
            node[i]=0.5
            
    if j %2000==0:
        print(j)
    np.save("path/"+str(j),node) 
        





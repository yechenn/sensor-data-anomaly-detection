# -*- coding: utf-8 -*-





import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats


import csv

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

f=open('path')
df=pd.DataFrame(pd.read_csv(f))
df=df.replace(np.nan,0.1)


m='input your number'
l={}


for i in range(m):
    l['num%d'%(i)]=df['num%d'%(i+1)]

    
  
l=pd.DataFrame(l)
b=l.corr('spearman')




c=[]
k=[]
for i in range(m):
    for j in range(m):
        c=[]
        if i == j:
            continue
        if abs(b['num%d'%i]['num%d'%j])>0.75:
            
            c.append(i)
            c.append(j)
            k.append(c)

np.save("path",k)  



print(k)
print(len(k))
            














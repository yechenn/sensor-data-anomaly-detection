# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:30:51 2022

@author: YT
"""


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats

import seaborn as sns
import csv





m='input your number'

f=open('path')
df=pd.DataFrame(pd.read_csv(f))
lamda=df['lamda']




l=[]
he=[]
bb=[]
c=-1
for i in range(m):
 
    try:
        
        value=df['num%d'%(i+1)]
        value=value.replace(0,0.01)
        c+=1
        value=abs(value)
        he.append('num%d'%(i+1))
        
        
        valuee = stats.boxcox(value, lamda[c])
        
        print(i)
       
        
        l.append(valuee)
    except:
        p=0


print(len(l))
csvFile = open('path', "w",newline='')
writer = csv.writer(csvFile) 

writer.writerow(he)
    
csvFile.close()






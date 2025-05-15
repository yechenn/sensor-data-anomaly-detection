# -*- coding: utf-8 -*-



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


he=[]

mean=[]
var=[]

for i in range(m):
    try:
        value=df['num%d'%(i+1)]
        he.append('num%d'%(i+1))
        mean.append(np.mean(value))
        var.append(np.var(value))

    except:
        p=1
        
csvFile = open('path', "w",newline='')
writer = csv.writer(csvFile) 

writer.writerow(he)
writer.writerow(mean)
writer.writerow(var)

csvFile.close()



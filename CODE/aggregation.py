# -*- coding: utf-8 -*-




import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import csv

m='input your number'
    
f=open('path')
df=pd.DataFrame(pd.read_csv(f))
value=[]
for i in range(m):
    
    value.append(df['num%d'%(i+1)].tolist())


h=10
p=int(len(value[0])/h)

csvFile = open('path', "w",newline='')
writer = csv.writer(csvFile) 
he=[]

for i in range(m):
    he.append('num%d'%(i+1))
    
writer.writerow(he)

for i in range(p):
    l=[]
    for n in range(m):
        
        hee=0
        for j in range(h):
            hee+=value[n][i*h+j]
        l.append(hee)
    writer.writerow(l)
    
csvFile.close()
    











    
    
    
    
    
    
    
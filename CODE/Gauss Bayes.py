# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 19:01:37 2022

@author: YT
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats

import seaborn as sns
import csv
import math


f=open('path')
df=pd.DataFrame(pd.read_csv(f))

m='input your number'
n='input your number'

normal_mean=[]
normal_var=[]



for i in range(m):
    try:
        normal_value=df['num%d'%(i+1)]
        normal_mean.append(normal_value[0])
        normal_var.append(normal_value[1])
    except:
        p=0



f=open('path')
df=pd.DataFrame(pd.read_csv(f))


abnormal_mean=[]
abnormal_var=[]



for i in range(m):
    try:
        abnormal_value=df['num%d'%(i+1)]
        abnormal_mean.append(abnormal_value[0])
        abnormal_var.append(abnormal_value[1])
    except:
        p=0







f=open('path')
df=pd.DataFrame(pd.read_csv(f))

normal_valuee=[]


c=-1
for i in range(m):
    try:
        h=df['num%d'%(i+1)]
       
            
        normal_valuee.append(h)
    except:
        p=0

normal=[]

for j in range(len(df['num1'])):
    normall=[]
    for i in range(n):
        normall.append(normal_valuee[i][j])
    normal.append(normall)



normal_flag=0
for i in range(len(normal)):
    summ=0
    for j in range(len(normal[1])):
        p0=(math.exp(-(normal[i][j]-normal_mean[j])**2/2*normal_var[j])/math.sqrt(2*math.pi*normal_var[j]))*('ration')
        p1=(math.exp(-(normal[i][j]-abnormal_mean[j])**2/2*abnormal_var[j])/math.sqrt(2*math.pi*abnormal_var[j]))*('ration')
        if p0>p1:
            summ+=1
    if summ>'threshold':
        normal_flag+=1
            
print(normal_flag)
print(len(normal))








f=open('path')
df=pd.DataFrame(pd.read_csv(f))

abnormal_valuee=[]

c=-1
for i in range(m):
    try:
      
       h=df['num%d'%(i+1)]

           
       abnormal_valuee.append(h)
    except:
        p=0


abnormal=[]

for j in range(len(df['num1'])):
    abnormall=[]
    for i in range(n):

        abnormall.append(abnormal_valuee[i][j])
    abnormal.append(abnormall)



abnormal_flag=0
for i in range(len(abnormal)):
    
    summ=0
    for j in range(len(abnormal[1])):
        p0=(math.exp(-(abnormal[i][j]-normal_mean[j])**2/2*normal_var[j])/math.sqrt(2*math.pi*normal_var[j]))*('ration')
      
        
        p1=(math.exp(-(abnormal[i][j]-abnormal_mean[j])**2/2*abnormal_var[j])/math.sqrt(2*math.pi*abnormal_var[j]))*('ration')
  
        if p0<=p1:
            summ+=1
    if summ>'threshold':
        abnormal_flag+=1
    
print(abnormal_flag)
print(len(abnormal))




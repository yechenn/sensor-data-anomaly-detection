# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:23:57 2022

@author: YT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from torch_geometric.nn import GCNConv
from torch_scatter import  scatter_max

from load_wadi_graph import load_mnist_graph_train, load_mnist_graph_test
from sklearn.metrics import precision_score, recall_score, f1_score



normal_train_size='input your number'
normal_test_size='input your number'

abnormal_train_size='input your number'
abnormal_test_size='input your number'



batch_size = 'input your number'
epoch_num = 'input your number'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)






class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 64)
        self.conv4 = GCNConv(64, 128)
        self.conv5 = GCNConv(128, 256)
        self.conv6 = GCNConv(256, 512)

    
        self.n_layers = 2
        self.hidden_dim = 4
        self.lstm1 = nn.LSTM(16,16,4,batch_first=True)
        self.lstm2 = nn.LSTM(32,32,4,batch_first=True)
        self.lstm3 = nn.LSTM(64,64,4,batch_first=True)
        self.lstm4 = nn.LSTM(128,128,4,batch_first=True)
        self.lstm5 = nn.LSTM(256,256,4,batch_first=True)


        
        
        

        self.classifier = nn.Linear(512,2)


 

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
    
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        b=x
       
        x=torch.unsqueeze(x,1)
        out,(h_n,c_n) = self.lstm1(x)
        x = h_n[-1,:,:]
        x=b+x
        

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        b=x
        
        x=torch.unsqueeze(x,1)
        out,(h_n,c_n) = self.lstm2(x)
        x = h_n[-1,:,:]
        x=b+x
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        b=x
        
        x=torch.unsqueeze(x,1)
        out,(h_n,c_n) = self.lstm3(x)
        x = h_n[-1,:,:]
        x=b+x
        
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        b=x
        
        x=torch.unsqueeze(x,1)
        out,(h_n,c_n) = self.lstm4(x)
        x = h_n[-1,:,:]
        x=b+x
        
        
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        b=x
         
        x=torch.unsqueeze(x,1)
        out,(h_n,c_n) = self.lstm5(x)
        x = h_n[-1,:,:]
        x=b+x
        
        
        x = self.conv6(x, edge_index)
        x = F.relu(x)
        x, _ = scatter_max(x, data.batch, dim=0)

        
        x = self.classifier(x)

        return x




def main():
    #前準備
    trainset = load_mnist_graph_train(normal_train_size,abnormal_train_size)
    testset = load_mnist_graph_test(normal_test_size,abnormal_test_size)
    
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
       
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
 
    testloader = DataLoader(testset, batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss()
    
    history = {
        "train_loss": [],
        "test_loss": [],
        "test_acc": []
    }

    print("Start Train")
    


    #学習部分
    model.train()
    for epoch in range(epoch_num):
        train_loss = 0.0
        for i, batch in enumerate(trainloader):
         
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs,batch.t)
            loss.backward()
            optimizer.step()
   
            train_loss += loss.cpu().item()
        print('epoch:',epoch,'train_loss:',train_loss)          


        correct = 0
        total = 0
   
        y_pred=[]
        y_true=[]
        
        with torch.no_grad():
            for data in testloader:
                data = data.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                for j in range(len(predicted)):
                    y_pred.append(predicted.cpu().tolist()[j])
                    y_true.append(data.t.cpu().tolist()[j])
                total += data.t.size(0)
              
                correct += (predicted == data.t).sum().cpu().item()

            p=precision_score(y_true, y_pred,average='binary')
            r=recall_score(y_true, y_pred,average='binary')
            f1=f1_score(y_true, y_pred,average='binary')
            acc=correct/total
            print('epoch:',epoch,'p:',p,'r:',r,'f1',f1,'acc',acc)

        
        


    #最終結果出力
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += data.t.size(0)
            correct += (predicted == data.t).sum().cpu().item()
    print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)))

if __name__=="__main__":
    main()




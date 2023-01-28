import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader



positive_distribution = np.random.multivariate_normal([2.5,0], np.identity(2), size=750)
ones = torch.ones([750,1]);
positive_distribution = np.append(positive_distribution,ones,1);
negative_distribution = np.random.multivariate_normal([-2.5,0], np.identity(2), size = 750)
minusOnes = torch.zeros([750,1]);
#minusOnes = minusOnes * -1
negative_distribution = np.append(negative_distribution,minusOnes,1);
total_dataset = np.append(positive_distribution, negative_distribution, 0)

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim).double()
        
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

class customDataSet(Dataset):
    def __init__(self):

        positive_distribution = np.random.multivariate_normal([2.5,0], np.identity(2), size=750)
        ones = torch.ones([750,1]);
        positive_distribution = np.append(positive_distribution,ones,1);
        negative_distribution = np.random.multivariate_normal([-2.5,0], np.identity(2), size = 750)
        minusOnes = torch.ones([750,1]);
        minusOnes = minusOnes * -1
        negative_distribution = np.append(negative_distribution,minusOnes,1);
        self.total_dataset = np.append(positive_distribution, negative_distribution, 0)
        
    def __len__(self):
        return len(self.total_dataset)
  
    def __getitem__(self, idx):
        return total_dataset[idx];

w = torch.zeros(2, 1, dtype=torch.float)
b = torch.scalar_tensor(0)
ds = customDataSet()
train_loader = torch.utils.data.DataLoader(dataset=ds, 
                                           batch_size=32, shuffle=True)
criterion = torch.nn.BCELoss()
lr_model = LogisticRegression(2,1)

optimizer = torch.optim.SGD(lr_model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    correct = 0.
    total = 0.
    for i, input in enumerate(train_loader):
        optimizer.zero_grad()
        x = input[:,:-1];
        y = input[:,2];
        outputs = lr_model(x)
        y = y.reshape([y.size(0),1])
        loss_lr = criterion(outputs, y)
        loss_lr.backward()
        optimizer.step()
    if(epoch%2 == 0):
        for input in train_loader:
            x = input[:,:-1];
            y = input[:,2];
            outputs = lr_model(x)   
            predicted = outputs.data >= 0.5
            real = y >= 0.5
            total += y.size(0) 
            correct += (predicted.view(-1).long() == real).sum()    
        print(correct, total, correct/total, loss_lr);
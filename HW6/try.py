import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
torch.manual_seed(0)
np.random.seed(0)

circles_dataset = make_circles(1500, factor=0.1, noise=0.1)
X_train, labels_train = circles_dataset
X = torch.FloatTensor(X_train)
labels = torch.FloatTensor(labels_train)

X_train, X_test, Y_train, Y_test = train_test_split(X, labels,test_size=(1-5/6), random_state=73)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,test_size=(1-4/5), random_state=73)


# X_train = (X_train).double()
# X_test = (X_test).double()
# X_val = (X_val).double()

# Y_train = (Y_train).double()
# Y_test = (Y_test).double()
# Y_val = (Y_val).double()

# df = pd.DataFrame(dict(x1=X_train[:, 0], x2=X_train[:, 1], y=Y_train))
# colors = {0:'red', 1:'blue'}
# fig, ax = plt.subplots()
# grouped = df.groupby('y')
# for key, group in grouped:
#     group.plot(ax=ax, kind='scatter', x='x1', y='x2', label=key, color = colors[key])
# #plt.show()

# df = pd.DataFrame(dict(x1=X_test[:, 0], x2=X_test[:, 1], y=Y_test))
# colors = {0:'red', 1:'blue'}
# fig, ax = plt.subplots()
# grouped = df.groupby('y')
# for key, group in grouped:
#     group.plot(ax=ax, kind='scatter', x='x1', y='x2', label=key, color = colors[key])
# n_input_dim = X_train.shape[1]

# ##PLAIN LOGISTIC REGRESSION
# # class LogisticRegression(torch.nn.Module):
# #     def __init__(self, input_dim, output_dim):
# #         super(LogisticRegression, self).__init__()
# #         self.linear = torch.nn.Linear(input_dim, output_dim).double()
        
# #     def forward(self, x):
# #         outputs = torch.sigmoid(self.linear(x))
# #         return outputs

# # criterion = torch.nn.BCELoss()
# # lr_model = LogisticRegression(X_train.shape[1],1)
# # optimizer = torch.optim.SGD(lr_model.parameters(), lr=0.1)
# # num_epochs = 1000

# def accuracy(X, Y_train, model, src):
#   output = model(X);
#   predicted = output.data > 0.5
#   print(src, accuracy_score(predicted, Y_train),X.size(0))

# # accuracy(X_train,Y_train,lr_model, 'train')
# # for epoch in range(300):
# #   optimizer.zero_grad()
# #   output = lr_model(X_train).squeeze(dim = 1);
# #   loss = criterion(output, Y_train)
# #   loss.backward()
# #   optimizer.step()
# #   if(epoch % 20 == 0):
# #       print(epoch, loss.item())
# #       accuracy(X_train,Y_train,lr_model, 'train')
# #       accuracy(X_test,Y_test,lr_model, 'test')

# def poly(X_train, X, num):
#   return torch.pow((X @ (X_train.t()) + 1) ,num)

# def rbf(X_train, X, gamma):
#   Y = X_train.repeat(X.size(0), 1, 1)
#   return torch.exp(- gamma*((X[:,None]-Y)**2).sum(dim=2))

# def linear(X):
#   return X;

# class LogisticRegressionKernel(torch.nn.Module):
#     def __init__(self, X, labels, kernalType ='linear', gamma = 1.0, c = 1.0, num = 2 ):
#         super(LogisticRegressionKernel, self).__init__()
#         self.X = X;
#         self.labels = labels;
#         self.kernalType = kernalType
#         if(self.kernalType == 'linear'):
#           self.k = linear;
#           self.linearLayer = torch.nn.Linear(self.X.size(1), 1).float()

#         if(self.kernalType == 'rbf'):
#           self.k = rbf;
#           self.gamma = torch.nn.Parameter(torch.FloatTensor([gamma]),
#                                                   requires_grad=True)
#           self.linearLayer = torch.nn.Linear(self.X.size(0), 1).float()

#         if(self.kernalType == 'poly'):
#           self.k = poly;
#           self.num = num
#           self.linearLayer = torch.nn.Linear(self.X.size(0), 1).float()
        
#     def forward(self, x):
#       if(self.kernalType == 'linear'):
#         return torch.sigmoid(self.linearLayer(self.k(x)))
#       if(self.kernalType == 'rbf'):
#         return torch.sigmoid(self.linearLayer(self.k(self.X,x,self.gamma)))
#       if(self.kernalType == 'poly'):
#         return torch.sigmoid(self.linearLayer(self.k(self.X,x,self.num)))

  
# lrk = LogisticRegressionKernel(X_train, Y_train, 'poly', 1,1,3)
# lrkOptim = torch.optim.SGD(lrk.parameters(), lr=0.1)
# criterion = torch.nn.BCELoss()
# num_epochs = 1000

# for epoch in range(num_epochs):
#   lrkOptim.zero_grad()
#   output = lrk(X_train).squeeze(dim = 1)
#   loss = criterion(output, Y_train)
#   loss.backward()
#   lrkOptim.step()
#   if(epoch % 20 == 0):
#       print(epoch, loss.item())
#       accuracy(X_train,Y_train,lrk, 'train')
#       accuracy(X_test,Y_test,lrk, 'test')
      
# grid_x, grid_y = torch.meshgrid(torch.arange(X_train[:,0].min()*1.1, X_train[:,0].max()*1.1, step=0.1),
#                                 torch.arange(X_train[:,1].min()*1.1, X_train[:,1].max()*1.1, step=0.1))
# viz_x = torch.stack((grid_x, grid_y)).reshape(2, -1).transpose(1,0).float()
# print(viz_x.dtype)
# viz_y = lrk(viz_x).detach()
# viz_y = viz_y.transpose(1,0).reshape(grid_x.shape).numpy()
# fig, ax = plt.subplots(1,2, figsize=(15,7))

# cs0 = ax[0].contourf(grid_x.numpy(), grid_y.numpy(), viz_y)
# ax[0].contour(cs0, '--', levels=[0.5], colors='black', linewidths=2)
# ax[0].plot(np.nan, label='decision boundary', color='black')
# ax[0].scatter(X_train[np.where(Y_train==0),0], X_train[np.where(Y_train==0),1])
# ax[0].scatter(X_train[np.where(Y_train==1),0], X_train[np.where(Y_train==1),1])
# ax[0].legend()
# ax[0].set_title('Logistic Regression poly Kernel')

Y_train = torch.where(Y_train == 0, -1, 1)
Y_test = torch.where(Y_test == 0, -1, 1)
Y_val = torch.where(Y_val == 0, -1, 1)

def accuracySVM(X, Y, model, src):
  output = model(X).detach();
  predicted = output.data >= 0
  Y_real = Y >= 0
  print(src, accuracy_score(predicted, Y_real),X.size(0))

def svmloss(labels, outputs):    
  return torch.max(torch.zeros_like(labels), 1-labels*outputs).mean()

def poly(X_train, X, num):
  return torch.pow((X @ (X_train.t()) + 1) ,num)

class PolyModel(torch.nn.Module):
    def __init__(self, X, labels, kernelType, num = 1):
        super().__init__()
        self.X = X;
        self.labels = labels
        self.kernelType = kernelType

        self.k = None
        if(kernelType == 'poly'):
          self.k = poly
          self.num = num
          self.linearLayer = torch.nn.Linear(X.size(0), 1);
 
    def forward(self, X):
        val = self.k(self.X, X, self.num)
        return self.linearLayer(val);

def poly(X_train, X, num):
  return torch.pow((X @ (X_train.t()) + 1) ,num)


polyModel = PolyModel(X_train, Y_train, 'poly', 2)
polyOptim = torch.optim.SGD(polyModel.parameters(), lr=0.1)
accuracySVM(X_train, Y_train, polyModel, 'start')
num_epochs = 1000

for epoch in range(num_epochs):
  polyOptim.zero_grad()
  output = polyModel(X_train);
  loss = svmloss(output, Y_train)
  loss.backward()
  polyOptim.step()
  if(epoch % 20 == 0):
      print(epoch, loss.item())
      accuracySVM(X_train, Y_train, polyModel, 'train')
      accuracySVM(X_test, Y_test, polyModel, 'test')

grid_x, grid_y = torch.meshgrid(torch.arange(X_train[:,0].min()*1.1, X_train[:,0].max()*1.1, step=0.1),
                                torch.arange(X_train[:,1].min()*1.1, X_train[:,1].max()*1.1, step=0.1))
viz_x = torch.stack((grid_x, grid_y)).reshape(2, -1).transpose(1,0).float()
print(viz_x.dtype)
viz_y = polyModel(viz_x).detach()
viz_y = viz_y.transpose(1,0).reshape(grid_x.shape).numpy()
fig, ax = plt.subplots(1,2, figsize=(15,7))

cs0 = ax[0].contourf(grid_x.numpy(), grid_y.numpy(), viz_y)
ax[0].contour(cs0, '--', levels=[0.5], colors='black', linewidths=2)
ax[0].plot(np.nan, label='decision boundary', color='black')
ax[0].scatter(X_train[np.where(Y_train==-1),0], X_train[np.where(Y_train==-1),1])
ax[0].scatter(X_train[np.where(Y_train==1),0], X_train[np.where(Y_train==1),1])
ax[0].legend()
ax[0].set_title('SVM poly Kernel')
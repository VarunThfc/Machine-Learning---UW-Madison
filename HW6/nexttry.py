import torch
import numpy as np
from sklearn.datasets import load_breast_cancer
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.special import comb
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X_train = data['data']
Y_train = data['target']
print(X_train.shape)
X = torch.FloatTensor(X_train)
labels = torch.FloatTensor(Y_train)
labels = torch.where(labels == 0, -1, 1);


X_train, X_test, Y_train, Y_test = train_test_split(X, labels,test_size=(1-5/6), random_state=73)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,test_size=(1-4/5), random_state=73)

print(X_train.shape, X_test.shape, X_val.shape)
df = pd.DataFrame(dict(x1=X_train[:, 0], x2=X_train[:, 1], y=Y_train))
colors = {-1:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('y')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x1', y='x2', label=key, color = colors[key])
#plt.show()

df = pd.DataFrame(dict(x1=X_test[:, 0], x2=X_test[:, 1], y=Y_test))
colors = {-1:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('y')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x1', y='x2', label=key, color = colors[key])
n_input_dim = X_train.shape[1]

#plt.show()



def accuracySVM(X, Y, model, src):
  output = model(X).detach();
  predicted = output.data >= 0
  Y_real = Y >= 0
  print(src,accuracy_score(predicted, Y_real),X.size(0))

def svmloss(labels, outputs):    
  return torch.max(torch.zeros_like(labels), 1-labels*outputs).mean()


# def rbf(X_train, X, gamma):
#   Y = X_train.repeat(X.size(0), 1, 1)
#   return torch.exp(- gamma*((X[:,None]-Y)**2).sum(dim=2))


# class RBFModel(torch.nn.Module):
#     def __init__(self, X, labels, kernelType, featureDim1 = 1, featureDim2 = 1, gamma = 1):
#         super().__init__()
#         self.X = X;
#         self.labels = labels
#         self.kernelType = kernelType

#         self.k = None
#         if(kernelType == 'rbf'):
#           self.k = self.X.repeat(self.X.size(0), 1, 1)
#           self.gamma = torch.nn.Parameter(torch.FloatTensor([gamma]),
#                                                   requires_grad=True)
#           self.k = rbf
#           self.linearLayer = torch.nn.Linear(self.X.size(0), 1);
 
#     def forward(self, X):


#         val = self.k(self.X,X, self.gamma)
#         return self.linearLayer(val);

# rbfModel = RBFModel(X_train, Y_train, 'rbf', X_train.size(1), 1)
# rbfOptim = torch.optim.SGD(rbfModel.parameters(), lr=0.1)
# accuracySVM(X_train, Y_train, rbfModel, 'start')
# num_epochs = 1000

# for epoch in range(1000):
#   rbfOptim.zero_grad()
#   y_pred_rbf = rbfModel(X_train)
#   loss_rbf = svmloss(y_pred_rbf, Y_train)
#   loss_rbf.backward()
#   rbfOptim.step()
#   if(epoch % 20 == 0):
#     print(epoch, loss_rbf);
#     accuracySVM(X_train, Y_train, rbfModel, 'train')
#     accuracySVM(X_test, Y_test, rbfModel, 'test')
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
        
polyModel = PolyModel(X_train, Y_train, 'poly', 4)
polyOptim = torch.optim.SGD(polyModel.parameters(), lr=0.01)
accuracySVM(X_train, Y_train, polyModel, 'start')
num_epochs = 500

for epoch in range(num_epochs):
  polyOptim.zero_grad()
  y_pred_poly = polyModel(X_train)
  loss_poly = svmloss(y_pred_poly, Y_train)
  loss_poly.backward()
  polyOptim.step()
  if(epoch % 20 == 0):
    print(epoch, loss_poly);
    accuracySVM(X_train, Y_train, polyModel, 'train')
    accuracySVM(X_test, Y_test, polyModel, 'test')
import torch
import numpy as np
from sklearn.datasets import make_circles
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

torch.manual_seed(0)
np.random.seed(0)

#Prepare Datasets
circles_dataset = make_circles(1500, factor=0.1, noise=0.1)
X_train, labels_train = circles_dataset
X = torch.FloatTensor(X_train)
labels = torch.FloatTensor(labels_train)
labels = torch.where(labels == 0, -1, 1);

df = pd.DataFrame(dict(x1=X_train[:, 0], x2=X_train[:, 1], y=labels))

# colors = {-1:'red', 1:'blue'}
# fig, ax = plt.subplots()
# grouped = df.groupby('y')
# for key, group in grouped:
#     group.plot(ax=ax, kind='scatter', x='x1', y='x2', label=key, color = colors[key])
# plt.show()

rbfk = X.repeat(X.size(0), 1, 1)
print(rbfk.shape,"dsjn")


def rbf(X, k, gamma):
    print(torch.exp(-gamma*(((X[:,None])-k)**2).sum(dim=2)).shape,"asdnj")
    return torch.exp(-gamma*(((X[:,None])-k)**2).sum(dim=2))

def poly(X, k, const,pow):
    return torch.pow(X * k + const,pow)



polyKernel = torch.zeros([X.size(0), X.size(0)], dtype=np.float) 
for i in range(X.size(0)):
    for j in range(X.size(0)):
        polyKernel[i,j] = poly(X[i,0], X[j,1], 1, 2)
    
#define SVM loss
class Generic_SVM_Loss(nn.modules.Module):    
    def __init__(self):
        super(Generic_SVM_Loss,self).__init__()
    def forward(self, outputs, labels):
        labels = torch.unsqueeze(labels,1)
        return torch.max(torch.zeros_like(labels), 1-labels*outputs).mean()

##Define Model
class Model(torch.nn.Module):
    def __init__(self, X, labels, featureDim1, featureDim2, kernelType, gamma = 1):
        super().__init__()
        self.X = X;
        self.labels = labels
        self.linearLayer = torch.nn.Linear(featureDim1, featureDim2);
        self.kernelType = kernelType
        self.gamma = torch.nn.Parameter(torch.FloatTensor([gamma]),
                                             requires_grad=True)

    def forward(self, X):
        k = None
        if(self.kernelType == 'rbf'):
            k = rbf(X, rbfk, self.gamma)
        if(self.kernelType == 'poly'):
            print(polyKernel.size())
            k = (X.t * polyKernel).t
            print(k.shape)
        output = self.linearLayer(k);
        return output;

svm_loss_criteria = Generic_SVM_Loss()
rbfModel = Model(X, labels, X.size(0), 1, 'rbf', 1)
polyModel = Model(X, labels, X.size(0), 1, 'poly')
rbfOptim = torch.optim.SGD(rbfModel.parameters(), lr=0.1)
polyOptim = torch.optim.SGD(polyModel.parameters(), lr=0.1)

def accuracy(X,y, apply):
    output = apply(X)
    predicted = output.data >= 0
    real = y >= 0
    print(torch.sum(predicted.view(-1).long() == real),X.size(0))
accuracy(X, labels, polyModel)    

for i in range(500):
    
    # rbfOptim.zero_grad()
    # output = rbfModel(X);
    # loss = svm_loss_criteria(output, labels)
    # loss.backward()
    # rbfOptim.step()
    # if(i % 20 == 0):
    #     print(i, loss.item())
    #     accuracy(X,labels,rbfModel)
    
    polyOptim.zero_grad()
    output = polyModel(X);
    loss = svm_loss_criteria(output, labels)
    loss.backward()
    polyOptim.step()
    print(i);
    if(i % 20 == 0):
        print(i, loss.item())
        accuracy(X,labels,polyModel)

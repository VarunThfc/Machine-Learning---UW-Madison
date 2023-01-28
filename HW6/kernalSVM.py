
import math
import torch
import numpy as np;
torch.manual_seed(0)
np.random.seed(9)
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def polynomial_kernel(x, y, c=1, power=2):
    kernel_out = (np.dot(x,y) + c)**power
    return kernel_out





positive_distribution = np.random.multivariate_normal([2.5,0], np.identity(2), size=750)
ones = torch.ones([750,1]);
positive_distribution = np.append(positive_distribution,ones,1);
negative_distribution = np.random.multivariate_normal([-2.5,0], np.identity(2), size = 750)
minusOnes = torch.ones([750,1]);
minusOnes = minusOnes * -1
negative_distribution = np.append(negative_distribution,minusOnes,1);
total_dataset = np.append(positive_distribution, negative_distribution, 0)


# class customDataSet(Dataset):
#     def __init__(self):

#         positive_distribution = np.random.multivariate_normal([2.5,0], np.identity(2), size=750)
#         ones = torch.ones([750,1]);
#         positive_distribution = np.append(positive_distribution,ones,1);
#         negative_distribution = np.random.multivariate_normal([-2.5,0], np.identity(2), size = 750)
#         minusOnes = torch.ones([750,1]);
#         minusOnes = minusOnes * -1
#         negative_distribution = np.append(negative_distribution,minusOnes,1);
#         self.total_dataset = np.append(positive_distribution, negative_distribution, 0)
#         self.K=np.zeros((total_dataset.shape[0],total_dataset.shape[0]))
#         print(self.K.shape)
#         for i, row in enumerate(self.K):
#             for j, col in enumerate(self.K.T):
#                 self.K[i,j]=polynomial_kernel(total_dataset[i,0],total_dataset[j,0])

        
#     def __len__(self):
#         return len(self.K)
  
#     def __getitem__(self, idx):
#         return self.K[idx];

def plot(dataset):
    plt.scatter(dataset[:,0], dataset[:,1], c=dataset[:,2])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()     

alpha = np.zeros((total_dataset.shape[0],1), dtype=np.float64) #alpha
K=np.zeros((total_dataset.shape[0],total_dataset.shape[0])) #kernal
X = total_dataset
#plot(total_dataset)

n_samples, n_features = X.shape
epochs = 80;
for i, row in enumerate(K):
    for j, col in enumerate(K.T):
        K[i,j]=polynomial_kernel(total_dataset[i,0],total_dataset[j,0])
print("X", X.shape)
print("alpa", alpha.shape)
print("kernal", K.shape)
print("K[:, i]", K[:, 0].shape)
for epoch in range(epochs):
    for i in range(n_samples):
        u = np.sign(np.sum(K[i,:] * alpha * X[:,2]))
        if(X[i,2] * u <= 0):
            alpha[i] = alpha[i] + 1.0;
            
    acc_v=0
    for j in range(n_samples):
        #print(av.shape)
        uv = np.sign(np.sum(K[j,:] * alpha * X[:,2]))
        if uv* X[j,2]<=0:
            acc_v+=1
    print("accuracy" , acc_v)


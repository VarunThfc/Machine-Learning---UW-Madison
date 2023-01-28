import torch
import numpy as np;
torch.manual_seed(0)
np.random.seed(9)
from matplotlib import pyplot as plt

# torch.norm(2.5, 0 )
# def loss(y):
#     return max(0, 1 - y[inds[i]] * (torch.dot(w, torch.Tensor(X_train[inds[i]])) - b))**2
    
#prepare Datasets
positive_distribution = np.random.multivariate_normal([2.5,0], np.identity(2), size=750)
ones = torch.ones([750,1]);
positive_distribution = np.append(positive_distribution,ones,1);
negative_distribution = np.random.multivariate_normal([-2.5,0], np.identity(2), size = 750)
minusOnes = torch.ones([750,1]);
minusOnes = minusOnes * -1
negative_distribution = np.append(negative_distribution,minusOnes,1);
total_dataset = np.append(positive_distribution, negative_distribution, 0)
#split the datasets

#loss for linear SVM

w = torch.autograd.Variable(torch.rand([2,1]), requires_grad=True)
b = torch.autograd.Variable(torch.rand(1),   requires_grad=True)

def loss(index, total_dataset):
    return max(0, 1 - total_dataset[index][2] * (torch.matmul(w.T, torch.Tensor(total_dataset[index][:-1]).reshape(2,1)) - b))**2

def accuracy(X):
    correct = 0
    for i in range(len(X)):
        y_predicted = int(np.sign((torch.matmul(w.T, torch.Tensor(X[i][:-1]).reshape(2,1)) - b).detach().numpy()[0]))
        if y_predicted == X[i][2]: correct += 1
    return float(correct)/len(X)


def plot(dataset):
    plt.scatter(dataset[:,0], dataset[:,1], c=dataset[:,2])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
    
num_epochs = 10
plot(total_dataset)
print(accuracy(total_dataset))
for epoch in range(num_epochs):
    #Shuffle
    for i in range(len(positive_distribution)):
        l = loss(i, total_dataset)
        if(l != 0):
            l.backward()
            w.data -= 0.01 * w.grad.data 
            b.data -= 0.01 * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()
    print(l)
    print(accuracy(total_dataset))

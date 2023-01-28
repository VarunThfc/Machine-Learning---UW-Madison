import torch
import numpy as np;
torch.manual_seed(0)
np.random.seed(9)
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

positive_distribution = np.random.multivariate_normal([2.5,0], np.identity(2), size=750)
ones = torch.ones([750,1]);
positive_distribution = np.append(positive_distribution,ones,1);
negative_distribution = np.random.multivariate_normal([-2.5,0], np.identity(2), size = 750)
minusOnes = torch.ones([750,1]);
minusOnes = minusOnes * -1
negative_distribution = np.append(negative_distribution,minusOnes,1);
total_dataset = np.append(positive_distribution, negative_distribution, 0)

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
        

ds = customDataSet()

train_loader = torch.utils.data.DataLoader(dataset=ds, 
                                           batch_size=32, shuffle=True)
svm_model = nn.Linear(2,1).double()

class SVM_Loss(nn.modules.Module):    
    def __init__(self):
        super(SVM_Loss,self).__init__()
    def forward(self, outputs, labels):
         return torch.sum(torch.clamp(1 - outputs.t()*labels, min=0))/32
     
svm_loss_criteria = SVM_Loss()
svm_optimizer = torch.optim.SGD(svm_model.parameters(), lr=0.01)


correct = 0.
total = 0.
for input in train_loader:
    x = input[:,:-1];
    y = input[:,2];
    
    outputs = svm_model(x)    
    predicted = outputs.data >= 0
    real = y >= 0
    total += y.size(0) 
    correct += (predicted.view(-1).long() == real).sum()    
    
print(correct/total, correct, total)
        
def plot(dataset):
    plt.scatter(dataset[:,0], dataset[:,1], c=dataset[:,2])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
    
plot(total_dataset)


for epoch in range(100):
    avg_loss_epoch = 0
    batch_loss = 0
    total_batches = 0
    for i, input in enumerate(train_loader):
        x = input[:,:-1];
        y = input[:,2];
        print(x.shape)
        outputs = svm_model(x)
        loss_svm = svm_loss_criteria(outputs, y)   
        svm_optimizer.zero_grad()
        loss_svm.backward()
        svm_optimizer.step()   
        total_batches += 1     
        batch_loss += loss_svm.item()
        
    avg_loss_epoch = batch_loss/total_batches

correct = 0.
total = 0.
for input in train_loader:
    x = input[:,:-1];
    y = input[:,2];
    
    outputs = svm_model(x)    
    predicted = outputs.data >= 0
    real = y >= 0
    total += y.size(0) 
    correct += (predicted.view(-1).long() == real).sum()    
    
print(correct/total, correct, total)
        
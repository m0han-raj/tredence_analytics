from prunable_layer import PrunableLinear
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3*32*32 , 512)
        self.fc2 = PrunableLinear(512,256)
        self.fc3 = PrunableLinear(256,10)
    
    def forward(self,x):
        x=x.flatten(1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return self.fc3(x)



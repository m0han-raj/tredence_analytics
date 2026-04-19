import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch

tf = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
train_loader = DataLoader(
    torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=tf),
    batch_size=256, shuffle=True)
test_loader = DataLoader(
    torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=tf),
    batch_size=512)
    
device = "cuda" if torch.cuda.is_available() else "cpu"
 

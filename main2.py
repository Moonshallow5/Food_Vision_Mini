from torch import nn
import torch
from torchvision import transforms
import torchvision
import os
from pathlib import Path
import requests
import data_setup
import engine
import mlflow
import mlflow.pytorch
from torchmetrics import Accuracy

#mlflow.autolog()
if(torch.cuda.is_available):
    device='cuda'
else:
    device='cpu'
    
torch.manual_seed(42)
torch.cuda.manual_seed(42)


class TinyVGG(nn.Module):
    def __init__(self, num_classes):
        super(TinyVGG, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc_block = nn.Sequential(
            nn.Linear(in_features=128*32*32, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=num_classes)
        )
    
    def forward(self, x):
        x = self.conv_block_1(x)
    
        x = self.conv_block_2(x)
      
        x = self.conv_block_3(x)
       
        x = self.flatten(x)
        
        x = self.fc_block(x)
        
        return x

# Example usage
num_classes = 3  # For sushi, pizza, and steak
model = TinyVGG(num_classes=num_classes)
'''
class TinyVGG(nn.Module):
  def __init__(self, input_shape, hidden_units, output_shape):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=9,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=9,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=9,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=9,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*20*20,
                  out_features=output_shape))
        
  def forward(self, x):
    x = self.conv_block_1(x)
    print(f"Layer 1 shape: {x.shape}")
    x = self.conv_block_2(x)
    print(f"Layer 2 shape: {x.shape}")
    x = self.classifier(x)
    print(f"Layer 3 shape: {x.shape}")
    return x
'''
class VGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_block_1=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=(1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=(1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        )
        self.conv_block_2=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        )
        self.conv_block_3=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        )
        self.conv_block_4=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        )
        self.conv_block_5=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        )
        height,width=8,8
        self.classifier=torch.nn.Sequential(
            torch.nn.Linear(512*height*width,4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096,4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096,3)
        )
        
        self.avgpool=torch.nn.AdaptiveAvgPool2d((height,width))
    def forward(self,x):
        x = self.conv_block_1(x)
        print(f"Layer 1 shape: {x.shape}")
        x = self.conv_block_2(x)
        print(f"Layer 2 shape: {x.shape}")
        x = self.conv_block_3(x)
        print(f"Layer 3 shape: {x.shape}")
        x = self.conv_block_4(x)
        print(f"Layer 4 shape: {x.shape}")
        x = self.conv_block_5(x)
        print(f"Layer 5 shape: {x.shape}")
        x=self.avgpool(x)
        print(f"Layer 6 shape: {x.shape}")
        x=x.view(x.size(0),-1)
        print(x)
        logits=self.classifier(x)
        print(f"Layer 7 shape: {logits.shape}")
        return logits
        
        
        
    
         
         



                            
    
from torch import nn
import torch
from torchvision import transforms
import torchvision
import os
from pathlib import Path
import requests
import data_setup
import engine

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
import torch.nn.functional as F

        
class Model1Conv2D(nn.Module):
    def __init__(self, input_shape=(3, 260, 260), num_classes=3):
        super(Model1Conv2D, self).__init__()
        
        # Define the layers
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=10, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Calculate the size of the flattened features after all convolutions and pooling
        self._to_linear = self.calculate_output_shape(input_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_classes)
    
    def calculate_output_shape(self, input_shape):
        # Pass a dummy tensor through the model to calculate the shape of the feature map before the fully connected layers
        x = torch.rand(1, *input_shape)
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        return x.numel()  # Number of elements in the tensor after flattening

    def forward(self, x):
        # Apply conv and pooling layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        # Flatten
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No softmax needed as CrossEntropyLoss handles it
        return x

# Assuming input shape is (3, 100, 100) for your 100x100 RGB images

    
         
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model2BestOnPaper2(nn.Module):
    def __init__(self, input_shape=(3, 260, 260), num_classes=3):
        super(Model2BestOnPaper2, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)  # Adjusted size after pooling
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x):
        # Convolutional layers with relu activations and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten the output
        x = x.view(-1, 128 * 16 * 16)  # Adjusted to match output after pooling
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No softmax since CrossEntropyLoss is used during training
        
        return x




                            
    
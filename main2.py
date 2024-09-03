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

mlflow.autolog()
if(torch.cuda.is_available):
    device='cuda'
else:
    device='cpu'
    
torch.manual_seed(42)
torch.cuda.manual_seed(42)

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
        
        
        
    
         
         



simple_transform2=transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(128, 128)),
    # Flip the images randomly on the horizontal
    #transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor()
] )

simple_transform3=transforms.Compose([
    transforms.Resize((70,70)),
    transforms.CenterCrop((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    
    
    
])

def create_resNet50(out_features=3):
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights).to(device)
    
    dropout=0.3
    in_features=1408
    for param in model.parameters():
        param.requires_grad = True
    
    # Set the seeds
    # Update the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features, 
                  out_features=out_features)
    ).to(device) 

    # Set the model name
    model.name = "resnet50"
    #print(f"[INFO] Creating {model.name} feature extractor model...")
    return model


model_2=create_resNet50(3)
print("yoo")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # values per colour channel [red, green, blue]
                                 std=[0.229, 0.224, 0.225])

# Create a transform pipeline
simple_transform4 = transforms.Compose([
                                       transforms.Resize((224, 224)),
                                       transforms.ToTensor(), # get image values between 0 & 1
                                       normalize
])
print("yoooo")

from torchvision import datasets


data_20_percent_path = "data\pizza_steak_sushi_20_percent"
train_dir=data_20_percent_path+"\\train"
test_dir=data_20_percent_path+"\\test"

#train_data=datasets.ImageFolder(root=train_dir,transform=simple_transform2,target_transform=None)
#test_data=datasets.ImageFolder(root=test_dir,transform=simple_transform2)

#train_data2=datasets.ImageFolder(root=train_dir,transform=simple_transform3,target_transform=None)
#test_data2=datasets.ImageFolder(root=test_dir,transform=simple_transform3)

from torch.utils.data import DataLoader
'''
train_dataloader = DataLoader(dataset=train_data2, 
                              batch_size=32, # how many samples per batch?
                              num_workers=4, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?
test_dataloader = DataLoader(dataset=test_data2, 
                             batch_size=32, 
                             num_workers=4, 
                             shuffle=False) # don't usually need to shuffle testing data
'''


train_dataloader_20_percent, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                          test_dir=test_dir,
                                                                                          transform=simple_transform4,
                                                                                          batch_size=32)

print("yeh")






#model_0 = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                  #hidden_units=10, 
                  #output_shape=3).to(device)

#model_1=VGG16(num_classes=3).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_2.parameters(), 
                             lr=0.001)

from helper_functions import plot_loss_curves
if __name__=="__main__":
    
    
    model_1_results = engine.train(model=model_2,
                            train_dataloader=train_dataloader_20_percent,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            epochs=4,loss_fn=loss_fn,device='cpu')
    fig1=plot_loss_curves(model_1_results)
    

                            
    
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


from tqdm.auto import tqdm
mlflow.autolog()
if(torch.cuda.is_available):
    device='cuda'
else:
    device='cpu'


torch.manual_seed(42)






data_20_percent_path = "data\pizza_steak_sushi_20_percent"
train_dir_20_percent = data_20_percent_path+"\\train"
test_dir = data_20_percent_path+"\\test"


# Create a transform to normalize data distribution to be inline with ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # values per colour channel [red, green, blue]
                                 std=[0.229, 0.224, 0.225])

# Create a transform pipeline
simple_transform = transforms.Compose([
                                       transforms.Resize((224, 224)),
                                       transforms.ToTensor(), # get image values between 0 & 1
                                       normalize
])

BATCH_SIZE=32




train_dataloader_20_percent, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir_20_percent,
                                                                                          test_dir=test_dir,
                                                                                          transform=simple_transform,
                                                                                          batch_size=BATCH_SIZE)

def set_seeds(seed=42):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    
    
    
    
    
    
    


'''
import torchvision.models as models
effnetb2_weights=models.EfficientNet_B2_Weights.DEFAULT
effnetb2=models.efficientnet_b2(weights=effnetb2_weights)
'''
from torchinfo import summary
'''

#effnetv2_s = create_model(model_name="effnetv2_s")


summary(model=effnetb2,
        input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])
num_epochs = [5, 10]
print(summary)
'''
def create_effnetb2(out_features=len(class_names)):
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)
    dropout=0.3
    in_features=1408
    for param in model.features.parameters():
      param.requires_grad=False
    # Set the seeds
    set_seeds() 

    # Update the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features, 
                  out_features=out_features)
    ).to(device) 

    # Set the model name
    model.name = "effnetb2"
    #print(f"[INFO] Creating {model.name} feature extractor model...")
    return model
model=create_effnetb2(3)



loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
#metric_fn = Accuracy(task="multiclass", num_classes=3).to(device)
from helper_functions import *

from utils import *

if __name__ == "__main__":
    #mlflow.set_tracking_uri("http://127.0.0.1:5000")
    #mlflow.set_experiment("check-localhost-connection")
    
    
    epochs=2
    learning_rate=1e-3
    batch_size=32

        # Log training parameters.
    #mlflow.log_metrics(params)
    print("yo")
    #mlflow.log_param("alpha", 1.0)
    print("yo2")
    '''
    with open("model_summary.txt", "w",encoding="utf-8") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt")
    '''
    
    x=engine.train(model=model,train_dataloader=train_dataloader_20_percent,test_dataloader=test_dataloader,optimizer=optimizer,loss_fn=loss_fn,epochs=1,device=device)
    fig1=plot_loss_curves(x)
        #fig2=plot_accuracy_curves(x)
        #mlflow.log_figure()
        #mlflow.log_figure(fig2,"accuracy curves")
    '''
    mlflow.pytorch.log_model(model, "model")
    '''
        

        #plot_loss_curves(x)
    '''
        save_filepath = f"{model.name}_data_20_percent_without_aug 7_epochs.pth"
        save_model(model=model,
        target_dir="models",
        model_name=save_filepath)
    '''
        #mlflow.log_artifact("/tmp/corr_plot.png")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''
def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)
    print(image_path)
    
    return image_path
'''



    
   
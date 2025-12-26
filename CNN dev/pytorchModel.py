import os
from scipy import io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import optuna
import torch


def process_data():    
    #folder_path = r"C:\Users\1bout\OneDrive\SamaniLab\MESc Files\MLData\Results"
    folder_path = r"/home/deeplearningtower/Documents/MATLAB/JonahCode/MLTestData/FullResults/ClinicalResults"
    test_file_path = r"C:\Users\1bout\OneDrive\SamaniLab\MESc Files\403170837836d865566676a2b5ddade1.mat"

    num_files = 1936
    image_shape = (2500,256,2)
    images = np.empty((num_files,*image_shape),dtype=np.float32)
    labels = np.empty((num_files,),dtype=np.float32)

    i = 0
    # numbenign = 0
    # nummalignant = 0
    for file in os.listdir(folder_path):
        if os.fsdecode(file).endswith('.mat'):

            # Load data as a test
            try:
                mat_data = io.loadmat(folder_path + '//' + file)
            except FileNotFoundError as e:
                print(e)
            else:
                frame_one = mat_data['result'][0,0]['Frame1']
                frame_two = mat_data['result'][0,0]['Frame2']
                # label = mat_data['result'][0,0]['label'][0] #Grab label
                tumor_mask = mat_data['result'][0,0]['image_information'][0,0]['tumor_mask'].astype(bool) #tumor_mask
                # tumor_mask = tumor_mask.astype(bool) #Convert to bool matrix
                youngs_modulus_matrix = mat_data['result'][0,0]['image_information'][0,0]['YM_image'] #YM matrix

                tumor_mean_YM = np.mean(youngs_modulus_matrix[tumor_mask])    
                background_mean_YM = np.mean(youngs_modulus_matrix[~tumor_mask])

                # stiffness_ratio = tumor_mean_YM/background_mean_YM

                # appending = np.stack((frame_one, frame_two), axis=-1)
                images[i] = np.stack((frame_one, frame_two), axis=-1)

                labels[i] = np.round(tumor_mean_YM/background_mean_YM,1)

                       
                i = i + 1
                print('Done File ',i)
    np.save('images.npy', images)
    np.save('labels.npy', labels)
    print("Done Processing")

def load_data(images,labels,batch_size = 16,train_split=0.8):
    """
    Prepare the data for the model, once it's been processed
    """
    images = np.transpose(images,(0,3,1,2)) 
    images_tensor = torch.tensor(images,dtype=torch.float32)
    labels_tensor = torch.tensor(labels,dtype=torch.float32).unsqueeze(1) #Make lables shape (N,1)

    dataset = TensorDataset(images_tensor,labels_tensor)
    train_size = int(train_split*len(dataset))
    val_size = len(dataset) - train_size

    train_dataset,val_dataset = random_split(dataset,[train_size,val_size])
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

    print(f"Train Samples: {train_size}, Validation samples: {val_size}")
    return train_loader,val_loader
    
    # labels_tensor = torch.tensor(labels,dtype=torch.int)
    # images = np.ascontiguousarray(images)
    # images = np.transpose(images,(0,3,1,2))
    # images_tensor = torch.from_numpy(images).float()
    # # images_tensor = torch.tensor(images)#,dtype=torch.float32)
    # # labels_tensor = torch.tensor(labels,dtype=torch.long)
    # print('Made tensors')

    # dataset = TensorDataset(images_tensor,labels_tensor) #Dataset for model

    # data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True) #Dataloader for batching
    # return data_loader

class CNN(nn.Module):
    def __init__(self,num_filters=[32,64,128],kernel_size=3,stride=2,padding=1):
        super(CNN,self).__init__()
        self.num_filters = num_filters
        self.conv_layers = self._create_conv_layers(num_filters)

        self.fc_input_size = self._compute_fc_input_size()

        self.fc1 = nn.Linear(self.fc_input_size,512)
        self.fc2 = nn.Linear(512,1)

    def _create_conv_layers(self,num_filters):
        layers = []
        in_channels = 2

    #Multiple conv layers
        for out_channels in num_filters:
            layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2,2))
            in_channels = out_channels
        
        self.conv = nn.Sequential(*layers)
        fc_size = in_channels * (2500 // (2**len(num_filters))) * (256 // (2**len(num_filters)))
        self.fc1 = nn.Linear(fc_size,512)
        self.fc2 = nn.Linear(512,1)
        return nn.Sequential(*layers)
    
    def _compute_fc_input_size(self):
        x = torch.randn(1,2,2500,256)#Random tensor
        x = self.conv_layers(x)
        return x.numel()

    def forward(self,x):
        x = self.conv_layers(x)
        x = x.view(x.size(0),-1) #Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x) #Linear output
        return x

    # def __init__(self,num_classes = 4):
    #     super(CNN,self).__init__()

    #     #Conv layers
    #     self.conv1 = nn.Conv2d(in_channels=2,out_channels=32,kernel_size=3,stride=2,padding=1)
    #     self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1)
    #     self.conv3 = nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)

    #     #Max pooling layer
    #     self.pool = nn.MaxPool2d(2,2)

    #     fc_size = 128 * (2500//64) *(256//64)

    #     #Fully connected layer
    #     self.fc1 = nn.Linear(fc_size,512)
    #     self.fc2 = nn.Linear(512,num_classes)

    # def forward(self,x):
    #     x = self.pool(F.relu(self.conv1(x))) #Conv + ReLU + Pooling
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = self.pool(F.relu(self.conv3(x)))
    #     x = x.view(x.size(0),-1)
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x
    
# def initialize_model(num_classes = 4,learning_rate = 0.001):
#     model = CNN(num_classes)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(),lr=learning_rate)

#     return model,criterion,optimizer,device

def train_model(model,train_loader,val_loader,criterion, optimizer, device, num_epochs = 10):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = validate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        torch.cuda.empty_cache()
def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Hyperparameter search space
    num_layers = trial.suggest_int("num_layers",3,5)
    num_filters = [trial.suggest_int(f"num_filters_{i}",32,256,step=32) for i in range(num_layers)]
    learning_rate = trial.suggest_loguniform("learning_rate",1e-5,1e-2)
    batch_size = trial.suggest_categorical("batch_size",[16,32,64])

    #Load data
    images = np.load('images.npy')
    labels = np.load('labels.npy')

    train_loader, val_loader = load_data(images,labels,batch_size)

    model = CNN(num_filters).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    train_model(model,train_loader,val_loader,criterion,optimizer,device,num_epochs=10)
    val_loss = validate_model(model,val_loader,criterion,device)

    return val_loss
# def train_model(model,data_loader,criterion,optimizer,device,num_epochs=10):
    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0.0
    #     correct_preds  = 0
    #     total_preds = 0

    #     for inputs, targets in data_loader:
    #         targets = targets.type(torch.LongTensor)
    #         inputs,targets = inputs.to(device),targets.to(device)

    #         optimizer.zero_grad()

    #         outputs = model(inputs)

    #         loss = criterion(outputs,targets)

    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()

    #         _,predicted = torch.max(outputs,1)
    #         correct_preds += (predicted==targets).sum().item()
    #         total_preds += targets.size(0)

    #         epoch_loss = running_loss / len(data_loader)
    #         epoch_accuracy = correct_preds / total_preds * 100

    #         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

def main():

    study = optuna.create_study()
    study.optimize(objective,n_trials=20) #Run optimization for 20 trials

    print(f"Best Hyperparameters: {study.best_params}")

    # Load data with best hyperparameters
    best_params = study.best_params
    num_filters = [best_params[f"num_filters_{i}"] for i in range(best_params["num_layers"])]
    batch_size = best_params["batch_size"]
    learning_rate = best_params["learning_rate"]

    images = np.load('images.npy')
    labels = np.load('labels.npy')

    train_loader,val_loader = load_data(images,labels,batch_size)

    #Init model with best hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_filters).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model,train_loader,val_loader,criterion,optimizer,device,num_epochs=15)

    validate_model(model,val_loader,criterion,device)

main()
    # images,labels = process_data()
    # print('Unique labels: ',len(set(labels)))
    # data_loader = load_data(images,labels,batch_size=32) #Preprocess data
    # print('test')
    # model,criterion,optimizer,device = initialize_model(num_classes=4)

    # num_epochs = 10
    # train_model(model,data_loader,criterion,optimizer,device,num_epochs)
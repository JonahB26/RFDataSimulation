"""
This code represents the modified version of my (Jonah Boutin's) neural network, by Zhenbang Wang,
with the purpose of using 1D CNN rather than 2D
"""

import os
from scipy import io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.transforms as transforms
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import torchvision.models as models  

import os
print(os.cpu_count())  # Total CPU cores
torch.cuda.empty_cache() #Clear before starting

def load_data(images, labels, batch_size=16, train_split=0.8):
    #images = np.transpose(images,(0,3,1,2)) 
    print(images.shape)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0,translate=(0.1,0.1)),
        transforms.GaussianBlur(kernel_size=3)
    ])

    augmented_images = torch.tensor(images,dtype=torch.float32)
    print('loading data')
    #for i in range(len(augmented_images)):
       # augmented_images[i] = transform(augmented_images[i])
    # images_tensor = torch.tensor(images,dtype=torch.float32)
    labels_tensor = torch.tensor(labels,dtype=torch.float32).unsqueeze(1)
    # images_tensor = torch.tensor(augmented_images,dtype=torch.float32)

    dataset = TensorDataset(augmented_images,labels_tensor)
    train_size = int(train_split*len(dataset))
    val_size = len(dataset) - train_size

    train_dataset,val_dataset = random_split(dataset,[train_size,val_size])
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=10,pin_memory=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=10,pin_memory=True)
    print(f"Train Samples: {train_size}, Validation samples: {val_size}")
    
    return train_loader,val_loader

class CNN_Hilbert_ResNet(nn.Module):
    def __init__(self, resnet_type="resnet34", dropout_rate=0.5):
        super(CNN_Hilbert_ResNet, self).__init__()

        self.conv1d_hilbert1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1d_hilbert1 = nn.BatchNorm1d(64)
        
        self.conv1d_hilbert2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1d_hilbert2 = nn.BatchNorm1d(64)
        
        self.pool1d_hilbert = nn.MaxPool1d(kernel_size=2, stride=2)  

        match(resnet_type):
            case "resnet34":
                self.resnet = models.resnet34(weights=None)
            case "resnet18":
                self.resnet = models.resnet18(weights=None)
            case "resnet50":
                self.resnet = models.resnet50(weights=None)
        
        self.resnet.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  
        self.resnet.maxpool = nn.Identity()  

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1) 
		
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (batch, 2, 2500, 256) → (batch, 2, 2500, 256f)

        x = x.reshape(x.shape[0] * x.shape[3], x.shape[1], x.shape[2])  # (batch*256, 2, 2500)
        
        x = F.relu(self.bn1d_hilbert1(self.conv1d_hilbert1(x)))  
        x = F.relu(self.bn1d_hilbert2(self.conv1d_hilbert2(x)))  
        x = self.pool1d_hilbert(x)  # (batch*256, 64, new_depth)

        new_depth = x.shape[2]  
        x = x.view(-1, 64, new_depth, 256)  

        x = self.resnet(x)  

        return x 

class EarlyStopping:
    def __init__(self,patience=5,min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def check(self,val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print("Early stopping triggered.")
            return True
        return False

class TrainingHistory:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.predictions = []
        self.actual_values = []
        
    def add_epoch(self, epoch, train_loss, val_loss):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
    def add_predictions(self, preds, actuals):
        self.predictions.extend(preds.cpu().numpy())
        self.actual_values.extend(actuals.cpu().numpy())

def train_with_gradient_accumulation(model, train_loader, val_loader, criterion, optimizer,scheduler, 
                                   device, num_epochs=10, accumulation_steps=8):
    history = TrainingHistory()
    model.to(device)
    early_stopping = EarlyStopping(patience=7,min_delta=0.001)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for i, (inputs, targets) in enumerate(train_loader):
            # Move to GPU in smaller batches
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass with gradient accumulation
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            running_loss += loss.item() * accumulation_steps
            
            # Clear unnecessary tensors
            del outputs, loss
        
        val_loss, val_preds, val_targets = validate_model(model, val_loader, criterion, device)
        train_loss = running_loss/len(train_loader)
        history.add_epoch(epoch + 1, train_loss, val_loss)
        history.add_predictions(val_preds, val_targets)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")
        scheduler.step()

        if early_stopping.check(val_loss):
            break
        
        # Clear cache after each epoch
        torch.cuda.empty_cache()
    
    return history

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            all_predictions.append(outputs)
            all_targets.append(targets)
    
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    
    return val_loss / len(val_loader), predictions, targets
def objective(trial, images, labels):
    resnet_type = trial.suggest_categorical("resnet_type", ["resnet18", "resnet34", "resnet50"])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4,8, 16])
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    train_loader,val_loader = load_data(images=images,labels=labels,batch_size=batch_size)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNN_Hilbert_ResNet(resnet_type=resnet_type, dropout_rate=dropout_rate).to(device)
        #criterion = nn.SmoothL1Loss()
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        history = train_with_gradient_accumulation(
            model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=15, accumulation_steps=4
        )

        del model
        torch.cuda.empty_cache()
        
        return min(history.val_losses)
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            trial.report(float("inf"),step=0) #Log the shitty trial
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned() #Let optuna decide if it wants to prune
            return float("inf") #Otherwise we return shitty loss
        else:
            raise e
def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.epochs, history.train_losses, label='Training Loss')
    plt.plot(history.epochs, history.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    # Save the figure instead of displaying it
    plt.savefig("Training and Validation Loss Over Time.png", dpi=300, bbox_inches="tight")  
    plt.ylim(0,1)
    plt.close()  # Close the figure to free memory

def plot_prediction_scatter(history):
    plt.figure(figsize=(10, 10))
    r2 = r2_score(history.actual_values, history.predictions)
    
    plt.scatter(history.actual_values, history.predictions, alpha=0.5)
    plt.plot([min(history.actual_values), max(history.actual_values)], 
             [min(history.actual_values), max(history.actual_values)], 
             'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual Values (R² = {r2:.3f})')
    plt.legend()
    plt.grid(True)
    # Save the figure instead of displaying it
    plt.savefig(f'Predicted vs Actual Values (R² = {r2:.3f}).png', dpi=300, bbox_inches="tight")  
    plt.close()  # Close the figure to free memory

def plot_error_distribution(history):
    errors = np.array(history.predictions) - np.array(history.actual_values)
    plt.figure(figsize=(10, 5))
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True)
    # Save the figure instead of displaying it
    plt.savefig("Distribution of Prediction Errors.png", dpi=300, bbox_inches="tight")  
    plt.close()  # Close the figure to free memory

def plot_optuna_results(study):
    plt.figure(figsize=(10, 5))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title('Optuna Optimization History')
    # Save the figure instead of displaying it
    plt.savefig("Optuna Optimization History.png", dpi=300, bbox_inches="tight")  
    plt.close()  # Close the figure to free memory
    
    plt.figure(figsize=(10, 5))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title('Hyperparameter Importance')
    # Save the figure instead of displaying it
    plt.savefig("Hyperparameter Importance.png", dpi=300, bbox_inches="tight")  
    plt.close()  # Close the figure to free memory
def main():
    if os.path.exists('Data/images_final.npy') and os.path.exists('Data/labels_final.npy'):
        print("Loading saved final data...")
        images = np.load('Data/images_final.npy')
        labels = np.load('Data/labels_final.npy')
    else:
        print("Data not found. Please generate it first.")
        return

   # batch_size = 16
    #train_loader, val_loader = load_data(images, labels, batch_size)

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())

    try:
        study.optimize(lambda trial: objective(trial, images, labels), n_trials=10, catch=(RuntimeError,))
    except Exception as e:
        print(f"Optimization stopped due to error: {e}")

    print("\nBest trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    torch.cuda.empty_cache()

    best_params = study.best_params
    resnet_type = best_params["resnet_type"]
    batch_size = best_params["batch_size"]
    learning_rate = best_params["learning_rate"]
    dropout_rate = best_params["dropout_rate"]

    train_loader, val_loader = load_data(images, labels, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_Hilbert_ResNet(resnet_type=resnet_type, dropout_rate=dropout_rate).to(device)
    #criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    history = train_with_gradient_accumulation(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=15, accumulation_steps=4
    )

    plot_training_history(history)
    plot_prediction_scatter(history)
    plot_error_distribution(history)
    
    torch.save(model.state_dict(), 'best_model_fake_data.pth')

if __name__ == "__main__":
    main()

"""
This code represents the modified version of my (Jonah Boutin's) neural network, by Zhenbang Wang,
with the purpose of using 1D CNN rather than 2D
"""

import os
import pickle
from scipy import io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import torchvision.models as models  

#import ProcessData

torch.cuda.empty_cache() #Clear before starting

def load_data(images, labels, batch_size=16, train_split=0.8):
    #images = np.transpose(images,(0,3,1,2)) 
    print(images.shape)
    print('before')
    augmented_images = torch.tensor(images,dtype=torch.float32)
    print('after')
    print('loading data')
 
    labels_tensor = torch.tensor(labels,dtype=torch.float32).unsqueeze(1)

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

        # 1D Hilbert Transform Layers
        self.conv1d_hilbert1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1d_hilbert1 = nn.BatchNorm1d(64)
       
        self.conv1d_hilbert2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, dilation=2)
        self.bn1d_hilbert2 = nn.BatchNorm1d(64)
       
        # Load Pretrained ResNet
        match(resnet_type):
            case "resnet34":
                self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            case "resnet18":
                self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            case "resnet50":
                self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Modify First Conv Layer in ResNet to Accept 64-Channel Input
        self.resnet.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  
        self.resnet.maxpool = nn.Identity()  # Remove pooling to retain spatial information

        # Fully Connected Layers with Swish Activation
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 128),
            nn.SiLU(),  # Swish activation
            nn.Dropout(dropout_rate),
            #nn.Dropout(0.2),

            nn.Linear(128, 1)  # Single output neuron for regression
        )

    def forward(self, x):
        # Input shape: (batch, 2, 2500, 256)
        x = x.permute(0, 3, 1, 2)  # Reshape to (batch, 2, 2500, 256)

        # Convert to 1D format for Hilbert processing
        x = x.reshape(x.shape[0] * x.shape[3], x.shape[1], x.shape[2])  # (batch*256, 2, 2500)

        # Apply Hilbert Transform Convolutions
        x = F.silu(self.bn1d_hilbert1(self.conv1d_hilbert1(x)))  # Swish activation
        x = F.silu(self.bn1d_hilbert2(self.conv1d_hilbert2(x)))  # Swish activation

        # Reshape for ResNet input
        new_depth = x.shape[2]
        x = x.view(-1, 64, new_depth, 256)  # (batch, 64, new_depth, 256)

        # Pass through ResNet
        x = self.resnet(x)

        return F.relu(x)#Ensure output is at least 0

# class EarlyStopping:
#     def __init__(self,patience=5,min_delta=0.001):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.best_loss = float("inf")
#         self.counter = 0

#     def check(self,val_loss):
#         if val_loss < self.best_loss - self.min_delta:
#             self.best_loss = val_loss
#             self.counter = 0
#         else:
#             self.counter += 1

#         if self.counter >= self.patience:
#             print("Early stopping triggered.")
#             return True
#         return False

class EarlyStopping:
    def __init__(self, patience=25, min_delta=0.001, save_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.save_path = save_path  # Save best model path

    def check(self, val_loss, model,is_optimizing):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if is_optimizing is False:
                torch.save(model.state_dict(), self.save_path)  # Save best model
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print("Early stopping triggered. Restoring best model...")
            model.load_state_dict(torch.load(self.save_path))  # Restore best model
            return True  # Stop training

        return False  # Continue training

# class TrainingHistory:
#     def __init__(self):
#         self.train_losses = []
#         self.val_losses = []
#         self.epochs = []
#         self.predictions = []
#         self.actual_values = []
        
#     def add_epoch(self, epoch, train_loss, val_loss):
#         self.epochs.append(epoch)
#         self.train_losses.append(train_loss)
#         self.val_losses.append(val_loss)
        
#     def add_predictions(self, preds, actuals):
#         self.predictions.extend(preds.cpu().numpy())
#         self.actual_values.extend(actuals.cpu().numpy())

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

    def save_to_file(self, filename="best_training_history.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_file(filename="best_training_history.pkl"):
        with open(filename, "rb") as f:
            return pickle.load(f)

def train_with_gradient_accumulation(model, train_loader, val_loader, criterion, optimizer,scheduler, 
                                   device, num_epochs=10, accumulation_steps=8,is_optimizing=False):
    history = TrainingHistory()
    model.to(device)
    # early_stopping = EarlyStopping(patience=25,min_delta=0.001)
    early_stopping = EarlyStopping(patience=30, min_delta=0.001, save_path="best_model.pth")
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
        #scheduler.step()
        scheduler.step()

        # **Check Early Stopping & Restore Best Model If Needed**
        if early_stopping.check(val_loss, model,is_optimizing):
            break  # Stop training if early stopping triggers

        # if early_stopping.check(val_loss):
        #     break
        
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

def objective(trial, images, labels, resnet_type,learning_rate,batch_size):
    """resnet_type = trial.suggest_categorical("resnet_type", ["resnet18", "resnet34", "resnet50"])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4,8, 16])"""
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.1, 0.15, 0.2, 0.25, 0.3])
    gamma = trial.suggest_categorical("gamma",[0.95, 0.96, 0.97, 0.98, 0.99])
    train_loader,val_loader = load_data(images=images,labels=labels,batch_size=batch_size)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNN_Hilbert_ResNet(resnet_type=resnet_type, dropout_rate=dropout_rate).to(device)
        #criterion = nn.SmoothL1Loss()
        criterion = nn.SmoothL1Loss(beta=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        history = train_with_gradient_accumulation(
            model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=25, accumulation_steps=4,is_optimizing=True
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
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('Loss',fontsize=20)
    plt.legend(fontsize=20,loc='best')
    plt.legend(frameon=True,fancybox=True,framealpha=1,edgecolor='black',fontsize=20)
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
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    print(f'Min training loss: {np.min(np.array(history.train_losses)):.4f}')
    print(f'Min val loss: {np.min(np.array(history.val_losses)):.4f}')
    print(f'Histogram Center (Mean): {mean_error:.4f}')
    print(f'Spread (Standard Deviation): {std_error:.4f}')
    plt.figure(figsize=(10, 5))
    sns.histplot(errors, kde=True,legend=False,alpha=0.5)
    plt.xlim(-4.0,4.0)
    plt.xlabel('Residual Error',fontsize=20)
    plt.ylabel('Count',fontsize=20)
    plt.grid(True)
    
    # Save the figure instead of displaying it
    plt.savefig("Distribution of Residual Error.png", dpi=300, bbox_inches="tight")  
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
    """torch.cuda.empty_cache()
    if os.path.exists('Data/images_final.npy') and os.path.exists('Data/labels_final.npy'):
        print("Loading saved final data...")
        images = np.load('Data/images_final.npy')
        labels = np.load('Data/labels_final.npy')
    else:
        print("Data not found. Please generate it first.")
        #ProcessData.process_data()

      # batch_size = 16
    #train_loader, val_loader = load_data(images, labels, batch_size)

    resnet_type = 'resnet18'#best_params["resnet_type"]
    batch_size = 4#best_params["batch_size"]
    learning_rate = 0.0004129511601565426#best_params["learning_rate"""

    """if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())

    try:
        study.optimize(lambda trial: objective(trial, images, labels,resnet_type,learning_rate,batch_size), n_trials=12,catch=(RuntimeError,))
    except Exception as e:
        print(f"Optimization stopped due to error: {e}")

    print("\nBest trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    torch.cuda.empty_cache()

    best_params = study.best_params"""

    """images = np.load('Data/images_final.npy')
    labels = np.load('Data/labels_final.npy')
    resnet_type = 'resnet18'#best_params["resnet_type"]
    batch_size = 4#best_params["batch_size"]
    learning_rate = 0.0004129511601565426#best_params["learning_rate"]"""

    """dropout_rate = 0.2#best_params["dropout_rate"]
    gamma = 0.98#best_params["gamma"]

    train_loader, val_loader = load_data(images, labels, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_Hilbert_ResNet(resnet_type=resnet_type, dropout_rate=dropout_rate).to(device)
    #criterion = nn.SmoothL1Loss()
    criterion = nn.SmoothL1Loss(beta=0.8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=5)

    # history = train_with_gradient_accumulation(
    #     model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=400, accumulation_steps=4,is_optimizing=False
    # )

    history = train_with_gradient_accumulation(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=500, accumulation_steps=5,is_optimizing=False
    )

    history.save_to_file("best_training_history.pkl")

    plot_training_history(history)
    plot_prediction_scatter(history)
    plot_error_distribution(history)
    
    torch.save(model.state_dict(), 'modelGoodData.pth')"""

    # Mess with final model for poster
    training = TrainingHistory()
    history = training.load_from_file("BestData/D15ST5/best_training_history.pkl")
    plot_training_history(history)
    plot_error_distribution(history)


if __name__ == "__main__":
    main()

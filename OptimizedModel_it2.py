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
from ProcessData import process_data

def load_data(images, labels, batch_size=16, train_split=0.8):
    images = np.transpose(images,(0,3,1,2)) 

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0,translate=(0.1,0.1)),
        transforms.GaussianBlur(kernel_size=3)
    ])

    images_tensor = torch.tensor(images,dtype=torch.float32)
    labels_tensor = torch.tensor(labels,dtype=torch.float32).unsqueeze(1)

    class ModDataset(torch.utils.data.Dataset):
        def __init__(self,images,labels,transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.images)
        
        def __getitem__(self,idx):
            image = self.images[idx]
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image,label
    
    dataset = ModDataset(images_tensor,labels_tensor,transform=transform)
    train_size = int(train_split*len(dataset))
    val_size = len(dataset) - train_size

    train_dataset,val_dataset = random_split(dataset,[train_size,val_size])
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=True)

    print(f"Train Samples: {train_size}, Validation samples: {val_size}")
    return train_loader,val_loader

class OptimizedCNN(nn.Module):
    def __init__(self, num_filters,dropout_rate=0.5,in_layer_dropout=0.25):
        super(OptimizedCNN, self).__init__()

        layers = []
        in_channels = 2  # Input has 2 channels (Frame1 & Frame2)

        for i in  range(len(num_filters)):
            layers.append(nn.Conv2d(in_channels, num_filters[i], kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(num_filters[i]))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            #layers.append(nn.SiLU()),  # Swish activation function
            layers.append(nn.AvgPool2d(2, 2))
            if i % 2 == 0:
                # layers.append(nn.MaxPool2d(2, 2))
                layers.append(nn.Dropout(in_layer_dropout))
            in_channels = num_filters[i]  # Update input channels for next layer

        self.features = nn.Sequential(*layers)
        self.feature_size = self._get_feature_size()

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate-0.1),
            #nn.SiLU(),
            nn.Linear(64, 1)
        )

    def _get_feature_size(self):
        x = torch.zeros(1, 2, 2500, 256)  # Use zeros instead of random noise
        x = self.features(x)
        return x.numel()  # Returns total flattened size


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

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

def objective(trial):
    # More conservative hyperparameter ranges
    num_layers = trial.suggest_int("num_layers", 3, 4)  # Reduced max layers
    
    # Progressively increasing filter sizes with smaller ranges
    num_filters = []
    prev_filters = 16  # Start small
    for i in range(num_layers):
        # Each layer can only increase filters by a maximum factor of 2
        max_filters = min(prev_filters * 2, 128)  # Cap at 128 filters
        num_filters.append(trial.suggest_int(f"num_filters_{i}", prev_filters, max_filters))
        prev_filters = num_filters[-1]
    
    # Ensure that the number of filters corresponds to the num_layers
    num_filters = num_filters[:num_layers]  # This ensures the list is the correct length
    
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2,log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16])  # Smaller batch sizes
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)

    try:
        # Load and prepare data
        images = np.load('images_prePCA.npy')
        labels = np.load('labels_prePCA.npy')
        train_loader, val_loader = load_data(images, labels, batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = OptimizedCNN(num_filters,dropout_rate).to(device)
        criterion = nn.SmoothL1Loss()
        #optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9,weight_decay=5e-4)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # Clear GPU memory before training
        torch.cuda.empty_cache()

        history = train_with_gradient_accumulation(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer,
            scheduler, 
            device, 
            num_epochs=15,  # Reduced epochs for trials
            accumulation_steps=8  # Increased accumulation steps
        )

        # Clear memory after training
        del model
        torch.cuda.empty_cache()
        
        return min(history.val_losses)
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
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
    # Set memory allocation configuration
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of available GPU memory

    # First process the data if needed
    """ if not os.path.exists('images_prePCA.npy') or not os.path.exists('labels_prePCA.npy'):
        process_data() """

    # Create and run Optuna study with pruning
    study = optuna.create_study(direction="minimize", 
                               pruner=optuna.pruners.MedianPruner())
    
    try:
        study.optimize(objective, n_trials=25, 
                      catch=(RuntimeError,))  # Catch memory errors
    except Exception as e:
        print(f"Optimization stopped due to error: {e}")

    print("\nBest trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Clear memory before final training
    torch.cuda.empty_cache()
    
    # Train final model with best parameters
    best_params = study.best_params
    num_filters = [best_params[f"num_filters_{i}"] for i in range(best_params["num_layers"])]
    batch_size = best_params["batch_size"]
    learning_rate = best_params["learning_rate"]
    dropout_rate = best_params["dropout_rate"]
    # in_layer_dropout = best_params["in_layer_dropout"]
    
    # Load data
    images = np.load('images_prePCA.npy')
    labels = np.load('labels_prePCA.npy')
    train_loader, val_loader = load_data(images, labels, batch_size)
    
    # Initialize and train final model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OptimizedCNN(num_filters,dropout_rate).to(device)
    criterion = nn.SmoothL1Loss()
   # optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9,weight_decay=5e-4)
   # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Train final model with increased accumulation steps
    history = train_with_gradient_accumulation(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler,
        device, 
        num_epochs=100,
        accumulation_steps=8
    )
    
    # Plot results
    plot_training_history(history)
    plot_prediction_scatter(history)
    plot_error_distribution(history)
    
    # Save the model
    torch.save(model.state_dict(), 'best_model.pth')

if __name__ == "__main__":
    main()

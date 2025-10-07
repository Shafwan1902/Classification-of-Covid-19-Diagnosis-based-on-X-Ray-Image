import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from utils import split_dataset
from sklearn.metrics import accuracy_score

# -------------------------------
# STEP 1: Split dataset (only once)
# -------------------------------
split_dataset("dataset", "split_dataset")

# -------------------------------
# STEP 2: Set parameters
# -------------------------------
data_dir = "split_dataset"
batch_size = 16
num_classes = 3
num_epochs = 30
learning_rate = 0.0005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# STEP 3: Data transforms with augmentation
# -------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

datasets_dict = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform),
    'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_test_transform),
    'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transform)
}

dataloaders = {
    x: DataLoader(datasets_dict[x], batch_size=batch_size, shuffle=True)
    for x in ['train', 'val', 'test']
}

# -------------------------------
# STEP 4: Load ResNet18
# -------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# -------------------------------
# STEP 5: Train the model
# -------------------------------
def train_model():
    print("Training started...")
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss, correct = 0.0, 0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels).item()
        epoch_loss = running_loss / len(datasets_dict['train'])
        epoch_acc = correct / len(datasets_dict['train'])
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_correct += torch.sum(preds == labels).item()
        val_epoch_loss = val_loss / len(datasets_dict['val'])
        val_epoch_acc = val_correct / len(datasets_dict['val'])
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

        print(f"Epoch {epoch+1}/{num_epochs} => "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

        scheduler.step()

    print("Training complete.")
    torch.save(model.state_dict(), "covid_resnet18_model.pth")
    print("Model saved as covid_resnet18_model.pth")
    plot_training(train_losses, val_losses, train_accuracies, val_accuracies)

# -------------------------------
# STEP 6: Plot training curves
# -------------------------------
def plot_training(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses)+1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r', marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, 'orange', marker='o', label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', marker='o', label='Train Acc')
    plt.plot(epochs, val_accuracies, 'green', marker='o', label='Val Acc')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_plot.png")
    print("Training graph saved as training_plot.png")

# -------------------------------
# STEP 7: Start training
# -------------------------------
if __name__ == "__main__":
    train_model()

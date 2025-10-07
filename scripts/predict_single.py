import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

# Setup
image_path = "n.jpg"  # Replace with your image file
num_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transform (same as test set)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load image
image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

# Load model with SAME structure as training
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Dropout(0.6),  # SAME dropout used in training
    nn.Linear(model.fc.in_features, num_classes)
)
model.load_state_dict(torch.load("covid_resnet18_model.pth", map_location=device))
model.to(device)
model.eval()

# Predict
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    _, predicted = torch.max(probabilities, 1)

# Map prediction to class name
class_names = ['early', 'normal', 'severe']
predicted_class = class_names[predicted.item()]
confidence = probabilities[0][predicted.item()] * 100

# Print result
print(f"\nPrediction: {predicted_class.upper()} ({predicted.item()})")
print(f"Confidence: {confidence:.2f}%")

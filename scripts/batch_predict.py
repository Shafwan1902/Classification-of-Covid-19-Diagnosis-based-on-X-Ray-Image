import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
from tqdm import tqdm

# Setup
input_folder = "batch_images"  # Replace with your folder containing images
model_path = "covid_resnet18_model.pth"
num_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transform (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Class labels
class_names = ['early', 'normal', 'severe']

# Prediction loop
results = []
print(f"\n[INFO] Predicting on images in: {input_folder}\n")
for filename in tqdm(os.listdir(input_folder)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, filename)
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        results.append({
            'filename': filename,
            'prediction': class_names[pred.item()],
            'confidence': f"{conf.item()*100:.2f}%"
        })

# Display results
print("\nBatch Prediction Results:")
for r in results:
    print(f"{r['filename']:<30} → {r['prediction'].upper()} ({r['confidence']})")

# Optional: Save to CSV
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("batch_predictions.csv", index=False)
print("\n[INFO] Saved predictions to 'batch_predictions.csv'")

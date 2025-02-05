import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import sys

MODEL_PATH = "models/resnet50_ohm.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # แก้ชั้น output เป็น 2 class
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return "Ohm" if predicted.item() == 0 else "Not Ohm"

if __name__ == "__main__":
    image_path = sys.argv[1]  # รับ path ของภาพจาก command-line
    print(f"Prediction: {predict(image_path)}")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import os

# Configurations
DATA_DIR = "data"
MODEL_SAVE_PATH = "models/resnet50_ohm.pth"
BATCH_SIZE = 16
MAX_EPOCHS = 50  # ตั้งค่าฝึกสูงสุด
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

# Data Transformations (เพิ่ม Augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Dataset
train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load Pre-trained Model (ResNet50)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, 2)  # แก้ Fully Connected Layer เป็น 2 Classes

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # ลด LR ทุก 5 Epoch

best_val_acc = 0.0
early_stop_count = 0
EARLY_STOPPING_THRESHOLD = 5  # ถ้า accuracy ไม่เพิ่ม 5 รอบติด ให้หยุด

# Training Loop
for epoch in range(MAX_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    
    # Validation Loop
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{MAX_EPOCHS}], Loss: {avg_train_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

    # Save Best Model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"🎯 New Best Model Saved! Accuracy: {best_val_acc:.2f}%")

    # Early Stopping Check
    if val_acc >= 95.0:
        print(f"✅ Training Stopped: Validation Accuracy reached {val_acc:.2f}%!")
        break
    elif val_acc <= best_val_acc:
        early_stop_count += 1
    else:
        early_stop_count = 0  # รีเซ็ตถ้ามี improvement

    if early_stop_count >= EARLY_STOPPING_THRESHOLD:
        print("🛑 Early Stopping: No improvement for 5 epochs!")
        break

    scheduler.step()  # ลด Learning Rate ถ้าจำเป็น

print(f"🔥 Best Validation Accuracy: {best_val_acc:.2f}%")

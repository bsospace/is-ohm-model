from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
from PIL import Image
import io
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 กำหนดค่าหลัก
DATA_DIR = "data"  # ตรวจสอบให้แน่ใจว่าโฟลเดอร์นี้มีอยู่
MODEL_PATH = "models/resnet50_ohm.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 🔹 โหลด Dataset เพื่อตรวจสอบ Mapping ของคลาส
train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=transform)
class_to_idx = train_dataset.class_to_idx  # {'Ohm': 0, 'Not Ohm': 1} หรืออาจเป็น {'Not Ohm': 0, 'Ohm': 1}
idx_to_class = {v: k for k, v in class_to_idx.items()}  # กลับเป็น {0: 'Ohm', 1: 'Not Ohm'}

# 🔹 โหลดโมเดล
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))  # ใช้ len() แทน 2 เผื่อมีคลาสเพิ่มในอนาคต
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    result = idx_to_class[predicted.item()]  # ใช้ Mapping จาก Dataset
    return {"prediction": result}

# 🔹 Run API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

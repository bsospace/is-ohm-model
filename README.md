# is-ohm-model

## 📌 Overview
**is-ohm-model** เป็นโมเดล Machine Learning ที่ใช้ **Pre-trained ResNet50 (ImageNet)** ในการจำแนกภาพว่าเป็น "โอม" หรือไม่ โดยใช้ **PyTorch** และสามารถใช้งานผ่าน API ที่พัฒนาโดย **FastAPI**

---

## 📂 Project Structure
```
is-ohm-model/
│── data/                          # เก็บรูปภาพทั้งหมด
│   ├── train/
│   │   ├── ohm/                   # รูปของ "โอม" (Positive Class)
│   │   ├── not_ohm/               # รูปของคนอื่น (Negative Class)
│   ├── val/
│   │   ├── ohm/
│   │   ├── not_ohm/
│
│── models/                        # เก็บโมเดลที่ฝึกเสร็จ
│   ├── resnet50_ohm.pth           # ไฟล์ weights ที่ถูกฝึกแล้ว
│
│── scripts/                        # โค้ดหลักของโปรเจค
│   ├── train.py                    # ไฟล์สำหรับฝึกโมเดล
│   ├── infer.py                    # ทำนายผลจากโมเดล
│   ├── utils.py                    # ฟังก์ชันช่วยต่าง ๆ
│
│── api.py                          # FastAPI สำหรับให้บริการ inference
│── requirements.txt                 # รายการ dependencies
│── README.md                        # คำอธิบายโปรเจค
```

---

## ⚙️ Installation
### 1️⃣ **ติดตั้ง Dependencies**
```bash
pip install -r requirements.txt
```

หากใช้ GPU NVIDIA ให้ติดตั้ง Torch เวอร์ชันที่รองรับ:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2️⃣ **เตรียม Dataset**
- วางภาพไว้ที่ `data/train/` และ `data/val/`
- ไฟล์รูปต้องอยู่ภายใต้โฟลเดอร์ `ohm/` และ `not_ohm/`

---

## 🚀 Training the Model
รันคำสั่งนี้เพื่อฝึกโมเดล:
```bash
python scripts/train.py
```

เมื่อฝึกเสร็จ โมเดลจะถูกบันทึกไว้ที่ `models/resnet50_ohm.pth`

---

## 🔍 Inference (Prediction)
### 1️⃣ **รันคำสั่งผ่าน CLI**
```bash
python scripts/infer.py data/val/ohm/test1.jpg
```

### 2️⃣ **ใช้ API ผ่าน FastAPI**
รัน API:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

ทดสอบ API ด้วย `cURL`:
```bash
curl -X 'POST' 'http://localhost:8000/predict/' -F 'file=@test.jpg'
```

ตัวอย่าง Response:
```json
{
  "prediction": "Ohm"
}
```

---

## 🎯 Next Steps
- ✅ ปรับปรุง Data Augmentation ให้ดียิ่งขึ้น
- ✅ ทดลองใช้โมเดล EfficientNet / ViT
- ✅ พัฒนา Web UI สำหรับอัปโหลดและทำนายผล

หากมีคำถามหรือข้อเสนอแนะ สามารถติดต่อได้! 🚀
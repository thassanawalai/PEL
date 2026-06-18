# Leaf Classification with YOLOv8-seg + PEL

โปรเจคนี้แยกชนิดใบไม้ 11 คลาสจากโฟลเดอร์ `DATA - Copy` โดยแบ่งงานเป็น 2 ส่วน:

1. **YOLOv8-seg background removal**: ใช้โมเดล instance segmentation ที่เทรนจาก polygon mask ของใบไม้ เพื่อสร้างภาพใบไม้ที่ครอปและตัดพื้นหลังออก
2. **Prototype-enhanced Learning (PEL)**: เทรน classifier ที่เรียนรู้ class prototypes/label embeddings เพื่อให้โมเดลเห็นความสัมพันธ์ระหว่างใบไม้ที่คล้ายกัน แทนการพึ่ง one-hot label เพียงอย่างเดียว

## โครงสร้างไฟล์

```text
PEL/
├── DATA - Copy/                 # raw images: 11 class folders
├── data_segmented/              # output จาก YOLOv8-seg (สร้างหลังรัน phase 1)
├── data/                        # train/val/test ImageFolder dataset
├── checkpoints/                 # YOLO weights และ classifier checkpoints
├── configs/config.py            # config กลางของระบบ
├── models/model.py              # baseline ResNet classifier
├── models/pel_model.py          # PEL classifier
├── scripts/phase1_segmentation.py
├── scripts/train.py
├── scripts/evaluate.py
├── split_dataset.py
└── requirements.txt
```

## การติดตั้ง

```powershell
venv\Scripts\activate
pip install -r requirements.txt
```

## ขั้นตอนการทำงาน

### 1. เตรียม YOLOv8-seg weight

นำ weight ที่เทรนจาก Roboflow polygon mask มาวางที่:

```text
checkpoints/leaf_yolov8_seg.pt
```

หรือระบุ path เองตอนรัน:

```powershell
python scripts/phase1_segmentation.py --model path\to\best.pt
```

หมายเหตุ: ไม่ควรใช้ `yolov8n-seg.pt` ของ COCO เป็นตัวหลัก เพราะไม่ได้เทรนมาเพื่อ mask ใบไม้ชนิดนี้โดยตรง

### 2. ตัดพื้นหลังและครอปใบไม้

```powershell
python scripts/phase1_segmentation.py
```

ค่า default:

- input: `DATA - Copy`
- output: `data_segmented`
- background: `black`

ถ้าต้องการพื้นหลังโปร่งใส:

```powershell
python scripts/phase1_segmentation.py --background transparent
```

### 3. แบ่ง dataset

```powershell
python split_dataset.py
```

สคริปต์จะใช้ `data_segmented` ถ้ามีอยู่ ไม่เช่นนั้นจะใช้ `DATA - Copy` และจะล้าง `data/train`, `data/val`, `data/test` ก่อน split เพื่อกันข้อมูลเก่าค้าง

### 4. เทรน PEL classifier

```powershell
python scripts/train.py
```

ระบบจะตรวจจำนวนคลาสจาก `data/train` โดยอัตโนมัติ ถ้าต้องการปิด PEL ให้แก้ `USE_PEL = False` ใน `configs/config.py`

### 5. ประเมินผล

```powershell
python scripts/evaluate.py
```

## Config สำคัญ

ดูและแก้ได้ที่ `configs/config.py`

- `RAW_DATA_DIR`: โฟลเดอร์ภาพดิบ
- `SEGMENTED_DATA_DIR`: โฟลเดอร์ภาพที่ตัดพื้นหลังแล้ว
- `DATA_DIR`: dataset สำหรับ train/val/test
- `PRETRAINED`: ค่า default เป็น `False` เพื่อให้เทรนได้แม้ไม่มี cached ImageNet weights; เปิดเป็น `True` ได้ถ้ามีอินเทอร์เน็ตหรือ cache พร้อม
- `USE_PEL`: เปิด/ปิด PEL
- `PEL_PULL_LOSS_WEIGHT`: น้ำหนัก loss ที่ดึงภาพเข้าหา prototype ของคลาสจริง
- `PEL_SOFT_TARGET_LOSS_WEIGHT`: น้ำหนัก soft-label loss จากความสัมพันธ์ของ prototypes
- `YOLO_SEG_MODEL`: path ของ YOLOv8-seg weight

## จุดที่ต้องระวัง

- `DATA - Copy` ต้องเป็น class-folder structure เช่น `DATA - Copy/001/*.jpg`
- `data/train`, `data/val`, `data/test` ต้องมีชื่อ class folders ตรงกัน
- ถ้ายังไม่มี YOLOv8-seg weight ที่เทรนเอง ให้ข้าม phase 1 แล้วรัน `split_dataset.py --source "DATA - Copy"` เพื่อเทรน classifier จาก raw images ไปก่อน

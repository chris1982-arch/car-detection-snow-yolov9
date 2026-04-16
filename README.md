# Car Detection in Snow — YOLOv9

## Project Description
Vehicle detection in snowy conditions using YOLOv9 fine-tuned on the Nordic Vehicle Dataset (NVD). UAV footage from northern Sweden.

## Dataset
[Nordic Vehicle Dataset (NVD)](https://nvd.ltu-ai.dev/)

**Split:**
- Train: 3,034 frames (Asjo 01, Asjo 01_HD, Bjenberg 02)
- Val: 236 frames (Nyland 01)
- Test: 2,027 frames (Bjenberg 02_stabilized)

## Model
- Architecture: YOLOv9c
- Pretrained on: COCO dataset
- Fine-tuned on: NVD dataset
- Training: 100 epochs with augmentation

## Results
| Model | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| YOLOv5s (paper) | 54.2% | 33.7% | 47.3% |
| YOLOv8s (paper) | 65.8% | 22.4% | 45.1% |
| YOLOv9c 50ep (ours) | 55.5% | 25.0% | 24.1% |
| YOLOv9c 100ep+Aug (ours) | 58.2% | 42.5% | 42.5% |

## Trained Weights
Download best.pt from Google Drive: [best.pt](https://drive.google.com/file/d/10861JkuUGhX0lDWQy5_PGk1AZLA8enBt/view?usp=sharing)

## How to Run
pip install ultralytics

yolo predict model=best.pt source=YOUR_IMAGE conf=0.01

## Requirements
- Python 3.10
- PyTorch 2.5.1 + CUDA
- Ultralytics
- OpenCV

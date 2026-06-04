from pathlib import Path
from ultralytics import YOLO

parent_dir = Path(__file__).resolve().parent

# 1. 사전 학습된 기본 YOLOv8 모델 로드 (전이 학습용)
# Nano(n), Small(s), Medium(m) 중 선택. 빠르고 가벼운 Nano 모델로 시작하는 것을 권장합니다.
model = YOLO("yolov8n.pt")

# 2. 모델 학습 실행
# data: 2단계에서 작성한 yaml 파일 경로
# epochs: 전체 데이터를 반복 학습할 횟수
# batch: 한 번에 학습할 이미지 수 (macOS 메모리 용량에 따라 8, 16 등으로 조절)
# device: 'mps' (Apple Silicon 가속), Intel Mac의 경우 'cpu' 입력
results = model.train(
    data=f"{parent_dir}/dataset/data.yml",
    epochs=50,
    batch=16,
    imgsz=640,
    device="mps",
    name="safety_sign_model",
)

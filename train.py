from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="https://universe.roboflow.com/ds/VD3UKpYNPHwb3mCO5Pqt/3/download/yolov8",
    epochs=50,
    imgsz=640,
    batch=8,
    name="road_damage",
    project="runs",
)

print("Training done! Model saved to runs/road_damage/weights/best.pt")
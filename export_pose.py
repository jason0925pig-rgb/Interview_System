from ultralytics import YOLO

model = YOLO("yolo11m-pose.pt")
model.export(format="onnx")
print("Exported ONNX model")

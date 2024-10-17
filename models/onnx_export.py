import torch.onnx

from ultralytics import YOLO

# model_path = "models/tennisball_yolov8.pt"
# # model = torch.load(model_path)
# model = YOLO(model_path)
# model = model.model
# model.eval() # set model in eval mode

# dummy_input = torch.randn(1, 3, 640, 640)

# torch.onnx.export(
#     model, 
#     dummy_input, 
#     "tennisball_yolov8.onnx", 
#     verbose=True,
#     export_params=True
# )


# model = YOLO("models/tennisball_yolov8.pt")
# model.export(format="onnx", imgsz=320)


model = YOLO("box5_yolov8n.pt")
model.export(format="onnx", imgsz=640)

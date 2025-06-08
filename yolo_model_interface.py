from ultralytics import YOLO
from model_interface import BaseModel

class YOLOModel(BaseModel):
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.names = self.model.names

    def predict(self, frame):
        results = self.model(frame, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'class_id': int(box.cls[0]),
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].cpu().numpy().astype(int)
                })
        return detections

    def get_label(self, class_id):
        return self.names[class_id]

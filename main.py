from model_interfaces.yolo_model_interface import YOLOModel
from garbage_classifier import GarbageClassifier


# Example usage
if __name__ == "__main__":
    detection_model = YOLOModel(model_path='./models/best.pt')
    gc = GarbageClassifier(detection_model=detection_model)  # Using a generic model for testing
    gc.run_video_stream(video_source=0)  # Use 0 for default webcam, or your IP cam URL
from model_interfaces.base_model_interface import BaseModel
import tensorflow as tf
import numpy as np
import cv2

class TensorFlowModel(BaseModel):
    def __init__(self, model_path):
        # Load the SavedModel (signature: 'serving_default')
        self.model = tf.saved_model.load(model_path)
        self.infer = self.model.signatures["serving_default"]

        # Example label map (should match your model’s training)
        self.label_map = {
            1: 'plastic',
            2: 'metal',
            3: 'paper'
        }

    def preprocess(self, frame):
        # Resize and normalize image if required
        # Assume the model expects 640x640 and normalized float32
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(img_rgb)
        input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dimension
        return input_tensor

    def predict(self, frame):
        input_tensor = self.preprocess(frame)
        output_dict = self.infer(input_tensor)

        num_detections = int(output_dict['num_detections'].numpy()[0])
        boxes = output_dict['detection_boxes'][0].numpy()
        classes = output_dict['detection_classes'][0].numpy().astype(int)
        scores = output_dict['detection_scores'][0].numpy()

        height, width, _ = frame.shape
        results = []

        for i in range(num_detections):
            score = scores[i]
            if score < 0.4:
                continue

            y_min, x_min, y_max, x_max = boxes[i]
            x1 = int(x_min * width)
            y1 = int(y_min * height)
            x2 = int(x_max * width)
            y2 = int(y_max * height)

            results.append({
                'class_id': classes[i],
                'confidence': float(score),
                'bbox': [x1, y1, x2, y2]
            })

        return results

    def get_label(self, class_id):
        return self.label_map.get(class_id, "unknown")

import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from trash_servo import TrashCom, CMDS  # Assuming this exists
import threading
from jetsonConsumer import JetsonConsumer

# Define your target input size as used in cnn_garbage
target_size = (128, 128)  # <-- Adjust this to match your CNN input shape

class cnn_garbage:
    def __init__(self):
        self.model = self.get_model_structure()
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()]
             )
    
    def get_model_structure(self):
        from tensorflow.keras import layers
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, ZeroPadding2D
        model = Sequential()
        model.add(ZeroPadding2D(padding=(1, 1), input_shape=(target_size[0], target_size[1], 3)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Dropout(0.4))
        model.add(MaxPool2D((2, 2)))
        model.add(Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(64, activation='relu'))
        model.add(Dense(6, activation='softmax'))
        return model

class GarbageClassifier:
    def __init__(self, model_weights_path='last.h5'):
        # Load cnn_garbage model and weights
        self.cnn_model_obj = cnn_garbage()
        self.model = self.cnn_model_obj.model
        print(f"Loading weights from {model_weights_path} ...")
        self.model.load_weights(model_weights_path)
        print("Weights loaded.")

        self.bin = TrashCom()
        self.consumer = JetsonConsumer()

        # Define your classes according to the CNN output order
        # Update this list according to your dataset/classes
        self.class_names = ['plastic_bottle', 'plastic_cup', 'plastic_bag', 'metal_can', 'paper_box', 'other']  

        # --- Threading and State Variables ---
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.camera_running = threading.Event()
        self.camera_thread = None
        self.video_source = 0

        # --- Detection Stability Variables ---
        self.CONFIDENCE_THRESHOLD = 0.4  # Softmax confidence threshold
        self.MIN_DETECTION_DURATION = 1.0  # seconds

        self.current_detection_category = None
        self.detection_start_time = None
        self.action_triggered_for_this_detection = False

        # --- Post-Action Sleep Variables ---
        self.post_action_sleep_end_time = 0
        self.POST_ACTION_SLEEP_DURATION = 2.0

        self.last_processed_display_frame = None  # To show during sleep

    def _camera_reader_thread(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.video_source}.")
            self.camera_running.clear()
            return

        print("Camera thread started.")
        while self.camera_running.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            with self.frame_lock:
                self.latest_frame = frame.copy()
            time.sleep(1 / 30)

        cap.release()
        print("Camera thread stopped and released.")

    def start_camera_capture(self, video_source=0):
        if self.camera_thread is not None and self.camera_thread.is_alive():
            print("Camera is already running.")
            return

        self.video_source = video_source
        self.camera_running.set()
        self.camera_thread = threading.Thread(target=self._camera_reader_thread, daemon=True)
        self.camera_thread.start()
        time.sleep(1)
        if self.latest_frame is None and self.camera_running.is_set():
            print("Warning: Camera started but no frame received yet. Check camera connection/source.")

    def stop_camera_capture(self):
        print("Stopping camera capture...")
        self.camera_running.clear()
        if self.camera_thread is not None and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2)
            if self.camera_thread.is_alive():
                print("Warning: Camera thread did not stop in time.")
        self.camera_thread = None
        print("Camera capture stopped.")

    def preprocess_frame(self, frame):
        # Resize and normalize to [0,1]
        img = cv2.resize(frame, target_size)
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)  # Batch dimension

    def classify_label(self, predicted_index, confidence):
        if confidence < self.CONFIDENCE_THRESHOLD:
            return None  # No confident prediction

        label = self.class_names[predicted_index]

        # Map your label names to simplified categories for servo commands
        # Adjust as per your actual class definitions
        if 'plastic' in label:
            return 'plastic'
        elif 'metal' in label:
            return 'metal'
        elif 'paper' in label:
            return 'paper'
        else:
            return 'rest'

    def _process_frame_logic(self, frame_to_process):
        if frame_to_process is None:
            return None, False

        output_frame = frame_to_process.copy()
        input_tensor = self.preprocess_frame(frame_to_process)

        preds = self.model.predict(input_tensor, verbose=0)[0]
        predicted_index = np.argmax(preds)
        confidence = preds[predicted_index]

        category = self.classify_label(predicted_index, confidence)

        # Annotate frame with prediction and confidence
        label_text = f"{category if category else 'Unknown'}: {confidence:.2f}"
        color_map = {'plastic': (0, 255, 0), 'metal': (0, 0, 255), 'paper': (255, 255, 0), 'rest': (128, 128, 128)}
        color = color_map.get(category, (255, 255, 255))
        cv2.putText(output_frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        current_time = time.time()
        action_taken_this_frame = False

        if category is not None:
            if self.current_detection_category == category:
                if not self.action_triggered_for_this_detection and \
                        (current_time - self.detection_start_time >= self.MIN_DETECTION_DURATION):
                    print(f"Category '{self.current_detection_category}' detected for {self.MIN_DETECTION_DURATION}s. Triggering action.")
                    if self.current_detection_category == 'plastic':
                        self.consumer.plastic()
                    elif self.current_detection_category == 'metal':
                        self.consumer.metal()
                    elif self.current_detection_category == 'paper':
                        self.consumer.other()
                    else:
                        self.consumer.other()

                    self.action_triggered_for_this_detection = True
                    action_taken_this_frame = True
            else:
                print(f"New potential detection: '{category}'. Starting timer.")
                self.current_detection_category = category
                self.detection_start_time = current_time
                self.action_triggered_for_this_detection = False
        else:
            # No confident prediction, reset detection
            self.current_detection_category = None
            self.detection_start_time = None
            self.action_triggered_for_this_detection = False

        self.last_processed_display_frame = output_frame
        return output_frame, action_taken_this_frame

    def run_video_stream(self, video_source=0):
        self.start_camera_capture(video_source)

        if not self.camera_running.is_set() or self.latest_frame is None:
            print("Camera not available or failed to start. Exiting.")
            self.stop_camera_capture()
            return

        print("Main loop started. Press 'q' to quit.")
        while self.camera_running.is_set():
            current_time = time.time()

            if current_time < self.post_action_sleep_end_time:
                if self.last_processed_display_frame is not None:
                    cv2.imshow("Garbage Classification", self.last_processed_display_frame)
                else:
                    with self.frame_lock:
                        if self.latest_frame is not None:
                            cv2.imshow("Garbage Classification", self.latest_frame)

                if cv2.waitKey(1)

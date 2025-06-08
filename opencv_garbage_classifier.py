import time
import cv2
from ultralytics import YOLO
from trash_servo import TrashCom, CMDS  # Assuming this file exists and works
import threading
from jetsonConsumer import JetsonConsumer 
from model_interfaces.base_model_interface import BaseModel
from model_interfaces.yolo_model_interface import YOLOModel



class GarbageClassifier:
    def __init__(self, detection_model: BaseModel):
        self.model = detection_model
        self.bin = TrashCom()
        self.consumer = JetsonConsumer()
        # Mapping labels (adjust based on your model's actual output names)
        # self.model.names will contain the actual names your model uses for class IDs
        # It's better to rely on self.model.names if possible, or ensure these lists match
        self.plastic_labels = ['bottle', 'cup', 'plastic bag']  # Example: Added 'plastic bag'
        self.metal_labels = ['can', 'metal']  # Example: Added 'metal'
        self.paper_labels = ['paper', 'box', 'book']
        # "rest" means all other detected objects or undetected garbage

        # --- Threading and State Variables ---
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.camera_running = threading.Event()
        self.camera_thread = None
        self.video_source = 0

        # --- Detection Stability Variables ---
        self.CONFIDENCE_THRESHOLD = 0.4
        self.MIN_DIAGONAL_PX = 50  # Minimum size to consider (example, adjust)
        self.MAX_DIAGONAL_PX = 700  # Max size to avoid huge false positives

        self.current_detection_category = None
        self.detection_start_time = None
        self.action_triggered_for_this_detection = False
        self.MIN_DETECTION_DURATION = 1.0  # seconds

        # --- Post-Action Sleep Variables ---
        self.post_action_sleep_end_time = 0
        self.POST_ACTION_SLEEP_DURATION = 2.0

        self.last_processed_display_frame = None  # To show during sleep

    def _camera_reader_thread(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.video_source}.")
            self.camera_running.clear()  # Signal main thread that camera failed
            return

        print("Camera thread started.")
        while self.camera_running.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Camera thread: Failed to grab frame or stream ended.")
                # Optionally try to reopen, or just break
                time.sleep(0.1)  # Avoid busy loop if stream ends
                continue  # Or break, depending on desired behavior

            with self.frame_lock:
                self.latest_frame = frame.copy()  # Store a copy
            time.sleep(1 / 30)  # Aim for ~30 FPS reading, adjust as needed

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
        # Wait a moment for the first frame to be captured
        time.sleep(1)  # Give camera time to initialize and get first frame
        if self.latest_frame is None and self.camera_running.is_set():
            print("Warning: Camera started but no frame received yet. Check camera connection/source.")

    def stop_camera_capture(self):
        print("Stopping camera capture...")
        self.camera_running.clear()
        if self.camera_thread is not None and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2)  # Wait for thread to finish
            if self.camera_thread.is_alive():
                print("Warning: Camera thread did not stop in time.")
        self.camera_thread = None
        print("Camera capture stopped.")

    def classify_label(self, label):
        label_lower = label.lower()
        # Prioritize specific labels if they overlap
        if any(pl in label_lower for pl in self.plastic_labels):
            return 'plastic'
        elif any(ml in label_lower for ml in self.metal_labels):
            return 'metal'
        elif any(ppl in label_lower for ppl in self.paper_labels):
            return 'paper'
        return 'rest'  # Default category

    def _process_frame_logic(self, frame_to_process):

        if frame_to_process is None:
            return None, False

        output_frame = frame_to_process.copy()
        detections = self.model.predict(frame_to_process)

        detected_categories_in_frame = set()
        action_taken_this_frame = False
        best_current_detection = None

        for det in detections:
            conf = det['confidence']
            if conf < self.CONFIDENCE_THRESHOLD:
                continue

            class_id = det['class_id']
            label = self.model.get_label(class_id)
            category = self.classify_label(label)

            x1, y1, x2, y2 = det['bbox']
            diagonal = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            if not (self.MIN_DIAGONAL_PX < diagonal < self.MAX_DIAGONAL_PX):
                continue

            detected_categories_in_frame.add(category)

            # Draw detection box
            color_map = {
                'plastic': (0, 255, 0),
                'metal': (0, 0, 255),
                'paper': (255, 255, 0),
                'rest': (128, 128, 128)
            }
            color = color_map.get(category, (255, 255, 255))
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output_frame, f'{category} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if best_current_detection is None:
                best_current_detection = category

        # --- Stability and Action Triggering ---
        current_time = time.time()

        if best_current_detection is not None:
            if self.current_detection_category == best_current_detection:
                if not self.action_triggered_for_this_detection and \
                        (current_time - self.detection_start_time >= self.MIN_DETECTION_DURATION):
                    print(f"Category '{self.current_detection_category}' detected for required duration. Triggering action.")

                    # API Calls for Jetson to act on the detection
                    # if self.current_detection_category == 'plastic':
                    #     self.consumer.plastic()
                    # elif self.current_detection_category == 'metal':
                    #     self.consumer.metal()
                    # elif self.current_detection_category == 'paper':
                    #     self.consumer.other()
                    # else:
                    #     self.consumer.other()

                    self.action_triggered_for_this_detection = True
                    action_taken_this_frame = True
            else:
                print(f"New potential detection: '{best_current_detection}'. Starting timer.")
                self.current_detection_category = best_current_detection
                self.detection_start_time = current_time
                self.action_triggered_for_this_detection = False

        self.last_processed_display_frame = output_frame
        return output_frame, action_taken_this_frame

    def run_video_stream(self, video_source=0):
        self.start_camera_capture(video_source)

        if not self.camera_running.is_set() or self.latest_frame is None:
            # This check handles if camera failed to start in _camera_reader_thread
            # or if start_camera_capture itself determined no frame was received.
            print("Camera not available or failed to start. Exiting.")
            self.stop_camera_capture()  # Ensure cleanup if partially started
            return

        print("Main loop started. Press 'q' to quit.")
        while self.camera_running.is_set():  # Loop as long as camera thread should be running
            current_time = time.time()

            # 1. Handle post-action sleep
            if current_time < self.post_action_sleep_end_time:
                # We are in the 2-second sleep period
                if self.last_processed_display_frame is not None:
                    cv2.imshow("Garbage Classification", self.last_processed_display_frame)
                else:  # Fallback if no frame processed yet
                    with self.frame_lock:
                        if self.latest_frame is not None:
                            cv2.imshow("Garbage Classification", self.latest_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.5)  # Don't busy-wait
                continue

            # 2. Get the latest frame from the camera thread
            frame_copy = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_copy = self.latest_frame.copy()

            if frame_copy is None:
                # print("No frame available from camera thread yet, sleeping briefly.")
                time.sleep(0.01)  # Wait for a frame
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Still allow quit
                    break
                continue

            # 3. Process the frame
            annotated_frame, action_triggered = self._process_frame_logic(frame_copy)

            # 4. If action was triggered, set up the 2-second sleep for NEXT iterations
            if action_triggered:
                print(f"Action triggered. Entering {self.POST_ACTION_SLEEP_DURATION}s cooldown.")
                self.post_action_sleep_end_time = current_time + self.POST_ACTION_SLEEP_DURATION
                # Reset detection state to require new full duration detection after cooldown
                self.current_detection_category = None
                self.detection_start_time = None
                self.action_triggered_for_this_detection = False

            # 5. Display the (potentially annotated) frame
            if annotated_frame is not None:
                cv2.imshow("Garbage Classification", annotated_frame)
            else:  # Should not happen if frame_copy was valid
                cv2.imshow("Garbage Classification", frame_copy)

            # 6. Handle quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed, exiting...")
                break

        # Cleanup
        self.stop_camera_capture()
        cv2.destroyAllWindows()
        print("Application finished.")


# Example usage
if __name__ == "__main__":
    detection_model = YOLOModel(model_path='./models/best.pt')
    gc = GarbageClassifier(detection_model=detection_model)  # Using a generic model for testing
    gc.run_video_stream(video_source=0)  # Use 0 for default webcam, or your IP cam URL
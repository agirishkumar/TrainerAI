import torch
from ultralytics import YOLO
import cv2
import time
import numpy as np
import openpose

class HumanDetector:
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.5):
        """
        Initializes a new instance of the class.

        Parameters:
            model_name (str): The name of the YOLO model to use. Defaults to 'yolov8n.pt'.
            confidence_threshold (float): The threshold for confidence scores of detected humans. Defaults to 0.5.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_name).to(self.device)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        """
        Detects humans in a given frame using a pre-trained model.

        Parameters:
            frame (numpy.ndarray): The frame in which humans are to be detected.

        Returns:
            Any: The results of the human detection.
        """
        results = self.model(frame)
        # print(results)
        return results

    def draw_boxes(self, frame, results):
        annotated_frame = results[0].plot()
        return annotated_frame

def resize_frame(frame, target_width=None, target_height=None):
    """
    Resizes a given frame to a specified target width and height. If only one of the target dimensions is provided,
    the other dimension is calculated based on the aspect ratio of the original frame. If neither target dimension is
    provided, the original frame is returned unchanged.

    Parameters:
        frame (numpy.ndarray): The frame to be resized.
        target_width (int, optional): The desired width of the resized frame. If not provided, the height parameter
            must be provided.
        target_height (int, optional): The desired height of the resized frame. If not provided, the width parameter
            must be provided.

    Returns:
        numpy.ndarray: The resized frame with the specified target dimensions. If the target dimensions are not
            provided, the original frame is returned unchanged.
    """
    height, width = frame.shape[:2]
    if target_width and not target_height:
        ratio = target_width / width
        target_height = int(height * ratio)
    elif target_height and not target_width:
        ratio = target_height / height
        target_width = int(width * ratio)
    elif not target_width and not target_height:
        return frame
    return cv2.resize(frame, (target_width, target_height))

def calculate_fps(start_time):
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    return fps

def process_video(video_path, target_width=640, output_path=None):
    detector = HumanDetector()
    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    total_time = 0

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (target_width, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * (target_width / cap.get(cv2.CAP_PROP_FRAME_WIDTH)))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.time()
        frame = resize_frame(frame, target_width=target_width)
        results = detector.detect(frame)
        if results:  
            annotated_frame = detector.draw_boxes(frame, results)
        else:
            annotated_frame = frame
        fps = calculate_fps(start_time)
        total_time += 1 / fps
        total_frames += 1
        cv2.imshow('Human Detection (q to quit)', annotated_frame)
        print(f"FPS: {fps:.2f}")
        if output_path:
            writer.write(annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if output_path:
        writer.release()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'squat.mp4'
    process_video(video_path)
from PyQt6.QtCore import QThread, pyqtSignal

from ultralytics import YOLO

import os
import cv2

from app.utils.pre_proccessing import ProccessingImage
from app.recognizer.OCR import OCR

class CameraThread(QThread):
    result_signal = pyqtSignal(object)

    def __init__(self, device="cuda", model_name="best_detection_cccd.pt"):
        super().__init__()
        self.frame_to_proccess = None
        self.root = os.getcwd()
        self.proccessingImage = ProccessingImage()
        self.ocr = OCR()
        if device not in ["cuda", "cpu"]:
            self.device = "cpu"
        else:
            self.device = device
        # Path file temporatory
        self.temp_dir = os.path.join(self.root, "temp")
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        self.img_temp_path = os.path.join(self.temp_dir, "temp.jpg")

        # Models initialize only one
        print("Loading YOLO model...")
        self.model_yolo = YOLO(os.path.join(self.root, "model", model_name))
        print("Loading OCR & Processing...")
        self.processing_image = ProccessingImage()
        self.ocr = OCR()
        print("All models loaded.")

    def set_frame(self, frame):
        self.frame_to_proccess = frame

    def run(self):
        if self.frame_to_proccess is None:
            return
        
        try:
            #1. YOLO Detect
            results = self.model_yolo.predict(self.frame_to_proccess, conf=0.8, device=self.device)

            if not results or len(results[0].boxes) == 0:
                self.result_signal.emit({"status": "No detection"})
                return
            
            result = results[0]
            xyxy = result.boxes.xyxy.cpu().numpy()
            top_x = int(xyxy[0][0])
            top_y = int(xyxy[0][1])
            bottom_x = int(xyxy[0][2])
            bottom_y = int(xyxy[0][3])

            fix_top_x, fix_bottom_x, fix_top_y, fix_bottom_y = top_x - 50, bottom_x + 50, top_y - 50, bottom_y + 50

            if all(v >= 0 for v in [fix_top_x, fix_bottom_x, fix_top_y, fix_bottom_y]):
                cropped_image = self.frame_to_proccess[fix_top_y:fix_bottom_y, fix_top_x:fix_bottom_x]    
            else:
                cropped_image = self.frame_to_proccess[top_y:bottom_y, top_x:bottom_x]

            # Xoay ảnh 90 độ theo chiều kim đồng hồ
            cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(self.img_temp_path, cropped_image)

            #2. Preproccessing Image
            image_proccessed = self.proccessingImage.focus_image(self.img_temp_path)

            #3. OCR
            information = self.ocr.predict(image_proccessed)
            result = {
                "status": "Successfully",
                "information": information
            }
            self.result_signal.emit(result)
        except Exception as e:
            print(f"Error in Thread: {e}")
            self.result_signal.emit({"status": str(e)})
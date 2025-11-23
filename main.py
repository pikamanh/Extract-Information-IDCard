from PyQt6.uic import load_ui
from PyQt6.QtWidgets import QApplication

import cv2
import os
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer

from app.utils.threading import CameraThread
class Camera:
    def __init__(self, skip_frame=30):
        self.root = os.getcwd()
        self.cameraThread = CameraThread()
        self.cap = None
        self.frame_count = 0
        self.skip = skip_frame
        self.mainui = load_ui.loadUi("app/resources/main.ui")
        self.mainui.show()
        
        self.cameraThread.result_signal.connect(self.handle_result)
        self.mainui.StartCamera.clicked.connect(self.start_camera)
        self.mainui.StopCamera.clicked.connect(self.stop_camera)
        
        self.video_label = self.mainui.CameraDisplay
        self.video_label.setScaledContents(True)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.frame_count = 0

        if not self.timer.isActive():
            self.timer.start(30)
    def update_frame(self):
        ret, frame = self.cap.read()

        if ret:
            self.frame_count+=1
            if self.frame_count % self.skip == 0:
                if not self.cameraThread.isRunning():
                    self.cameraThread.set_frame(frame.copy())
                    self.cameraThread.start()
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w, c = frame.shape
            bytes_per_line = c * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))
            self.video_label.repaint()

    def handle_result(self, data):
        if data["status"] == "No detection":
            print("No detection")
        elif data["status"] == "Successfully":
            print(data["information"])
        else:
            print(data["status"])

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_label.clear()

        if self.cameraThread.isRunning():
            self.cameraThread.terminate()

if __name__ == "__main__":
    app = QApplication([])
    main = Camera()
    app.exec()
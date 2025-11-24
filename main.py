from PyQt6.uic import load_ui
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox, QDialogButtonBox

import cv2
import os
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer

from app.utils.threading import CameraThread
class Setup(QDialog):
    def __init__(self):
        super().__init__()
        load_ui.loadUi("app/resources/setup.ui", self)

        self.BoxChoose.currentIndexChanged.connect(self.toggle_RSTP)
        self.btn_ok = self.buttonBox.button(QDialogButtonBox.StandardButton.Ok)
        self.btn_cancel = self.buttonBox.button(QDialogButtonBox.StandardButton.Cancel)

        try:
            self.buttonBox.accepted.disconnect() 
        except TypeError:
            pass

        self.btn_ok.clicked.connect(self.confirm)
        self.btn_cancel.clicked.connect(self.reject)

        self.toggle_RSTP()
    def toggle_RSTP(self):
        text = self.BoxChoose.currentText()

        if "RTSP" in text:
            self.HBoxRTSP.setVisible(True)
        else:
            self.HBoxRTSP.setVisible(False)
    
    def confirm(self):
        if self.BoxChoose.currentText() == "RTSP":
            rtsp_link = self.RTSPInput.text().strip()
            if not rtsp_link:
                QMessageBox.warning(self, "Warning", "Please input RTSP Stream!")
                return

        self.accept()

    def get_video_source(self):
        """Hàm helper để Main App lấy kết quả"""
        if self.BoxChoose.currentText() == "Webcam":
            return 0 # Trả về int cho webcam mặc định
        else:
            return self.RTSPInput.text().strip() # Trả về string RTSP
        
class Camera:
    def __init__(self, source, skip_frame=30):
        self.root = os.getcwd()
        self.cameraThread = CameraThread()
        self.cap = None
        self.source = source
        self.frame_count = 0
        self.skip = skip_frame
        self.mainui = load_ui.loadUi("app/resources/main.ui")
        self.mainui.show()
        
        self.cameraThread.result_signal.connect(self.handle_result)
        self.mainui.StartCamera.clicked.connect(self.start_camera)
        self.mainui.StopCamera.clicked.connect(self.stop_camera)
        
        self.video_label = self.mainui.CameraDisplay
        self.video_label.setScaledContents(True)

        self.information_label = self.mainui.DisplayInformation

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.source)
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
            self.information_label.setText("No detection")
            print("No detection")
        elif data["status"] == "Successfully":
            self.information_label.setText("Scanning Successfully")
            print(data["information"])
        else:
            self.information_label.setText("Error")
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
    setup = Setup()

    result_setup = setup.exec()
    if result_setup == QDialog.DialogCode.Accepted:
        source = setup.get_video_source()
        main = Camera(source=source)
        app.exec()
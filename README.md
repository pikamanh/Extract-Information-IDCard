# Extract Information ID Card

This is a simple extraction system by using YOLO, CTPN model and VietOCR. It capture ID Card from Webcam or RTSP Link to detect, crop image to fit information area, detect bounding box of text and OCR.

# Features
- Real-time detection
- Simple GUI to using
- Display information after extracting successfully

# Requirements
- Python 3.11
- CUDA 11.8
- and others package in `requirements.txt`

# Installation
1. Clone the repository
```bash
git clone git@github.com:pikamanh/Extract-Information-IDCard.git
```
2. Create virtual environment **(Recommend)**
```bash
py -m venv venv
```
3. Active and Install packages
```bash
py .\venv\Scripts\activate
py -r requirements.txt
```

# How to use
1. Run file `main.py`
```bash
py main.py
```
2. Choose Webcam or RTSP
3. Start Camera to detect and extract
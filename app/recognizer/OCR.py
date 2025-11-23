from PIL import Image
import cv2
import numpy as np

from app.utils.pre_proccessing import ProccessingImage
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from app.detector.predict import get_text_boxes

class OCR:
    def __init__(self, model_name="vgg_transformer", device="cuda"):
        config = Cfg.load_config_from_name(model_name)
        config['device'] = device

        self.detector = Predictor(config=config)
        self.results = {}
        self.img = None
        self.rois = {
            "id_number":        None,
            "name":             None,
            "dob":              None,
            "gender":           None,
            "national":         None,
            "place_orgin":      None,
            "place_of_residence1": None,
            "place_of_residence2": None,
            "date_expired": None
        }

    def predict(self, img):
        self.img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
        text, out_img = get_text_boxes(self.img)
        corner_list = []
        
        for corners_probs in text:
            corners = [int(corner) for corner in corners_probs[:-1]]
            corner_list.append(corners)

        corner_list = np.array(corner_list)
        corner_list = corner_list[corner_list[:, 1].argsort()]

        for key, value in zip(self.rois.keys(), corner_list):
            self.rois[key] = (value[0].item(), value[1].item(), value[6].item(), value[7].item())
        
        date_expired = self.rois["place_of_residence2"]
        self.rois["place_of_residence2"] = self.rois["date_expired"]
        self.rois["date_expired"] = date_expired

        try:
            for field, (x1, y1, x2, y2) in self.rois.items():
                roi_img = self.img[y1:y2, x1:x2]
                roi_img = Image.fromarray(roi_img)
                text = self.detector.predict(roi_img)
                self.results[field] = text
            return self.results
        except Exception as e:
            return {"status": "Thông tin trích xuất hiện không đủ. Vui lòng thử lại sau."}
    
if __name__ == "__main__":
    proccessingImage = ProccessingImage()
    ocr = OCR()
    cropped_image = proccessingImage.focus_image(r"temp/test1.jpg")
    test = ocr.predict(cropped_image)

    print(test)
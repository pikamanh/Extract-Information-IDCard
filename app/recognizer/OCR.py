from PIL import Image
import cv2
import numpy as np
import re

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from app.detector.predict import get_text_boxes

class OCR:
    def __init__(self, model_name="vgg_transformer", device="cuda"):
        config = Cfg.load_config_from_name(model_name)
        config['device'] = device

        self.detector = Predictor(config=config)

    def predict(self, img):
        results = {}
        rois_img = []
        rois = {
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
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
        text, out_img = get_text_boxes(img)
        corner_list = []
        
        for corners_probs in text:
            corners = [int(corner) for corner in corners_probs[:-1]]
            if corners[1] > 200:
                corner_list.append(corners)

        corner_list = np.array(corner_list)
        corner_list = corner_list[corner_list[:, 1].argsort()]
        print(corner_list)

        for key, value in zip(rois.keys(), corner_list):
            x1 = value[0].item()
            y1 = value[1].item()
            x2 = value[6].item()
            y2 = value[7].item()

            rois[key] = (x1, y1, x2, y2)
        
        date_expired = rois["place_of_residence2"]
        rois["place_of_residence2"] = rois["date_expired"]
        rois["date_expired"] = date_expired

        try:
            for field, (x1, y1, x2, y2) in rois.items():
                roi_img = img[y1:y2, x1:x2]
                roi_img = Image.fromarray(roi_img)
                text = self.detector.predict(roi_img)

                results[field] = text

                rois_img.append(roi_img)
            
            if "/" in results["gender"]:
                dob = results["gender"]
                results["gender"] = results["dob"]
                results["dob"] = dob
            
            if len(results["dob"]) == 8:
                temp = "".join([results["dob"][:2], "/", results["dob"][2:4], "/", results["dob"][4:]])
                results["dob"] = temp

            if len(results["date_expired"]) == 8:
                temp = "".join([results["date_expired"][:2], "/", results["date_expired"][2:4], "/", results["date_expired"][4:]])
                results["date_expired"] = temp

            return results
        except Exception as e:
            return {"status": "Thông tin trích xuất hiện không đủ. Vui lòng thử lại sau."}
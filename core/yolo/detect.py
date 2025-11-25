import time
from pathlib import Path
from .models.experimental import attempt_load
from .utils.general import check_img_size, non_max_suppression, \
    scale_coords, set_logging,xyxy2xywh
from .utils.torch_utils import select_device
from .utils.datasets import letterbox
import logging
import time
import numpy as np 
import cv2
import torch
from numpy import random


class Predictor:
    def __init__(self, weights_path,device):
        self.weights  = weights_path
        self.image_size = 640
        self.device = select_device(device)
        self.confidence_threshold = 0.1
        self.iou_threshold = 0.45
        self.model_loaded = False
        self.__load_model(self.weights)
        
    def __load_model(self,weights_path:str):
        try:
            file = Path(str(weights_path).strip().replace("'", '').lower())
            if not file.exists():
                logging.critical(f"Model file not found in the path {weights_path}")
                self.model_loaded = False
                return None
            model = attempt_load(weights_path, map_location=self.device)  # load FP32 model
            self.stride = int(model.stride.max())  # model stride
            self.image_size = check_img_size(self.image_size, s=self.stride)  # check img_size
            self.half = self.device.type != 'cpu'  

            if self.half:
                model.half()  # to FP16

            # Get names and colors
            self.names = model.module.names if hasattr(model, 'module') else model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

            # Run inference
            if self.device.type != 'cpu':
                model(torch.zeros(1, 3, self.image_size, self.image_size).to(self.device).type_as(next(model.parameters())))  # run once
            self.model = model
            self.model_loaded = True
        except Exception as e:
            self.model  = None
            self.model_loaded = False
            logging.error(f"| {type(e).__name__} | {e}")
            return None
        
    def predict_by_image_path(self,image_path):
        image = cv2.imread(image_path)
        self.predict_by_image_path(image)
    def detect(self,original_image):
        result = dict()
        if self.model is None:
            raise Exception(f"Model not found in path {self.weights}")

        set_logging()

        old_img_w = old_img_h = self.image_size
        old_img_b = 1

        t0 = time.time()
        
        img = letterbox(original_image, self.image_size, stride=self.stride)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
    
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)


        if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=False)[0]
        result["preprocess_time"] = f"{time.time() - t0:.3f}s"

        # Inference
        with torch.no_grad():   
            pred = self.model(img, augment=False)[0]
        result["prediction_time"] = f"{time.time() - t0:.3f}s"

        pred = non_max_suppression(pred, self.confidence_threshold, self.iou_threshold)

        result["filter_time"] = f"{time.time() - t0:.3f}s"

        gn = torch.tensor(original_image.shape)[[1, 0, 1, 0]]  # normalization gain whwh


        detections = []
        if pred is not None and len(pred):
            for det in pred:  # Loop through detections
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_image.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                        x1, y1, x2, y2 = map(int, xyxy)
                        detections.append({"min_x":x1,"min_y": y1,"max_x": x2, "max_y": y2,"confidence": float(conf),"label": self.names[int(cls)]})
        
            

        result["detections"] = detections
        
        result["total_time"] = f"{time.time() - t0:.3f}s"
        return result

from core.yolo import Predictor
import cv2
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s - %(filename)s",
    style="%",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.DEBUG,
)
model = Predictor("core/yolo/model/default_model.pt","cpu")

if not model.model_loaded:
    exit()
image = cv2.imread("images/bus.jpg")



result = model.detect(image)

print(result)

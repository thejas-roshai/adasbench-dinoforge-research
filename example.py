from core.yolo.detect import Predictor
import cv2

model = Predictor("core/yolo/model/default_model.pt","cpu")

if not model.model_loaded:
    exit()
image = cv2.imread("images/bus.jpg")



result = model.detect(image)

print(result)

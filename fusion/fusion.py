from core.groundingdino.util.inference import load_model, load_image, predict, annotate
from core.yolo import Predictor
import cv2
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s - %(filename)s",
    style="%",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.DEBUG,
)


IMAGE_PATH = "images/bus.jpg"
TEXT_PROMPT = "chair . person . dog ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

# ================================================================================================================
#                                                 GROUNDING DINO 
# ================================================================================================================
model = load_model("core/groundingdino/config/GroundingDINO_SwinT_OGC.py", "assets/weights/gDino/groundingdino_swint_ogc.pth")

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image_gdino.jpg", annotated_frame)

# ================================================================================================================
#                                                 YOLO MODEL
# ================================================================================================================


model = Predictor("core/yolo/model/default_model.pt","cpu")

if not model.model_loaded:
    exit()
image = cv2.imread(IMAGE_PATH)

boxes, logits, phrases = model.detect(image)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image_yolo.jpg", annotated_frame)

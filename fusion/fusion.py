from core.groundingdino.util.inference import load_model, load_image, predict, annotate
from core.yolo import Predictor
import cv2
import logging
from core.Florence2.florence import florence2
from PIL import Image
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s - %(filename)s",
    style="%",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.DEBUG,
)


IMAGE_PATH = "images/image.png"
TEXT_PROMPT = "hills, road, highway, guard rails, mountains, sky, bridge, tunnel, car, truck, bus, motorcycle, bicycle, emergency vehicles, parked vehicles, trailers, pedestrian, construction workers, cyclists, animals, lanes, lane markings, road edges, curbs, shoulders, intersections, crosswalk, roundabout, traffic islands, parking lots, toll booth, railroad crossing, traffic lights, traffic signs, speed limit signs, warning signs, stop signs, yield signs, signboards, cones, barrels, barriers, median, road work signs, trees, bushes, vegetation, water bodies, snow, ice, fog, rain, night, shadows, sun glare, buildings, walls, fences, sidewalks, poles, street lamps, billboards, potholes, construction debris, rocks, fallen branches, spilled cargo"


BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

# ================================================================================================================
#                                                 GROUNDING DINO 
# ================================================================================================================
model = load_model("core/groundingdino/config/GroundingDINO_SwinT_OGC.py", "assets/weights/dino/groundingdino_swint_ogc.pth")

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
print(logits)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image_yolo.jpg", annotated_frame)


# ================================================================================================================
#                                                 FLORENCE 2
# ================================================================================================================


image = Image.open(IMAGE_PATH).convert("RGB")

boxes, logits, phrases = florence2(image)
print(boxes)
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image_florence.jpg", annotated_frame)

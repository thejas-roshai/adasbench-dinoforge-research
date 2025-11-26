import time
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from torchvision.ops import box_convert
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "microsoft/Florence-2-large"

#####################################
# Measure model loading time
#####################################
# t0 = time.time()

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    trust_remote_code=True
).to(device)

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True
)

# t1 = time.time()
# print(f"Model + processor load time: {t1 - t0:.3f} seconds")
#####################################


###############################################################################################################

def florence2(image):
    prompt = "hills, road, highway, guard rails, mountains, sky, bridge, tunnel, car, truck, bus, motorcycle, bicycle, emergency vehicles, parked vehicles, trailers, pedestrian, construction workers, cyclists, animals, lanes, lane markings, road edges, curbs, shoulders, intersections, crosswalk, roundabout, traffic islands, parking lots, toll booth, railroad crossing, traffic lights, traffic signs, speed limit signs, warning signs, stop signs, yield signs, signboards, cones, barrels, barriers, median, road work signs, trees, bushes, vegetation, water bodies, snow, ice, fog, rain, night, shadows, sun glare, buildings, walls, fences, sidewalks, poles, street lamps, billboards, potholes, construction debris, rocks, fallen branches, spilled cargo"
    task="<CAPTION_TO_PHRASE_GROUNDING>"
    task="<OD>"
    prompt = ""
    prompt = task + prompt 

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(device)

    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch_dtype)

    # t_inf0 = time.time()

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    # t_inf1 = time.time()
    # print(f"Inference time: {t_inf1 - t_inf0:.3f} seconds")
    #####################################

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False
    )[0]

    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task,
        image_size=(image.width, image.height)
    )
    parsed_answer = parsed_answer[task]
    bboxes = torch.tensor(parsed_answer['bboxes'], dtype=torch.float32)
    
    h, w = image.height, image.width
    # Normalize by image width and height
    bboxes[:, [0, 2]] /= w
    bboxes[:, [1, 3]] /= h
    boxes = box_convert(bboxes, in_fmt="xyxy", out_fmt="cxcywh")
    logits = torch.ones(len(boxes), dtype=torch.float32)  # or provide real scores if available
    # 3. Labels as phrases
    phrases = parsed_answer['labels']
    return boxes,logits,phrases

###############################################################################################################

# image = Image.open('/home/roshai/Documents/images.jpeg')

# task="<CAPTION_TO_PHRASE_GROUNDING>"



# if task == "<OD>":
#     results = florence2(task, image)
# else:
#     results = florence2(task, image, text_input=prompt)

# print(results[task])

###############################################################################################################

# def plot_bbox(image, data):
#     fig, ax = plt.subplots()

#     ax.imshow(image)

#     for bbox, label in zip(data['bboxes'], data['labels']):
#         x1, y1, x2, y2 = bbox

#         rect = patches.Rectangle(
#             (x1, y1),
#             x2 - x1,
#             y2 - y1,
#             linewidth=1,
#             edgecolor='r',
#             facecolor='none'
#         )
#         ax.add_patch(rect)

#         plt.text(
#             x1, y1, label,
#             color='white',
#             fontsize=8,
#             bbox=dict(facecolor='red', alpha=0.5)
#         )

#     ax.axis('off')
#     plt.show()

# plot_bbox(image, results[task])

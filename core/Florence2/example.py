from Florence2.florence import florence2
from PIL import Image
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

image = Image.open('testimages/data/animal1.jpg')

task_prompt3 = '<OD>'
od_results = florence2(task_prompt3, image)  
print(od_results['<OD>'])

# {'bboxes': [[118.20000457763672,
#    184.6304931640625,
#    666.6000366210938,
#    717.2954711914062],
#   [826.2000122070312,
#    360.04949951171875,
#    955.800048828125,
#    678.8474731445312]],
#  'labels': ['elephant', 'person']}


def plot_bbox(image, data):
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for bbox, label in zip(data['bboxes'], data['labels']):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Annotate the label
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    # Remove the axis ticks and labels
    ax.axis('off')

    # Show the plot
    plt.show()

plot_bbox(image, od_results['<OD>'])
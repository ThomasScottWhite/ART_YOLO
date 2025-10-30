import pickle
import fuzzy_art
from onnx_inferencing import OnnxInferencer
from PIL import Image

with open('/home/thomas/Documents/ART_YOLO/inferencing/models/fuzzy_art.pkl', 'rb') as f:
    model = pickle.load(f)

DFINE_Path = "/home/thomas/Documents/ART_YOLO/inferencing/models/best_stg1_with_tap.onnx"
inferencer = OnnxInferencer(DFINE_Path)
image = Image.open("/home/thomas/Pictures/image.jpg").convert("RGB")
model_outputs, tap_outputs = inferencer.inference(image)
predictions = model.predict(tap_outputs)

from PIL import Image, ImageDraw, ImageFont
import numpy as np

def draw_bounding_boxes(image, model_outputs, confidence_threshold=0.5, class_labels=None):
    scores = np.squeeze(model_outputs[0])
    boxes = np.squeeze(model_outputs[1])

    img_width, img_height = image.size
    draw = ImageDraw.Draw(image)

    for i in range(len(scores)):
        max_score = np.max(scores[i])

        if max_score > confidence_threshold:
            box = boxes[i]
            center_x, center_y, width, height = box
            
            x_min = center_x - (width / 2)
            y_min = center_y - (height / 2)
            x_max = center_x + (width / 2)
            y_max = center_y + (height / 2)

            abs_x_min = x_min * img_width
            abs_y_min = y_min * img_height
            abs_x_max = x_max * img_width
            abs_y_max = y_max * img_height

            label_index = np.argmax(scores[i])
            label_name = class_labels[label_index] if class_labels else f"Class {label_index}"
            display_text = f"{label_name}: {max_score:.2f}"

            draw.rectangle([(abs_x_min, abs_y_min), (abs_x_max, abs_y_max)], outline="red", width=3)
            
            text_bbox = draw.textbbox((abs_x_min, abs_y_min), display_text)
            text_bbox = (text_bbox[0], text_bbox[1] - (text_bbox[3]-text_bbox[1]), text_bbox[2], text_bbox[1])
            draw.rectangle(text_bbox, fill="red")
            
            draw.text((text_bbox[0], text_bbox[1]), display_text, fill="white")

    return image

image_with_boxes = draw_bounding_boxes(
    image.copy(),
    model_outputs,
    confidence_threshold=0.8,
)

output_filename = 'detection_output.png'
print(f"Image with bounding boxes saved as '{output_filename}'")

image_with_boxes.show()

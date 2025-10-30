import onnxruntime as ort
import cv2
import numpy as np

class OnnxInferencer:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)


    def load_image_from_pil(self, pil_img, size: int = 640) -> tuple[np.ndarray, np.ndarray]:
        # Convert PIL image to NumPy array (RGB by default)
        img = np.array(pil_img)

        # Store original size before resizing
        orig_h, orig_w = img.shape[:2]
        orig_target_sizes = np.array([[orig_h, orig_w]], dtype=np.int64)

        # Resize to YOLOv8 input size (640x640)
        img = cv2.resize(img, (size, size))

        # Ensure image is RGB (in case it's RGBA or grayscale)
        if img.ndim == 2:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Convert to CHW (channel, height, width) and normalize
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0

        # Add batch dimension
        img = np.expand_dims(img, axis=0)  # shape: (1, 3, 640, 640)

        return img, orig_target_sizes

    def inference(self, image: str, strategy: str = "mean", keep=False) -> dict:
        img, orig_target_sizes = self.load_image_from_pil(image)
                
        # Load ONNX model
        # Get input names
        inputs = self.session.get_inputs()
        input_names = [input.name for input in inputs]

        # Assuming first is 'images', second is 'orig_target_sizes'
        input_feed = {
            input_names[0]: img,
        }

        # Run inference
        output_names = [output.name for output in self.session.get_outputs()]
        outputs = self.session.run(output_names, input_feed)

        model_outputs = outputs[0:2]
        tap_outputs = outputs[2][0].mean(axis=(1, 2))

        return model_outputs, tap_outputs
    

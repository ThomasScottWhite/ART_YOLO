import pickle
import fuzzy_art
from onnx_inferencing import OnnxInferencer
from PIL import Image
import time
import numpy as np

with open('/home/thomas/Documents/ART_YOLO/inferencing/models/fuzzy_art.pkl', 'rb') as f:
    model = pickle.load(f)
DFINE_Path = "/home/thomas/Documents/ART_YOLO/inferencing/models/best_stg1_with_tap.onnx"
inferencer = OnnxInferencer(DFINE_Path)
image = Image.open("/home/thomas/Pictures/image.jpg").convert("RGB")

print("Performing warm-up run...")
_, tap_outputs_warmup = inferencer.inference(image)
_ = model.predict(tap_outputs_warmup)
print("Warm-up complete.")

num_runs = 50
onnx_times = []
art_times = []

print(f"\nRunning benchmark for {num_runs} iterations...")
for _ in range(num_runs):
    start_onnx = time.perf_counter()
    model_outputs, tap_outputs = inferencer.inference(image)
    end_onnx = time.perf_counter()
    onnx_times.append((end_onnx - start_onnx) * 1000)
    
    start_art = time.perf_counter()
    _ = model.predict(tap_outputs)
    end_art = time.perf_counter()
    art_times.append((end_art - start_art) * 1000)

avg_onnx_ms = np.mean(onnx_times)
avg_art_ms = np.mean(art_times)
avg_total_ms = avg_onnx_ms + avg_art_ms

print("\n--- Average Inference Performance ---")
print(f"ONNX Model Time: {avg_onnx_ms:.2f} ms")
print(f"Fuzzy ART Time:  {avg_art_ms:.2f} ms")
print("-----------------------------------")
print(f"Total Inference Time: {avg_total_ms:.2f} ms")
print(f"(Based on {num_runs} runs)")
print("-----------------------------------")

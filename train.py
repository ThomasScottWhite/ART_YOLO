from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from torch.utils.data import ConcatDataset
from pathlib import Path
import numpy as np
import subprocess, shutil, pickle, time, sys
from tqdm import tqdm
import onnx
from onnx import helper
import pandas as pd
from save_dataset import save_dataset_to_file
from onnx_inferencing import OnnxInferencer
from fuzzy_art import OnlineFuzzyART

# =============================
# Config
# =============================
experment_name = "threshold_0.30_rho_0.5"
experment_dir = Path(f"experiments/{experment_name}")
experment_dir.mkdir(parents=True, exist_ok=True)

LOGS_DIR = experment_dir / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH = experment_dir / "state.pkl"

EXE = "/home/thomas/miniforge3/envs/dfine/bin/python"
DFINE_CWD = "/home/thomas/Documents/ART_YOLO/D-FINE"
DFINE_CFG = "configs/dfine/dfine_hgnetv2_n_coco.yml"
OUTPUT_DIR = Path("/home/thomas/Documents/ART_YOLO/D-FINE/output/dfine_hgnetv2_n_coco")

rho = 0.5
threshold = 0.30
subdivision_amount = 10

# =============================
# State (pickle only)
# =============================
def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH, "rb") as f:
            return pickle.load(f)
    return {"iterations": {}}

def save_state(state):
    with open(STATE_PATH, "wb") as f:
        pickle.dump(state, f)

def is_done(state, iteration, step_name):
    return state["iterations"].get(iteration, {}).get(step_name, False)

def mark_done(state, iteration, step_name):
    state["iterations"].setdefault(iteration, {})
    state["iterations"][iteration][step_name] = True
    save_state(state)

# =============================
# Dataset loading / splitting
# =============================
def load_and_split_dataset(subdivision_amount):
    ds = load_dataset("detection-datasets/coco", split="train")
    splits = ds.train_test_split(test_size=0.9)
    train_subsets = [splits['train']]
    remaining = splits['test']
    for i in range(8):
        splits = remaining.train_test_split(test_size=(8 - i) / (9 - i))
        train_subsets.append(splits['train'])
        remaining = splits['test']
    train_subsets.append(remaining)

    ds = load_dataset("detection-datasets/coco", split="val")
    splits = ds.train_test_split(test_size=0.9)
    test_subsets = [splits['train']]
    remaining = splits['test']
    for i in range(8):
        splits = remaining.train_test_split(test_size=(8 - i) / (9 - i))
        test_subsets.append(splits['train'])
        remaining = splits['test']
    test_subsets.append(remaining)

    return train_subsets, test_subsets

# =============================
# Utilities (no try/except)
# =============================
def run_cmd(cmd, cwd, log_path: Path, timeout=None, env=None):
    with open(log_path, "a") as lf:
        lf.write(f"\n=== RUN ===\nCMD: {' '.join(cmd)}\n")
    proc = subprocess.run(
        cmd, cwd=cwd, check=True, capture_output=True, text=True, timeout=timeout, env=env
    )
    with open(log_path, "a") as lf:
        lf.write(proc.stdout or "")
        lf.write("\n--- STDERR ---\n")
        lf.write(proc.stderr or "")
        lf.write("\n=== OK ===\n")

# =============================
# Step functions (no artifact checks, no try/except)
# =============================
def train_model(train_set, test_set, out_dir: Path, log_file: Path):
    save_dataset_to_file(train_set, test_set)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    cmd = [EXE, "train.py", "-c", DFINE_CFG, "--use-amp", "--seed=0"]
    run_cmd(cmd, DFINE_CWD, log_file)

    # Copy whatever D-FINE wrote to the run output dir
    shutil.copytree(OUTPUT_DIR, out_dir, dirs_exist_ok=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def export_onnx(checkpoint_pth: Path, out_dir: Path, log_file: Path):
    cmd = [EXE, "tools/deployment/export_onnx.py", "-c", DFINE_CFG, "-r", str(checkpoint_pth.resolve())]
    run_cmd(cmd, DFINE_CWD, log_file)

def step_tap_onnx(out_dir: Path, log_file: Path):
    onnx_in = out_dir / "best_stg1.onnx"
    onnx_out = out_dir / "best_stg1_with_tap.onnx"
    model = onnx.load(str(onnx_in))
    model.graph.output.append(helper.ValueInfoProto(name="/model/backbone/stages.3/blocks/blocks.0/Concat_output_0"))
    onnx.save(model, str(onnx_out))

def train_art(art_train_set, tapped_onnx: Path, out_dir: Path, log_file: Path, rho=0.5):
    print("Loading ONNX model and training ART")
    model_pkl = out_dir / "fuzzy_art.pkl"
    inferencer = OnnxInferencer(model_path=str(tapped_onnx))
    fuzzy_art = OnlineFuzzyART(input_dim=896, vigilance=rho)
    
    # Training ART
    for item in tqdm(art_train_set, desc="Training Fuzzy ART"):
        _, feat = inferencer.inference(item["image"])
        fuzzy_art.learn(feat)
        
    with open(model_pkl, "wb") as f:
        pickle.dump(fuzzy_art, f)

def filter_next_dataset(next_dataset, out_dir_base: Path, control_base: Path, tapped_onnx: Path, art_model_pkl: Path, log_file: Path, threshold=0.25):
    with open(art_model_pkl, "rb") as f:
        fuzzy_art : OnlineFuzzyART = pickle.load(f)

    inferencer = OnnxInferencer(model_path=str(tapped_onnx))
    keep = []
    art_outputs = []
    for i in range(len(next_dataset)):
        _, feat = inferencer.inference(next_dataset[i]["image"])
        cat, match = fuzzy_art.predict(feat)
        art_outputs.append((cat, match, fuzzy_art.num_categories()))
        if match >= threshold:
            keep.append(i)
    filtered = next_dataset.select(keep)
    filtered.save_to_disk(out_dir_base)

    # Save csv of art_outputs
    df = pd.DataFrame(art_outputs, columns=["category", "match_score", "total_categories"])
    df.to_csv(out_dir_base / "art_outputs.csv", index=False)


    np.random.seed(42)
    rand = np.random.choice(len(next_dataset), size=len(filtered), replace=False) if len(filtered) else []
    control = next_dataset.select(sorted(rand.tolist()))
    control.save_to_disk(control_base)

    # Save dataset filtering to same directory as dataset
    with open(out_dir_base / "filtering.txt", "w") as f:
        f.write(f"Total images in original dataset: {len(next_dataset)}\n")
        f.write(f"Images kept in ART dataset: {len(filtered)}\n")
        f.write(f"Images kept in Control dataset: {len(control)}\n")

# =============================
# One-time dataset materialization
# =============================
if subdivision_amount == 1:
    train_subsets = [load_dataset("detection-datasets/coco", split="train")]
    test_subsets = [load_dataset("detection-datasets/coco", split="val")]
else:
    # This needs to be updated to allow for subdivision_amount != 10
    train_subsets, test_subsets = load_and_split_dataset(subdivision_amount)

# Save splits to disk once
for i, ds in enumerate(train_subsets):
    ds.save_to_disk(experment_dir / f"dataset/split_full_dataset/train_split/{i}")
for i, ds in enumerate(test_subsets):
    ds.save_to_disk(experment_dir / f"dataset/split_full_dataset/test_split/{i}")

# Ensure per-iteration dirs
for i in range(subdivision_amount):
    (experment_dir / f"iteration_{i}/art_output").mkdir(parents=True, exist_ok=True)
    (experment_dir / f"iteration_{i}/control_output").mkdir(parents=True, exist_ok=True)

# Load persisted state
state = load_state()

# Reload persisted datasets
full_train_datasets = [load_from_disk(experment_dir / f"dataset/split_full_dataset/train_split/{i}") for i in range(subdivision_amount)]
full_test_dataset = [load_from_disk(experment_dir / f"dataset/split_full_dataset/test_split/{i}") for i in range(subdivision_amount)]

# =============================
# Main loop
# =============================

for iteration in range(subdivision_amount):
    print(f"\n=== ITERATION {iteration} ===")
    iter_dir = experment_dir / f"iteration_{iteration}"
    art_dir = iter_dir / "art_output"
    ctl_dir = iter_dir / "control_output"
    log_file = LOGS_DIR / f"iter_{iteration}.log"

    # Build datasets for this iteration
    if iteration == 0:
        art_datasets = [full_train_datasets[0]]
        control_datasets = [full_train_datasets[0]]
    else:
        art_datasets = [full_train_datasets[0]]
        control_datasets = [full_train_datasets[0]]
        for j in range(1, iteration):
            saved_art = experment_dir / f"dataset/art_dataset/{j}"
            saved_ctl = experment_dir / f"dataset/control_split/{j}"

            if saved_art.exists():
                art_datasets.append(load_from_disk(saved_art))
            else:
                raise ValueError(f"Expected ART dataset not found: {saved_art}")
            
            if saved_ctl.exists():
                control_datasets.append(load_from_disk(saved_ctl))
            else:
                raise ValueError(f"Expected Control dataset not found: {saved_ctl}")

    art_train_set = ConcatDataset(art_datasets)
    control_train_set = ConcatDataset(control_datasets)
    test_set = ConcatDataset(full_test_dataset[:iteration + 1])

    # Train YOLO model for ART
    if not is_done(state, iteration, "train_art"):
        print("Training YOLO model for ART")

        train_model(art_train_set, test_set, art_dir, log_file)
        mark_done(state, iteration, "train_art")

    # Train Control
    if subdivision_amount == 1:
        print("Skipping Control Training as Subdivision count = 1")
    elif not is_done(state, iteration, "train_control"):
        print("Training Control")
        train_model(control_train_set, test_set, ctl_dir, log_file)
        mark_done(state, iteration, "train_control")

    # Export ONNX
    if not is_done(state, iteration, "export_onnx"):
        print("Exporting ONNX")
        art_pth = art_dir / "best_stg1.pth"
        export_onnx(art_pth, art_dir, log_file)
        mark_done(state, iteration, "export_onnx")

    # Tap ONNX
    if not is_done(state, iteration, "tap_onnx"):
        print("Tapping ONNX")
        step_tap_onnx(art_dir, log_file)
        mark_done(state, iteration, "tap_onnx")

    # Train ART model
    if not is_done(state, iteration, "learn_art"):
        print("Learning ART model")
        tapped_path = art_dir / "best_stg1_with_tap.onnx"
        train_art(art_train_set, tapped_path, art_dir, log_file, rho=rho)
        mark_done(state, iteration, "learn_art")

    # Filter the next dataset 
    # the ART dataset is based on what the art module belives is novel
    # the control dataset the same size as art but randomly picked
    if subdivision_amount == 1:
        print("Skipping Dataset Filtering as subdivision_amount = 1")
    elif iteration < 9 and not is_done(state, iteration, "filter_next"):
        print("Filtering next dataset")
        tapped_path = art_dir / "best_stg1_with_tap.onnx"
        art_model_pkl = art_dir / "fuzzy_art.pkl"
        art_out_next = experment_dir / f"dataset/art_dataset/{iteration + 1}"
        ctl_out_next = experment_dir / f"dataset/control_split/{iteration + 1}"
        art_out_next.parent.mkdir(parents=True, exist_ok=True)
        ctl_out_next.parent.mkdir(parents=True, exist_ok=True)

        filter_next_dataset(
            full_train_datasets[iteration + 1],
            art_out_next,
            ctl_out_next,
            tapped_path,
            art_model_pkl,
            log_file,
            threshold=threshold
        )
        mark_done(state, iteration, "filter_next")

    # Plain summary (informational)
    (iter_dir / "summary.txt").write_text(
        f"iteration={iteration}\n"
        f"train_art_done={is_done(state, iteration, 'train_art')}\n"
        f"train_control_done={is_done(state, iteration, 'train_control')}\n"
        f"export_onnx_done={is_done(state, iteration, 'export_onnx')}\n"
        f"tap_onnx_done={is_done(state, iteration, 'tap_onnx')}\n"
        f"learn_art_done={is_done(state, iteration, 'learn_art')}\n"
        f"filter_next_done={is_done(state, iteration, 'filter_next')}\n"
    )

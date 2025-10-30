"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""
import os
import sys
import torch
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from src.core import YAMLConfig


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        # Skip postprocessor to save memory
        # self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        # If needed: outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs


def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    # Load checkpoint directly to CPU
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        state = checkpoint.get("ema", checkpoint.get("model"))
        if isinstance(state, dict) and "module" in state:
            state = state["module"]
        cfg.model.load_state_dict(state)

    model = Model(cfg).to("cpu").eval()

    # Use batch=1 dummy input
    data = torch.rand(1, 3, 640, 640)
    size = torch.tensor([[640, 640]])

    output_file = args.resume.replace(".pth", ".onnx") if args.resume else "model.onnx"

    with torch.no_grad():
        trial = model(data, size)
        if isinstance(trial, (list, tuple)):
            print("FORWARD RETURNS", len(trial), [t.shape if hasattr(t, "shape") else type(t) for t in trial])
        elif isinstance(trial, dict):
            print("FORWARD KEYS", list(trial.keys()))
        else:
            print("FORWARD TYPE", type(trial))

    torch.onnx.export(
        model,
        (data, size),
        output_file,
        input_names=["images", "orig_target_sizes"],
        output_names=["pred_logits", "pred_boxes"],  # <-- 2 outputs
        dynamic_axes={
            "images": {0: "N"},
            "orig_target_sizes": {0: "N"},
            "pred_logits": {0: "N", 1: "num_queries"},
            "pred_boxes":  {0: "N", 1: "num_queries"},
        },
        opset_version=16,
        do_constant_folding=True,
    )


    print(f"ONNX export done: {output_file}")

    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str)
    parser.add_argument("--check", action="store_true", default=True)
    args = parser.parse_args()
    main(args)
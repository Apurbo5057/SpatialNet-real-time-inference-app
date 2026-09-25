"""Export the on-device object classifier used inside the ROI.

Produces, in ObjectModel/:
  mobilenetv4_conv_medium_fp16w.onnx  input "image" [1,3,S,S] float32, output "logits" [1,1000] float32
                                      and "embedding" [1,960] float32 (pooled features for My Objects).
                                      Weights are stored as FLOAT16 and cast to float32 at load,
                                      the same layout as ROIModel/spatialnet_fastest_from_fp16.onnx.
  mobilenetv4_labels.txt              1000 lines, one friendly label per ImageNet class index.
                                      Classes that share a label are summed by the app.

Run from the repository root:
  pip install -r tools/classifier/requirements.txt
  python tools/classifier/export_classifier.py
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
from onnx import helper, numpy_helper, TensorProto
from PIL import Image
from timm.data import ImageNetInfo

from add_embedding_output import add_embedding_output

MODEL_NAME = "mobilenetv4_conv_medium.e500_r224_in1k"
OUT_DIR = "ObjectModel"
ONNX_NAME = "mobilenetv4_conv_medium_fp16w.onnx"
LABELS_NAME = "mobilenetv4_labels.txt"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ImageNet index -> friendly label. Classes sharing a label are merged (probabilities summed).
# Unlisted classes use the first synonym of their ImageNet description.
FRIENDLY = {
    "TV / monitor": [851, 664, 782, 598, 548, 527],
    "Laptop": [620, 681, 590],
    "Mobile phone": [487, 528, 707],
    "Remote control": [761],
    "Computer keyboard": [508, 878],
    "Computer mouse": [673],
    "Electric switch": [844],
    "Lamp": [846, 619, 818],
    "Clock": [409, 530, 892],
    "Bottle": [440, 737, 898, 907, 720],
    "Cup / mug": [504, 968, 647],
    "Chair": [423, 559, 765],
    "Sofa": [831],
    "Table": [532, 526],
    "Cabinet": [495, 553, 648, 894],
    "Fan": [545],
    "Heater / radiator": [811, 753],
    "Window / curtain": [904, 905, 794, 854],
    "Door": [799],
    "Refrigerator": [760],
    "Microwave": [651],
    "Washing machine": [897],
    "Speaker": [632],
    "Printer": [742],
    "Book": [917, 921],
    "Pen": [418, 563, 749],
    "Trash can": [412],
    "Bag": [414, 728, 636],
    "Plate": [923],
    "Bowl": [659, 809],
    "Pot / pan": [567, 544, 521, 505, 849],
    "Piano": [579, 881],
    "Guitar": [402, 546],
    "Camera": [732, 759],
    "Vase / flowerpot": [883, 738],
    "Bathtub / sink": [435, 896, 876],
    "Pillow": [721],
    "Wall": [825],
}


def friendly_labels():
    info = ImageNetInfo()
    labels = [info.index_to_description(i).split(",")[0].strip() for i in range(1000)]
    labels = [l[:1].upper() + l[1:] for l in labels]
    seen = set()
    for name, idxs in FRIENDLY.items():
        for i in idxs:
            assert i not in seen, f"class {i} mapped twice"
            seen.add(i)
            labels[i] = name
    return labels


def store_weights_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    """Store float32 initializers as FLOAT16 with a Cast back to float32 before use."""
    graph = model.graph
    casts, keep = [], []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT or np.prod(init.dims) < 16:
            keep.append(init)
            continue
        arr = numpy_helper.to_array(init)
        stored = numpy_helper.from_array(arr.astype(np.float16), init.name + "__stored_fp16")
        keep.append(stored)
        casts.append(helper.make_node("Cast", [stored.name], [init.name], to=TensorProto.FLOAT,
                                      name=init.name + "__cast"))
    del graph.initializer[:]
    graph.initializer.extend(keep)
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    return model


def preprocess(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--samples", nargs="*", default=["ROI_issues/test.jpg"])
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    model = timm.create_model(MODEL_NAME, pretrained=True).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    assert np.allclose(cfg["mean"], MEAN) and np.allclose(cfg["std"], STD), cfg

    fp32_path = os.path.join(OUT_DIR, "_fp32_tmp.onnx")
    dummy = torch.randn(1, 3, args.size, args.size)
    torch.onnx.export(model, dummy, fp32_path, input_names=["image"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    fp32 = onnx.load(fp32_path)
    onnx.checker.check_model(fp32)

    out_path = os.path.join(OUT_DIR, ONNX_NAME)
    onnx.save(add_embedding_output(store_weights_fp16(fp32)), out_path)
    onnx.checker.check_model(onnx.load(out_path))

    labels = friendly_labels()
    with open(os.path.join(OUT_DIR, LABELS_NAME), "w") as f:
        f.write("\n".join(labels) + "\n")

    # Parity: PyTorch vs ORT (fp32) vs ORT (fp16-stored weights)
    s32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    s16 = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    inputs = [preprocess(p, args.size) for p in args.samples if os.path.exists(p)]
    inputs.append(np.random.default_rng(0).standard_normal((1, 3, args.size, args.size)).astype(np.float32))
    for x in inputs:
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        o32 = s32.run(None, {"image": x})[0]
        o16 = s16.run(["logits"], {"image": x})[0]
        print(f"max|torch-ort32|={np.abs(ref - o32).max():.2e}  max|torch-ort16w|={np.abs(ref - o16).max():.2e}  "
              f"top1 torch={ref.argmax()} ort16w={o16.argmax()} ({labels[o16.argmax()]})")
    os.remove(fp32_path)

    data = open(out_path, "rb").read()
    print(f"{out_path}: {len(data) / 1024 / 1024:.2f} MB, badge digest {hashlib.sha256(data).hexdigest()[:8]}")


if __name__ == "__main__":
    sys.exit(main())

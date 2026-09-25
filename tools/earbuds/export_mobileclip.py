"""Export Apple's MobileCLIP2-S0 image encoder for the phone (earbuds recognition).

Produces ObjectModel/mobileclip2_s0_image_fp16w.onnx:
  input  "image"     [1,3,256,256] float32, RGB scaled to 0..1 (no mean/std normalization)
  output "embedding" [1,512]       float32, not normalized (the app L2-normalizes it)
Weights are stored as FLOAT16 and cast to float32 at load, like the other bundled models.
Only the image encoder ships; the text encoder is not needed for earbuds recognition.

On the Kaggle benchmark (10 earbud models, unseen photos), a linear classifier on these
features scored 69.5% top-1 and 88.8% top-3, the best of the phone-sized candidates.

Weights: apple/MobileCLIP2-S0 via open_clip ("dfndr2b"), Apple ML Research licence
(apple-amlr); used here in a private educational project.

Run from the repository root:
  pip install torch timm open_clip_torch onnx onnxruntime pillow
  python tools/earbuds/export_mobileclip.py
"""

import glob
import hashlib
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import open_clip
import timm
import torch
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

OUT = "ObjectModel/mobileclip2_s0_image_fp16w.onnx"
SIZE = 256


class ImageEncoder(torch.nn.Module):
    """Image tower plus projection: the same vector open_clip's encode_image returns."""

    def __init__(self, clip):
        super().__init__()
        self.visual = clip.visual

    def forward(self, image):
        return self.visual(image)


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


def preprocess(img: Image.Image) -> np.ndarray:
    """What the app will do: square crop resized straight to 256, RGB / 255."""
    x = np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255.0
    return x.transpose(2, 0, 1)[None]


def main():
    clip, _, _ = open_clip.create_model_and_transforms("MobileCLIP2-S0", pretrained="dfndr2b")
    clip.eval()
    # Merge FastViT's training-time branches into single convolutions (same outputs, faster).
    clip.visual = timm.utils.reparameterize_model(clip.visual)
    enc = ImageEncoder(clip).eval()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".fp32.tmp"
    with torch.no_grad():
        torch.onnx.export(enc, torch.rand(1, 3, SIZE, SIZE), tmp, input_names=["image"],
                          output_names=["embedding"], opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(tmp))
    onnx.save(store_weights_fp16(onnx.load(tmp)), OUT)
    onnx.checker.check_model(onnx.load(OUT))

    # Parity: PyTorch vs ONNX (fp32) vs ONNX (fp16-stored weights), as cosine similarity.
    s32 = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    s16 = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    samples = [preprocess(Image.open(p)) for p in sorted(glob.glob("ROI_issues/*.jpg"))[:3]]
    samples.append(np.random.default_rng(0).random((1, 3, SIZE, SIZE), dtype=np.float32))
    cos = lambda a, b: float((a @ b.T).item() / (np.linalg.norm(a) * np.linalg.norm(b)))
    for x in samples:
        with torch.no_grad():
            ref = enc(torch.from_numpy(x)).numpy()
        o32 = s32.run(None, {"image": x})[0]
        o16 = s16.run(None, {"image": x})[0]
        print(f"cosine torch-vs-onnx32 {cos(ref, o32):.6f}   torch-vs-onnx16w {cos(ref, o16):.6f}")
    os.remove(tmp)

    data = open(OUT, "rb").read()
    print(f"{OUT}: {len(data) / 1024 / 1024:.2f} MB, output {s16.get_outputs()[0].shape}, "
          f"badge digest {hashlib.sha256(data).hexdigest()[:8]}")


if __name__ == "__main__":
    sys.exit(main())

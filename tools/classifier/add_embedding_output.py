"""Expose the classifier's pooled backbone features as a second output, "embedding" [1,960].

The app's "My Objects" mode compares these vectors (cosine similarity) to recognise
objects the user has taught it. The logits output is unchanged, so one model run
serves both Identify and My Objects.

Pooled features separate objects better than the 1280-d head output: on jittered
crops of the ROI_issues captures, same-object similarity averaged 0.90 for both,
while different-object similarity averaged 0.46 (pooled) versus 0.58 (head).

Run from the repository root (needs only onnx and numpy, no PyTorch):
  python tools/classifier/add_embedding_output.py
"""

import hashlib
import sys

import onnx
from onnx import helper, TensorProto

MODEL = "ObjectModel/mobilenetv4_conv_medium_fp16w.onnx"
POOL = "/global_pool/pool/GlobalAveragePool_output_0"


def add_embedding_output(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    if any(o.name == "embedding" for o in graph.output):
        return model
    assert any(POOL in n.output for n in graph.node), f"{POOL} not found"
    graph.node.append(helper.make_node("Flatten", [POOL], ["embedding"], axis=1, name="embedding_flatten"))
    graph.output.append(helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 960]))
    return model


def main():
    model = add_embedding_output(onnx.load(MODEL))
    onnx.checker.check_model(model)
    onnx.save(model, MODEL)
    data = open(MODEL, "rb").read()
    print(f"{MODEL}: {len(data) / 1024 / 1024:.2f} MB, badge digest {hashlib.sha256(data).hexdigest()[:8]}")


if __name__ == "__main__":
    sys.exit(main())

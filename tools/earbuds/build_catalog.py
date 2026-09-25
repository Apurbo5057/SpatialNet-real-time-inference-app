"""Build the earbuds catalog the app loads, and measure how well it recognises each model.

Inputs:
  tools/earbuds/catalog_draft/<id>.json    specs, Bluetooth names, pairing steps, image list
  tools/earbuds/web_images/<id>/*.jpg       reference photos (git-ignored, private)
  tools/earbuds/removed/web_images/*/*.jpg  earbuds NOT in the catalog, used to test rejection
  ObjectModel/mobileclip2_s0_image_fp16w.onnx  (tools/earbuds/export_mobileclip.py)

Outputs in tools/earbuds/out/ (git-ignored: thumbnails come from web photos):
  earbuds_catalog.json     entries in class order (no image sources) + thresholds + model info
  earbuds_classifier.bin   linear classifier on L2-normalized embeddings, little-endian:
                           int32 classes, int32 dim, float32 W[classes][dim], float32 b[classes]
  earbuds_embeddings.bin   reference vectors for the "is this a supported model?" check:
                           int32 count, int32 dim, then per vector int32 class + dim float32
  thumbs/<id>.jpg          256 px thumbnail for the spec card

Recognition on the phone: embed the square ROI crop with MobileCLIP2-S0 (256 px, RGB / 255),
L2-normalize, then
  1. "supported?"  highest cosine similarity to any reference vector >= minSimilarity
  2. "which one?"  softmax(W @ e + b), top 3 candidates go to the Bluetooth search.

Evaluation holds out whole photos (5 folds): camera-like augmented views of unseen photos are
the queries. The rejection threshold keeps 95% of supported earbuds and is reported against
unsupported earbuds (removed/) and non-earbud scenes (ROI_issues/).

Run from the repository root:
  python3 tools/earbuds/build_catalog.py
"""

import glob
import hashlib
import json
import os
import random
import struct
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = "tools/earbuds"
DRAFTS = f"{ROOT}/catalog_draft"
IMAGES = f"{ROOT}/web_images"
UNSUPPORTED = f"{ROOT}/removed/web_images/*/*.jpg"
SCENES = "ROI_issues/*.jpg"
OUT = f"{ROOT}/out"
MODEL = "ObjectModel/mobileclip2_s0_image_fp16w.onnx"

SIZE = 256
AUG_EVAL = 8      # augmented views per photo for training/evaluating the classifier
AUG_REF = 4       # augmented views per photo kept as reference vectors in the app
C = 3.0           # logistic-regression regularization (chosen on the Kaggle benchmark)
KEEP_SUPPORTED = 0.95


def square(img: Image.Image) -> Image.Image:
    """Pad to a square with the border colour, like the app's square ROI crop (no stretching)."""
    w, h = img.size
    side = max(w, h)
    border = img.resize((1, 1), Image.BILINEAR).getpixel((0, 0))
    canvas = Image.new("RGB", (side, side), border)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """Camera-like view: crop, flip, lighting, blur, low resolution."""
    w, h = img.size
    s = rng.uniform(0.65, 1.0)
    cw, ch = int(w * s), int(h * s)
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    img = img.crop((x, y, x + cw, y + ch))
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.6, 1.3))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.7, 1.2))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
    if rng.random() < 0.5:
        small = rng.randint(80, 160)
        img = img.resize((small, max(1, small * img.size[1] // max(1, img.size[0]))), Image.BILINEAR)
    return img


class Embedder:
    """MobileCLIP2-S0 image encoder with exactly the app's preprocessing."""

    def __init__(self, path: str):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def __call__(self, img: Image.Image) -> np.ndarray:
        x = np.asarray(square(img.convert("RGB")).resize((SIZE, SIZE), Image.BILINEAR), np.float32)
        e = self.sess.run(None, {"image": (x / 255.0).transpose(2, 0, 1)[None]})[0].reshape(-1)
        return e / (np.linalg.norm(e) + 1e-12)


def load_entries():
    entries = []
    for path in sorted(glob.glob(f"{DRAFTS}/*.json")):
        entry = json.load(open(path))
        files = [f"{IMAGES}/{entry['id']}/{im['file']}" for im in entry.get("images", [])]
        files = [f for f in files if os.path.exists(f)]
        if len(files) < 3:
            print(f"skip {entry['id']}: only {len(files)} images on disk")
            continue
        entries.append((entry, files))
    return entries


def topk(probs, labels, k):
    return float((np.argsort(-probs, 1)[:, :k] == labels[:, None]).any(1).mean())


def main():
    rng = random.Random(0)
    embed = Embedder(MODEL)
    entries = load_entries()
    if not entries:
        print("no entries with images yet")
        return 1
    names = [e["id"] for e, _ in entries]
    n = len(entries)

    # Every photo: the original and AUG_EVAL camera-like views.
    X, y, grp, orig = [], [], [], []
    for ei, (entry, files) in enumerate(entries):
        for ii, f in enumerate(files):
            img = Image.open(f).convert("RGB")
            for k, v in enumerate([img] + [augment(img, rng) for _ in range(AUG_EVAL)]):
                X.append(embed(v)); y.append(ei); grp.append(ei * 1000 + ii); orig.append(k == 0)
        print(f"{entry['id']}: {len(files)} photos")
    X, y, grp, orig = map(np.array, (X, y, grp, orig))
    q = ~orig  # queries are augmented views of held-out photos
    ref_mask = np.array([k % (AUG_EVAL + 1) <= AUG_REF for k in range(len(y))])

    # ---- Evaluation: 5 folds, whole photos held out.
    probs = np.zeros((len(y), n))
    best_sim = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, grp):
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X[tr], y[tr])
        probs[te] = clf.predict_proba(X[te])
        refs = X[tr][ref_mask[tr]]
        best_sim[te] = (X[te] @ refs.T).max(1)

    pred = probs[q].argmax(1)
    print(f"\nHeld-out: top-1 {topk(probs[q], y[q], 1) * 100:.1f}%  top-3 {topk(probs[q], y[q], 3) * 100:.1f}%")
    for i, name in enumerate(names):
        m = y[q] == i
        wrong = np.bincount(pred[m & (pred != i)], minlength=n)
        conf = ", ".join(f"{names[j]} ({wrong[j]})" for j in np.argsort(-wrong)[:2] if wrong[j])
        print(f"  {name:22s} {(pred[m] == i).mean() * 100:5.1f}%   mistaken for: {conf or '-'}")

    # ---- "Not in the supported list": similarity threshold keeping 95% of supported earbuds.
    refs_all = X[ref_mask]
    threshold = float(np.percentile(best_sim[q], (1 - KEEP_SUPPORTED) * 100))

    def reject_rate(pattern, views):
        sims = []
        for f in glob.glob(pattern):
            img = Image.open(f).convert("RGB")
            sims += [float((refs_all @ embed(augment(img, rng))).max()) for _ in range(views)]
        return (np.array(sims) < threshold).mean() if sims else None

    unsup = reject_rate(UNSUPPORTED, 4)
    scenes = reject_rate(SCENES, 20)
    print(f"\nSupported-model threshold {threshold:.3f} (keeps {KEEP_SUPPORTED * 100:.0f}% of supported earbuds)")
    if unsup is not None:
        print(f"  unsupported earbuds (removed/) correctly rejected: {unsup * 100:.1f}%")
    if scenes is not None:
        print(f"  non-earbud scenes correctly rejected:              {scenes * 100:.1f}%")

    # ---- Final classifier on everything, and the app files.
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X, y)
    os.makedirs(f"{OUT}/thumbs", exist_ok=True)
    for old in glob.glob(f"{OUT}/thumbs/*.jpg"):  # drop thumbnails of removed models
        os.remove(old)
    catalog = []
    for entry, files in entries:
        slim = {k: v for k, v in entry.items() if k not in ("images", "sources")}
        slim["thumbnail"] = f"thumbs/{entry['id']}.jpg"
        catalog.append(slim)
        square(Image.open(files[0]).convert("RGB")).resize((256, 256), Image.BILINEAR).save(
            f"{OUT}/thumbs/{entry['id']}.jpg", quality=85)
    model_digest = hashlib.sha256(open(MODEL, "rb").read()).hexdigest()[:8]
    json.dump({
        "version": 2,
        "model": {"asset": os.path.basename(MODEL), "digest": model_digest, "inputSize": SIZE,
                  "normalization": "rgb/255", "embeddingDim": int(X.shape[1])},
        "thresholds": {"minSimilarity": round(threshold, 4)},
        "entries": catalog,
    }, open(f"{OUT}/earbuds_catalog.json", "w"), indent=1, ensure_ascii=False)
    with open(f"{OUT}/earbuds_classifier.bin", "wb") as fh:
        fh.write(struct.pack("<ii", n, X.shape[1]))
        fh.write(clf.coef_.astype("<f4").tobytes())
        fh.write(clf.intercept_.astype("<f4").tobytes())
    with open(f"{OUT}/earbuds_embeddings.bin", "wb") as fh:
        fh.write(struct.pack("<ii", int(ref_mask.sum()), X.shape[1]))
        for v, label in zip(X[ref_mask], y[ref_mask]):
            fh.write(struct.pack("<i", int(label)))
            fh.write(v.astype("<f4").tobytes())
    sizes = {f: os.path.getsize(f"{OUT}/{f}") // 1024 for f in
             ("earbuds_catalog.json", "earbuds_classifier.bin", "earbuds_embeddings.bin")}
    print(f"\nWrote {n} entries to {OUT}/: {sizes} KB")


if __name__ == "__main__":
    sys.exit(main())

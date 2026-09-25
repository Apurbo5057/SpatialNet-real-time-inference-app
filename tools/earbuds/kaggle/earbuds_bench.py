"""Earbuds recognition benchmark, runnable on a Kaggle GPU notebook.

Compares feature extractors for telling the 10 catalog earbud models apart, then (optionally)
fine-tunes MobileNetV4 on the photos. Every score is measured on photos the model never saw
(5-fold split by photo), using camera-like augmented views as queries, and reported as top-1
and top-3 accuracy (top-3 matters because Bluetooth makes the final choice among candidates).

MobileNetV4 variants are scored at 224 px, the size the phone runs; "@384" rows are the
models' native resolution, for reference only (about 3x the compute on the phone).

Kaggle setup:
  1. Upload earbuds_kaggle.zip as a PRIVATE Kaggle dataset (it contains web photos).
  2. New notebook -> Accelerator: GPU T4 x2 (or P100) -> Internet: On -> add the dataset.
  3. In a cell:
       !pip install -q timm open_clip_torch
       !python /kaggle/input/<dataset-path>/earbuds_bench.py --data /kaggle/input/<dataset-path> --finetune
  4. Copy the printed table back (results.json is also written to /kaggle/working).

Data layout expected under --data:
  catalog_draft/<id>.json
  web_images/<id>/<file>.jpg
"""

import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FOLDS = 5

MNV4_IN1K = "mobilenetv4_conv_medium.e500_r224_in1k"            # the app today
MNV4_IN12K = "mobilenetv4_conv_medium.e180_r384_in12k"
MNV4_IN12K_FT_IN1K = "mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k"  # drop-in candidate


# ---------------------------------------------------------------- data

def find_root(data):
    """Kaggle may unpack the zip one folder deeper; find the folder with catalog_draft/."""
    for d in [data] + glob.glob(os.path.join(data, "*")) + glob.glob(os.path.join(data, "*", "*")):
        if os.path.isdir(os.path.join(d, "catalog_draft")):
            return d
    raise SystemExit(f"catalog_draft/ not found under {data}")


def load(data):
    root = find_root(data)
    entries = []
    for path in sorted(glob.glob(f"{root}/catalog_draft/*.json")):
        e = json.load(open(path))
        files = [f"{root}/web_images/{e['id']}/{im['file']}" for im in e["images"]]
        files = [f for f in files if os.path.exists(f)]
        if len(files) >= 3:
            entries.append((e["id"], [Image.open(f).convert("RGB") for f in files]))
    print(f"{len(entries)} models, {sum(len(f) for _, f in entries)} photos from {root}")
    return entries


def square(img):
    """Pad to a square with the border colour, like the app's square ROI crop."""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), img.resize((1, 1)).getpixel((0, 0)))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def augment(img, rng):
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


def views(entries, n_aug, seed=0):
    """(image, model index, photo group, is_original) for every photo and its augmented views."""
    rng = random.Random(seed)
    out = []
    for ei, (_, photos) in enumerate(entries):
        for pi, img in enumerate(photos):
            out.append((img, ei, ei * 1000 + pi, True))
            for _ in range(n_aug):
                out.append((augment(img, rng), ei, ei * 1000 + pi, False))
    return out


def topk_hits(scores, labels, k):
    """Fraction of rows whose true label is among the k highest scores."""
    top = np.argsort(-scores, axis=1)[:, :k]
    return float((top == labels[:, None]).any(1).mean())


# ---------------------------------------------------------------- feature extractors

def timm_transform(m, size):
    import timm
    cfg = timm.data.resolve_data_config({}, model=m)
    cfg["crop_pct"] = 1.0
    if size:
        cfg["input_size"] = (3, size, size)
    return timm.data.create_transform(**cfg)


def timm_extractor(name, size=None, **kw):
    import timm
    m = timm.create_model(name, pretrained=True, num_classes=0, **kw).eval().to(DEVICE)
    return m, timm_transform(m, size), m


def clip_extractor(name, pretrained):
    import open_clip
    m, _, pre = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    m = m.eval().to(DEVICE)
    # Report only the image encoder's size: the text encoder never ships to the phone.
    return m.visual, pre, m.encode_image


EXTRACTORS = {
    "mnv4_in1k@224 (app today)": lambda: timm_extractor(MNV4_IN1K, 224),
    "mnv4_in12k@224": lambda: timm_extractor(MNV4_IN12K, 224),
    "mnv4_in12k@384": lambda: timm_extractor(MNV4_IN12K, 384),
    "mnv4_in12k_ft_in1k@224": lambda: timm_extractor(MNV4_IN12K_FT_IN1K, 224),
    "mnv4_in12k_ft_in1k@384": lambda: timm_extractor(MNV4_IN12K_FT_IN1K, 384),
    "dinov2_small@224": lambda: timm_extractor("vit_small_patch14_dinov2.lvd142m", 224, img_size=224),
    "mobileclip2_s0": lambda: clip_extractor("MobileCLIP2-S0", "dfndr2b"),
    "mobileclip_s1": lambda: clip_extractor("MobileCLIP-S1", "datacompdr"),
}


@torch.no_grad()
def embed_all(fn, tf, imgs, bs=64):
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.stack([tf(square(im)) for im in imgs[i:i + bs]]).to(DEVICE)
        v = fn(x).float()
        out.append(torch.nn.functional.normalize(v.flatten(1), dim=1).cpu().numpy())
    return np.concatenate(out)


def score_features(X, y, g, orig, n_models):
    """Top-1/top-3 for nearest-photo matching and for a linear classifier, on unseen photos."""
    q = np.where(~orig)[0]
    per = np.full((len(q), n_models), -1.0)
    for row, k in enumerate(q):
        mask = g != g[k]
        np.maximum.at(per[row], y[mask], X[mask] @ X[k])
    nn1, nn3 = topk_hits(per, y[q], 1), topk_hits(per, y[q], 3)

    probs = np.zeros((len(y), n_models))
    for tr, te in GroupKFold(FOLDS).split(X, y, g):
        clf = LogisticRegression(C=3, max_iter=3000).fit(X[tr], y[tr])
        probs[te] = clf.predict_proba(X[te])
    lin1, lin3 = topk_hits(probs[q], y[q], 1), topk_hits(probs[q], y[q], 3)
    return nn1, nn3, lin1, lin3, probs[q].argmax(1), y[q]


def per_model_report(names, pred, true):
    """Accuracy per model and what it is most often mistaken for."""
    lines = []
    for i, name in enumerate(names):
        mine = true == i
        acc = (pred[mine] == i).mean()
        wrong = np.bincount(pred[mine & (pred != i)], minlength=len(names))
        top = [f"{names[j]} ({wrong[j]})" for j in np.argsort(-wrong)[:2] if wrong[j] > 0]
        lines.append(f"    {name:22s} {acc * 100:5.1f}%   mistaken for: {', '.join(top) or '-'}")
    return "\n".join(lines)


# ---------------------------------------------------------------- fine-tuning

def finetune_cv(entries, backbone, size, head_epochs, full_epochs, repeat=4, seed=0):
    """
    Two-stage fine-tuning per fold, so the random new head cannot wreck the pretrained features:
      1. backbone frozen, only the new 10-class head trains (lr 1e-3);
      2. everything trains, backbone at lr 3e-5 and head at 3e-4, cosine schedule.
    Scored on 8 camera-like views of each held-out photo; returns (top-1, top-3, preds, labels).
    """
    import timm
    from torch.utils.data import DataLoader, Dataset

    class DS(Dataset):
        def __init__(self, items, tf):
            self.items, self.tf = items, tf

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            img, label = self.items[i]
            # DataLoader seeds Python's `random` differently in each worker.
            return self.tf(square(augment(img, random))), label

    photos = [(img, ei, ei * 1000 + pi) for ei, (_, ps) in enumerate(entries) for pi, img in enumerate(ps)]
    y = np.array([p[1] for p in photos])
    g = np.array([p[2] for p in photos])
    all_logits, all_true = [], []
    use_amp = DEVICE == "cuda"

    for fold, (tr, te) in enumerate(GroupKFold(FOLDS).split(np.zeros(len(y)), y, g)):
        torch.manual_seed(seed + fold)
        m = timm.create_model(backbone, pretrained=True, num_classes=len(entries)).to(DEVICE)
        tf = timm_transform(m, size)
        loader = DataLoader(DS([(photos[i][0], photos[i][1]) for i in tr] * repeat, tf),
                            batch_size=32, shuffle=True, num_workers=4 if DEVICE == "cuda" else 0,
                            drop_last=True)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        head = list(m.get_classifier().parameters())
        head_ids = {id(p) for p in head}
        body = [p for p in m.parameters() if id(p) not in head_ids]

        def run(opt, epochs, sched=None):
            for _ in range(epochs):
                m.train()
                for x, t in loader:
                    with torch.autocast("cuda", enabled=use_amp):
                        loss = torch.nn.functional.cross_entropy(m(x.to(DEVICE)), t.to(DEVICE),
                                                                 label_smoothing=0.1)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    if sched:
                        sched.step()

        # Stage 1: head only.
        for p in body:
            p.requires_grad_(False)
        run(torch.optim.AdamW(head, lr=1e-3, weight_decay=0.01), head_epochs)
        # Stage 2: everything, gently.
        for p in body:
            p.requires_grad_(True)
        opt = torch.optim.AdamW([{"params": body, "lr": 3e-5}, {"params": head, "lr": 3e-4}],
                                weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, full_epochs * len(loader)))
        run(opt, full_epochs, sched)

        # Same held-out views for every backbone (fixed seed per fold).
        m.eval()
        rng = random.Random(1000 + fold)
        test = [(augment(photos[i][0], rng), photos[i][1]) for i in te for _ in range(8)]
        logits = []
        with torch.no_grad():
            for i in range(0, len(test), 64):
                x = torch.stack([tf(square(im)) for im, _ in test[i:i + 64]]).to(DEVICE)
                logits.append(m(x).float().cpu().numpy())
        logits = np.concatenate(logits)
        labels = np.array([t for _, t in test])
        all_logits.append(logits)
        all_true.append(labels)
        print(f"    fold {fold + 1}/{FOLDS}: top-1 {topk_hits(logits, labels, 1) * 100:5.1f}%  "
              f"top-3 {topk_hits(logits, labels, 3) * 100:5.1f}%", flush=True)
        del m
        torch.cuda.empty_cache()

    logits, labels = np.concatenate(all_logits), np.concatenate(all_true)
    return topk_hits(logits, labels, 1), topk_hits(logits, labels, 3), logits.argmax(1), labels


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--aug", type=int, default=8, help="augmented views per photo")
    ap.add_argument("--models", nargs="*", default=list(EXTRACTORS))
    ap.add_argument("--finetune", action="store_true", help="also fine-tune MobileNetV4 (in1k and in12k_ft_in1k)")
    ap.add_argument("--head-epochs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20, help="full fine-tuning epochs after the head warm-up")
    args = ap.parse_args()

    print(f"device: {DEVICE}")
    entries = load(args.data)
    names = [e for e, _ in entries]
    allv = views(entries, args.aug)
    imgs = [v[0] for v in allv]
    y = np.array([v[1] for v in allv])
    g = np.array([v[2] for v in allv])
    orig = np.array([v[3] for v in allv])

    results, best = {}, None
    print(f"\n{'features':28s} {'img-enc':>8s} {'dim':>5s}   nearest-photo top1/top3   linear top1/top3")
    for name in args.models:
        t0 = time.time()
        try:
            model, tf, fn = EXTRACTORS[name]()
        except Exception as ex:
            print(f"{name}: failed to load ({str(ex)[:150]})")
            continue
        X = embed_all(fn, tf, imgs)
        nn1, nn3, lin1, lin3, pred, true = score_features(X, y, g, orig, len(entries))
        params = sum(p.numel() for p in model.parameters()) / 1e6
        results[name] = {"nearest_top1": nn1, "nearest_top3": nn3, "linear_top1": lin1, "linear_top3": lin3,
                         "dim": int(X.shape[1]), "image_encoder_params_M": params}
        print(f"{name:28s} {params:7.1f}M {X.shape[1]:5d}   {nn1 * 100:5.1f}% / {nn3 * 100:5.1f}%        "
              f"{lin1 * 100:5.1f}% / {lin3 * 100:5.1f}%   ({time.time() - t0:.0f}s)", flush=True)
        if best is None or lin1 > best[0]:
            best = (lin1, name, pred, true)
        del model
        torch.cuda.empty_cache()

    if best:
        print(f"\nPer model, best linear classifier ({best[1]}):\n{per_model_report(names, best[2], best[3])}")

    if args.finetune:
        for label, backbone in [("finetune mnv4_in1k@224", MNV4_IN1K),
                                ("finetune mnv4_in12k_ft_in1k@224", MNV4_IN12K_FT_IN1K)]:
            print(f"\n{label} ({args.head_epochs} head + {args.epochs} full epochs)…", flush=True)
            t0 = time.time()
            top1, top3, pred, true = finetune_cv(entries, backbone, 224, args.head_epochs, args.epochs)
            results[label] = {"top1": top1, "top3": top3}
            print(f"{label:32s} top-1 {top1 * 100:5.1f}%  top-3 {top3 * 100:5.1f}%  ({time.time() - t0:.0f}s)")
            print(per_model_report(names, pred, true), flush=True)

    out = "/kaggle/working/results.json" if os.path.isdir("/kaggle/working") else "results.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

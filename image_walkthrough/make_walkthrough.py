"""Save visible stages of the Android image pipeline using the real ONNX model.

Run from the repository root: python image_walkthrough/make_walkthrough.py
The source is a still image, so --rotation defaults to 0. Set it to the
ImageProxy rotation reported on a phone to simulate that particular frame.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "ROI_issues" / "test.jpg"
MODEL = ROOT / "ROIModel" / "spatialnet_v4.onnx"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def box_pixels(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = box
    return tuple(round(v * scale) for v, scale in zip(
        (max(0, cx - bw / 2), max(0, cy - bh / 2),
         min(1, cx + bw / 2), min(1, cy + bh / 2)),
        (width, height, width, height),
    ))


def draw_detection(image: Image.Image, result: dict, arrow_length: int,
                   bottom_bar: bool = False) -> Image.Image:
    out = image.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    if result["is_pointing"]:
        if result["is_hand"]:
            d.rectangle(box_pixels(result["hand"], w, h), outline="#2196f3", width=4)
            d.text((box_pixels(result["hand"], w, h)[0] + 5,
                    box_pixels(result["hand"], w, h)[1] + 5), "HAND", fill="#2196f3", font=font(24))
        if result["is_roi"]:
            d.rectangle(box_pixels(result["roi"], w, h), outline="#4caf50", width=4)
            d.text((box_pixels(result["roi"], w, h)[0] + 5,
                    box_pixels(result["roi"], w, h)[1] + 5), "ROI", fill="#4caf50", font=font(24))
        x, y = result["fingertip"][0] * w, result["fingertip"][1] * h
        d.ellipse((x - 13, y - 13, x + 13, y + 13), outline="red", width=4)
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill="red")
        d.text((x + 16, y - 8), "Fingertip", fill="red", font=font(24))
        sin_a, cos_a = result["angle"]
        end = (x + arrow_length * cos_a, y + arrow_length * sin_a)
        d.line((x, y, *end), fill="yellow", width=5)
        a = math.atan2(sin_a, cos_a)
        for side in (-1, 1):
            head = a + math.pi + side * math.pi / 6
            d.line((*end, end[0] + 18 * math.cos(head),
                    end[1] + 18 * math.sin(head)), fill="yellow", width=5)
    elif result["is_hand"]:
        d.rectangle(box_pixels(result["hand"], w, h), outline="red", width=4)
        d.text((box_pixels(result["hand"], w, h)[0] + 5,
                box_pixels(result["hand"], w, h)[1] + 5), "NON-POINTING", fill="red", font=font(24))
    if bottom_bar:
        d.rectangle((0, h - 50, w, h), fill=(35, 35, 35))
        active = (result["roi_probability"] if result["is_roi"] else
                  result["point_probability"] if result["is_pointing"] else
                  result["hand_probability"] if result["is_hand"] else 0)
        d.text((16, h - 39), f'{result["category"]} | {active:.0%}', fill="white", font=font(27))
    return out


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def preview_fill_center(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Illustrate aspect-preserving FILL_CENTER; real Preview is a separate stream."""
    return ImageOps.fit(image, size, method=Image.Resampling.BILINEAR, centering=(0.5, 0.5))


def preview_fit_with_geometry(image: Image.Image, size: tuple[int, int], result: dict) -> tuple[Image.Image, dict, dict]:
    """Full analysis frame centered on black, with model coordinates mapped to it."""
    w, h = size
    scale = min(w / image.width, h / image.height)
    dw, dh = round(image.width * scale), round(image.height * scale)
    ox, oy = (w - dw) // 2, (h - dh) // 2
    fitted = Image.new("RGB", size, "black")
    fitted.paste(image.resize((dw, dh), Image.Resampling.BILINEAR), (ox, oy))

    def point(p: list[float]) -> list[float]:
        return [(ox + p[0] * dw) / w, (oy + p[1] * dh) / h]

    def box(b: list[float]) -> list[float]:
        x, y = point(b[:2])
        return [x, y, b[2] * dw / w, b[3] * dh / h]

    mapped = {**result, "hand": box(result["hand"]), "roi": box(result["roi"]),
              "fingertip": point(result["fingertip"])}
    geometry = {"canvas_size": [w, h], "displayed_image_size": [dw, dh],
                "displayed_image_offset": [ox, oy], "scale": scale}
    return fitted, mapped, geometry


def labelled(image: Image.Image, title: str, subtitle: str,
             panel: tuple[int, int] = (470, 360)) -> Image.Image:
    tile = Image.new("RGB", panel, "#20242a")
    fit = ImageOps.contain(image.convert("RGB"), (panel[0] - 24, panel[1] - 80))
    tile.paste(fit, ((panel[0] - fit.width) // 2, 66 + (panel[1] - 80 - fit.height) // 2))
    d = ImageDraw.Draw(tile)
    d.text((12, 9), title, fill="white", font=font(22))
    d.text((12, 38), subtitle, fill="#b6c2ce", font=font(16))
    return tile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=ROOT / "image_walkthrough" / "output")
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--preview-width", type=int, default=720)
    parser.add_argument("--preview-height", type=int, default=1280)
    args = parser.parse_args()
    if args.preview_width <= 0 or args.preview_height <= 0:
        parser.error("preview width and height must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    source = Image.open(args.image).convert("RGB")
    source.save(args.output / "01_source.png")
    # Android's positive Matrix rotation is clockwise in image coordinates.
    rotated = source.rotate(-args.rotation, expand=True)
    rotated.save(args.output / "02_rotated_analysis.png")
    resized = rotated.resize((160, 160), Image.Resampling.BILINEAR)
    resized.save(args.output / "03_model_rgb_160x160.png")
    scaled = np.asarray(resized, dtype=np.float32) / 255.0
    normalized = (scaled - MEAN) / STD
    tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
    # Normalized floats are not directly viewable as RGB. This maps -2.5..2.5
    # to 0..255 per channel solely so a human can see an illustrative image.
    visual = np.clip((normalized + 2.5) / 5.0 * 255, 0, 255).astype(np.uint8)
    Image.fromarray(visual, "RGB").save(args.output / "04_normalized_visual_only.png")
    np.save(args.output / "04_actual_model_tensor.npy", tensor)

    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    raw = {info.name: values[0].tolist() for info, values in zip(session.get_outputs(), outputs)}
    hp, pp, rp = (sigmoid(raw[key][0]) for key in ("hand_conf", "point_conf", "roi_conf"))
    hand, pointing, roi = hp >= 0.5, hp >= 0.5 and pp >= 0.5, hp >= 0.5 and pp >= 0.5 and rp >= 0.5
    category = "NO HAND" if not hand else "NON-POINTING" if not pointing else "POINTING-EMPTY" if not roi else "POINTING"
    result = {**raw, "hand_probability": hp, "point_probability": pp, "roi_probability": rp,
              "is_hand": hand, "is_pointing": pointing, "is_roi": roi, "category": category,
              "source_size": list(source.size), "rotated_analysis_size": list(rotated.size),
              "model_rgb_size": [160, 160], "tensor_shape": list(tensor.shape),
              "illustrative_preview_size": [args.preview_width, args.preview_height],
              "saved_capture_size": list(rotated.size), "rotation_degrees": args.rotation}
    (args.output / "05_model_output.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    overlay = draw_detection(rotated, result, arrow_length=90)
    overlay.save(args.output / "06_prediction_on_analysis.png")
    capture = draw_detection(rotated, result, arrow_length=90, bottom_bar=True)
    capture.save(args.output / "07_capture_simulation.jpg", quality=95)
    # This is a geometry illustration. The app's Preview use case can deliver
    # a different source frame, and a still image cannot reveal its dimensions.
    preview_size = args.preview_width, args.preview_height
    cropped = preview_fill_center(rotated, preview_size)
    cropped.save(args.output / "08_fill_center_preview_example.png")
    naive = draw_detection(cropped, result, arrow_length=80)
    naive.save(args.output / "09_app_overlay_on_preview_example.png")

    fixed_dir = args.output.parent / "fixed_output"
    fixed_dir.mkdir(exist_ok=True)
    fit, mapped, geometry = preview_fit_with_geometry(rotated, preview_size, result)
    fit.save(fixed_dir / "07_full_frame_preview.png")
    fixed = draw_detection(fit, mapped, arrow_length=80)
    fixed.save(fixed_dir / "08_aligned_overlay_preview.png")
    (fixed_dir / "preview_geometry.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")

    fixed_frames = [
        (source, "1  SOURCE", f"{source.width} x {source.height}"),
        (rotated, "2  ROTATED ANALYSIS", f"{rotated.width} x {rotated.height}"),
        (resized, "3  MODEL RGB", "160 x 160; squeezed"),
        (Image.fromarray(visual, "RGB"), "4  MODEL TENSOR VIEW", "160 x 160; visualized floats"),
        (overlay, "5  MODEL RESULT", f"{category}; original coordinates"),
        (fit, "6  NEW LIVE FRAME", "Whole source visible; black margins"),
        (fixed, "7  NEW LIVE OVERLAY", "Same fit transform as the frame"),
        (capture, "8  CAPTURE", f"{capture.width} x {capture.height}; no margins"),
    ]
    fixed_sheet = Image.new("RGB", (940, 1440), "#15191e")
    for i, (im, title, subtitle) in enumerate(fixed_frames):
        fixed_sheet.paste(labelled(im, title, subtitle), ((i % 2) * 470, (i // 2) * 360))
    fixed_sheet.save(fixed_dir / "00_fixed_flow.jpg", quality=94)

    frames = [
        (source, "1  SOURCE", f"{source.width} x {source.height} pixels"),
        (rotated, "2  ROTATED ANALYSIS", f"{rotated.width} x {rotated.height} pixels; {args.rotation} deg"),
        (resized, "3  MODEL RGB", "160 x 160; whole frame stretched"),
        (Image.fromarray(visual, "RGB"), "4  NORMALIZED VIEW", "160 x 160; display mapping only"),
        (overlay, "5  MODEL RESULT", f"{category}; {rotated.width} x {rotated.height}"),
        (capture, "6  CAPTURE", f"JPEG; {capture.width} x {capture.height}"),
        (cropped, "7  PREVIEW EXAMPLE", f"FILL_CENTER; {args.preview_width} x {args.preview_height}"),
        (naive, "8  APP OVERLAY EXAMPLE", "May misalign after center crop"),
    ]
    sheet = Image.new("RGB", (470 * 2, 360 * 4), "#15191e")
    for i, (im, title, subtitle) in enumerate(frames):
        sheet.paste(labelled(im, title, subtitle), ((i % 2) * 470, (i // 2) * 360))
    sheet.save(args.output / "00_all_steps.jpg", quality=94)
    actual_preview = ROOT / "ROI_issues" / "preview1.jpg"
    actual_capture = ROOT / "ROI_issues" / "saved_capture1.jpg"
    if actual_preview.exists() and actual_capture.exists():
        phone_preview = Image.open(actual_preview).convert("RGB")
        phone_capture = Image.open(actual_capture).convert("RGB")
        comparison = Image.new("RGB", (940, 420), "#15191e")
        comparison.paste(labelled(phone_preview, "ACTUAL PHONE SCREENSHOT",
                                  f"{phone_preview.width} x {phone_preview.height}; includes UI",
                                  (470, 420)), (0, 0))
        comparison.paste(labelled(phone_capture, "ACTUAL APP CAPTURE",
                                  f"{phone_capture.width} x {phone_capture.height}; analysis frame",
                                  (470, 420)), (470, 0))
        comparison.save(args.output / "10_actual_phone_examples.jpg", quality=94)
    print(f"Saved {args.output} | input {source.size} | tensor {tensor.shape} | {category}")


if __name__ == "__main__":
    main()
